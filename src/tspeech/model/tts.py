import matplotlib

matplotlib.use("Agg")

from typing import Optional, Final

import lightning as pl
import matplotlib
from torch import nn
import torch
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import functional as F

from tspeech.data.tts import TTSBatch
from tspeech.model.tacotron2 import Tacotron2
from tspeech.model.tacotron2 import GST
from tspeech.model.bert_gst_encoder import BERTGSTEncoder
from tspeech.model.rl_gst_policy_option1 import RLGSTPolicy
from tspeech.model.htmodel import HTModel
from tspeech.vocoder import HiFiGANVocoder

class TTSModel(pl.LightningModule):
    def __init__(
        self,
        num_chars: int,
        encoded_dim: int,
        encoder_kernel_size: int,
        num_mels: int,
        prenet_dim: int,
        att_rnn_dim: int,
        att_dim: int,
        rnn_hidden_dim: int,
        postnet_dim: int,
        dropout: float,
        gst_enabled: bool,
        speaker_tokens_enabled: bool,
        speaker_count: int = 1,
        max_len_override: Optional[int] = None,
        teacher_forcing_dropout: float = 0.0,

        # Optional flags for RL / BERT / HuBERT integration.
        use_rl_training: bool = False,
        use_bert_gst: bool = False,
        bert_model_name: str = "bert-base-uncased",
        hubert_model_name: str = "facebook/hubert-base-ls960",
        hubert_checkpoint_path: Optional[str] = None,
        rl_temperature: float = 1.0,
        rl_entropy_coef: float = 0.01,
        use_vocoder: bool = False,
        vocoder_checkpoint_dir: Optional[str] = None,
        save_audio_dir: Optional[str] = None,
        save_audio_every_n_steps: int = 100,
        learning_rate: Optional[float] = None,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.encoded_dim = encoded_dim
        self.speaker_tokens = speaker_tokens_enabled
        self.max_len_override = max_len_override

        self.teacher_forcing_dropout = teacher_forcing_dropout

        # RL / BERT / HuBERT configuration (not yet changing behavior).
        self.use_rl_training = use_rl_training
        self.use_bert_gst = use_bert_gst
        self.bert_model_name = bert_model_name
        self.hubert_model_name = hubert_model_name
        self.hubert_checkpoint_path = hubert_checkpoint_path
        self.rl_temperature = rl_temperature
        self.rl_entropy_coef = rl_entropy_coef
        self.use_vocoder = use_vocoder
        self.vocoder_checkpoint_dir = vocoder_checkpoint_dir
        self.save_audio_dir = save_audio_dir
        self.save_audio_every_n_steps = save_audio_every_n_steps

        extra_encoded_dim = 0
        self.gst: nn.Module | None = None
        self.gst_enabled: Final[bool] = gst_enabled
        if self.gst_enabled:
            extra_encoded_dim += encoded_dim // 2
            self.gst = GST(out_dim=encoded_dim // 2)

        self.tacotron2 = Tacotron2(
            num_chars=num_chars,
            encoded_dim=encoded_dim,
            encoder_kernel_size=encoder_kernel_size,
            num_mels=num_mels,
            prenet_dim=prenet_dim,
            att_rnn_dim=att_rnn_dim,
            att_dim=att_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            postnet_dim=postnet_dim,
            dropout=dropout,
            speaker_tokens_enabled=speaker_tokens_enabled,
            speaker_count=speaker_count,
            extra_encoded_dim=extra_encoded_dim,
        )

        # Optional RL / BERT / HuBERT / vocoder components.
        self.bert_gst_encoder: Optional[BERTGSTEncoder] = None
        self.rl_policy: Optional[RLGSTPolicy] = None
        self.hubert_classifier: Optional[HTModel] = None
        self.vocoder: Optional[HiFiGANVocoder] = None
        self._rl_log_probs: Optional[Tensor] = None
        self._rl_entropy: Optional[Tensor] = None
        self._rl_style_proj: Optional[nn.Linear] = None

        if self.use_bert_gst or self.use_rl_training:
            # BERT encoder that maps text → pooled embeddings.
            self.bert_gst_encoder = BERTGSTEncoder(
                bert_model_name=self.bert_model_name,
                gst_token_num=10,
                freeze_bert=True,
            )

            # RL policy over GST token weights, driven by BERT embeddings.
            bert_hidden_size = self.bert_gst_encoder.bert.config.hidden_size
            self.rl_policy = RLGSTPolicy(
                bert_hidden_size=bert_hidden_size,
                gst_token_num=10,
                temperature=self.rl_temperature,
            )

            # HuBERT trustworthiness classifier (frozen).
            if self.hubert_checkpoint_path is None:
                raise ValueError("hubert_checkpoint_path is required when use_rl_training=True")
            self.hubert_classifier = HTModel.load_from_checkpoint(
                self.hubert_checkpoint_path,
                hubert_model_name=self.hubert_model_name,
                trainable_layers=0,
            )
            self.hubert_classifier.eval()
            for p in self.hubert_classifier.parameters():
                p.requires_grad = False

            # Vocoder for converting mels to waveforms during RL scoring.
            # Required for RL training (used to generate waveforms for HuBERT scoring).
            self.vocoder = HiFiGANVocoder(
                checkpoint_dir=self.vocoder_checkpoint_dir or "UNIVERSAL_V1",
            )
            self.vocoder.eval()

        # In RL mode, Tacotron2 + GST should act as fixed, pre-trained components.
        # Freeze their parameters so only the RL head and related modules train.
        if self.use_rl_training:
            for p in self.tacotron2.parameters():
                p.requires_grad = False
            if self.gst is not None:
                for p in self.gst.parameters():
                    p.requires_grad = False
            # Ensure they run in eval mode by default (no dropout/stat updates).
            self.tacotron2.eval()
            if self.gst is not None:
                self.gst.eval()

    def _get_style_embedding_from_mel(
        self,
        mel_spectrogram_style: Optional[Tensor],
        mel_spectrogram_style_len: Optional[Tensor],
    ) -> Optional[Tensor]:
        """Original GST path: reference mel → GST → style (batch, 1, dim)."""
        if not self.gst_enabled or self.gst is None:
            return None
        if mel_spectrogram_style is None or mel_spectrogram_style_len is None:
            raise ValueError("mel_spectrogram_style and mel_spectrogram_style_len are required when gst_enabled=True")
        return self.gst(mel_spectrogram_style, mel_spectrogram_style_len)

    def _get_style_embedding_from_bert(self, text: Optional[list[str]]) -> Optional[Tensor]:
        """
        RL path: text → BERT → RL policy → GST weights → style using checkpoint GST tokens.

        Returns style of shape (batch, 1, encoded_dim//2) to match original GST output.
        """
        if not (self.use_rl_training and self.use_bert_gst):
            return None
        if text is None:
            raise ValueError("text is required when use_bert_gst=True and use_rl_training=True")
        if self.bert_gst_encoder is None or self.rl_policy is None or self.gst is None:
            raise RuntimeError("BERT / RL policy / GST not initialized for RL training")

        # BERT pooled embeddings (batch, hidden)
        bert_embeddings = self.bert_gst_encoder.get_bert_embeddings(text)
        # Use deterministic sampling during validation/inference, stochastic during training
        gst_weights, log_probs, entropy = self.rl_policy(
            bert_embeddings,
            deterministic=not self.training,
        )
        self._rl_log_probs = log_probs
        self._rl_entropy = entropy

        # Use learned GST tokens from checkpoint (stl.embed) as a token matrix.
        tokens = torch.tanh(self.gst.stl.embed)  # (token_num, token_dim_per_head)
        style_vec = torch.matmul(gst_weights, tokens)  # (batch, token_dim_per_head)

        # Project to encoded_dim//2 if needed to match Tacotron2 extra_encoded_dim.
        target_dim = self.encoded_dim // 2
        token_dim = style_vec.shape[-1]
        if token_dim != target_dim:
            if self._rl_style_proj is None:
                # Create projection on the same device as the current style tensor.
                self._rl_style_proj = nn.Linear(token_dim, target_dim)
                self._rl_style_proj.to(style_vec.device)
            style_vec = self._rl_style_proj(style_vec)

        # Shape expected by Tacotron2: (batch, 1, dim)
        return style_vec.unsqueeze(1)

    def forward(
        self,
        chars_idx: Tensor,
        chars_idx_len: Tensor,
        teacher_forcing_dropout: float,
        teacher_forcing: bool = True,
        mel_spectrogram: Optional[Tensor] = None,
        mel_spectrogram_len: Optional[Tensor] = None,
        speaker_id: Optional[Tensor] = None,
        max_len_override: Optional[int] = None,
        mel_spectrogram_style: Optional[Tensor] = None,
        mel_spectrogram_style_len: Optional[Tensor] = None,
        text: Optional[list[str]] = None,
    ):
        style: Tensor | None = None

        if self.gst is not None:
            if self.use_rl_training and self.use_bert_gst:
                # RL path: use BERT + policy to select GST tokens
                style = self._get_style_embedding_from_bert(text)
            else:
                # Original GST path: use reference mel spectrogram
                style = self._get_style_embedding_from_mel(mel_spectrogram_style, mel_spectrogram_style_len)
        
        # Detach style if RL training to prevent gradients flowing through Tacotron2
        if self.use_rl_training and style is not None:
            style = style.detach()
        
        with torch.no_grad():
            return self.tacotron2(
                chars_idx=chars_idx,
                chars_idx_len=chars_idx_len,
                teacher_forcing=teacher_forcing,
                teacher_forcing_dropout=teacher_forcing_dropout,
                mel_spectrogram=mel_spectrogram,
                mel_spectrogram_len=mel_spectrogram_len,
                speaker_id=speaker_id,
                max_len_override=max_len_override,
                encoded_extra=style,
            )

    def validation_step(self, batch: TTSBatch, batch_idx):
        mel_spectrogram, mel_spectrogram_post, gate, alignment = self(
            chars_idx=batch.chars_idx,
            chars_idx_len=batch.chars_idx_len,
            teacher_forcing=True,
            teacher_forcing_dropout=0.0,
            speaker_id=batch.speaker_id,
            mel_spectrogram=batch.mel_spectrogram,
            mel_spectrogram_len=batch.mel_spectrogram_len,
            mel_spectrogram_style=batch.mel_spectrogram,
            mel_spectrogram_style_len=batch.mel_spectrogram_len,
            text=batch.text,
        )

        gate_loss = F.binary_cross_entropy_with_logits(gate, batch.gate)
        mel_loss = F.mse_loss(mel_spectrogram, batch.mel_spectrogram)
        mel_post_loss = F.mse_loss(mel_spectrogram_post, batch.mel_spectrogram)

        loss = gate_loss + mel_loss + mel_post_loss
        self.log(
            "val_mel_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )

        mel_spectrogram_len = batch.mel_spectrogram_len
        chars_idx_len = batch.chars_idx_len

        mel_spectrogram = batch.mel_spectrogram
        mel_spectrogram_pred = mel_spectrogram_post

        out = {
            "mel_spectrogram_pred": mel_spectrogram_pred[0, : mel_spectrogram_len[0]],
            "mel_spectrogram": mel_spectrogram[0, : mel_spectrogram_len[0]],
            "alignment": alignment[0, : mel_spectrogram_len[0], : chars_idx_len[0]],
            "gate": batch.gate[0],
            "gate_pred": gate[0],
        }

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )

        out["loss"] = loss
        return out

    def training_step(self, batch, batch_idx):
        if self.use_rl_training:
            self.tacotron2.eval()

        mel_spectrogram, mel_spectrogram_post, gate, _ = self(
            chars_idx=batch.chars_idx,
            chars_idx_len=batch.chars_idx_len,
            teacher_forcing=True,
            teacher_forcing_dropout=self.teacher_forcing_dropout,
            speaker_id=batch.speaker_id,
            mel_spectrogram=batch.mel_spectrogram,
            mel_spectrogram_len=batch.mel_spectrogram_len,
            mel_spectrogram_style=batch.mel_spectrogram,
            mel_spectrogram_style_len=batch.mel_spectrogram_len,
            text=batch.text,
        )

        gate_loss = F.binary_cross_entropy_with_logits(gate, batch.gate)
        mel_loss = F.mse_loss(mel_spectrogram, batch.mel_spectrogram)
        mel_post_loss = F.mse_loss(mel_spectrogram_post, batch.mel_spectrogram)
        tts_per_sample = (
            F.binary_cross_entropy_with_logits(gate, batch.gate, reduction="none")
            + F.mse_loss(mel_spectrogram, batch.mel_spectrogram, reduction="none")
            + F.mse_loss(mel_spectrogram_post, batch.mel_spectrogram, reduction="none")
        )
        tts_per_sample = tts_per_sample.reshape(tts_per_sample.shape[0], -1).mean(dim=1)
        tts_loss = gate_loss + mel_loss + mel_post_loss
        
        rl_loss_for_logging = None
        loss = tts_loss

        if self.use_rl_training:
            if self.vocoder is None:
                raise RuntimeError("Vocoder is required when use_rl_training=True")
            # Detach mel before vocoder to prevent gradients flowing through vocoder
            mel_for_vocoder = mel_spectrogram_post.detach()
            self.vocoder.eval()
            with torch.no_grad():
                ref_waveforms = self.vocoder(batch.mel_spectrogram)
            with torch.no_grad():
                waveforms = self.vocoder(mel_for_vocoder)
            self._save_audio_if_needed(waveforms, batch_idx)
            rl_loss_for_logging = self.compute_rl_loss(
                waveforms,
                ref_waveforms,
                tts_per_sample,
                sample_rate=22050,
            )
            loss = rl_loss_for_logging
        
        self.log(
            "training_gate_loss",
            gate_loss.detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )
        self.log(
            "training_mel_loss",
            mel_loss.detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )
        self.log(
            "training_mel_post_loss",
            mel_post_loss.detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )
        self.log(
            "training_loss",
            loss.detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )
        if rl_loss_for_logging is not None:
            self.log(
                "training_rl_loss",
                rl_loss_for_logging.detach(),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=mel_spectrogram.shape[0],
            )

        return loss

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx > 0:
            return

        with matplotlib.rc_context({"backend": "Agg"}):
            self.logger.experiment.add_figure(
                "val_mel_spectrogram",
                plot_spectrogram_to_numpy(outputs["mel_spectrogram"].cpu().T.numpy()),
                self.global_step,
                close=True,
            )
            self.logger.experiment.add_figure(
                "val_mel_spectrogram_predicted",
                plot_spectrogram_to_numpy(
                    outputs["mel_spectrogram_pred"].cpu().swapaxes(0, 1).numpy()
                ),
                self.global_step,
                close=True,
            )
            self.logger.experiment.add_figure(
                "val_alignment",
                plot_alignment_to_numpy(
                    outputs["alignment"].cpu().swapaxes(0, 1).numpy()
                ),
                self.global_step,
                close=True,
            )
            self.logger.experiment.add_figure(
                "val_gate",
                plot_gate_outputs_to_numpy(
                    outputs["gate"].cpu().squeeze().numpy(),
                    torch.sigmoid(outputs["gate_pred"]).squeeze().cpu().numpy(),
                ),
                self.global_step,
                close=True,
            )

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        text = batch.text

        mel_spectrogram, mel_spectrogram_post, gate, alignment = self(
            chars_idx=batch.chars_idx,
            chars_idx_len=batch.chars_idx_len,
            teacher_forcing=False,
            speaker_id=batch.speaker_id,
            mel_spectrogram=batch.mel_spectrogram,
            mel_spectrogram_len=batch.mel_spectrogram_len,
            max_len_override=self.max_len_override,
            mel_spectrogram_style=batch.mel_spectrogram,
            mel_spectrogram_style_len=batch.mel_spectrogram_len,
            text=text,
        )

        return mel_spectrogram, mel_spectrogram_post, gate, alignment, text



    # === RL helpers (ported from rl_gst_tts) =====================================================

    def _save_audio_if_needed(self, waveforms: Tensor, batch_idx: int):
        if not (self.save_audio_dir and batch_idx % self.save_audio_every_n_steps == 0):
            return
        import soundfile as sf
        from pathlib import Path
        import numpy as np

        for i in range(waveforms.shape[0]):
            w = waveforms[i].detach().cpu().numpy()
            w = w / (np.abs(w).max() + 1e-8)
            sf.write(str(Path(self.save_audio_dir) / f"epoch_{self.current_epoch}_step_{batch_idx}_sample_{i}.wav"), w, 22050)

    def _hubert_logits_and_scores(self, waveforms: Tensor, lengths: Tensor, sample_rate: int, device: torch.device):
        """Waveforms (batch, T) + lengths → resample to 16k, HuBERT → logits and scores (batch,)."""
        if sample_rate != 16000:
            import torchaudio
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(device)
            wav_16k = resampler(waveforms)
            len_16k = (lengths.float() * 16000 / sample_rate).long().clamp(min=1)
        else:
            wav_16k = waveforms
            len_16k = lengths
        seq_len = wav_16k.shape[1]
        mask = (torch.arange(seq_len, device=device)[None, :] < len_16k[:, None]).long()
        with torch.no_grad():
            logits = self.hubert_classifier(wav=wav_16k, mask=mask)
        scores = torch.sigmoid(logits).squeeze(-1)
        return logits, scores

    def compute_rl_loss(
        self,
        waveforms: Tensor,
        ref_waveforms: Tensor,
        tts_loss_per_sample: Tensor,
        sample_rate: int = 22050,
    ) -> Tensor:
        """REINFORCE loss. Reward to minimize = tts_loss + MSE(gen_scores, ref_scores). Advantage = -reward."""
        if self.hubert_classifier is None:
            raise ValueError("HuBERT classifier not initialized")

        device = waveforms.device
        # Detach waveforms to prevent gradients flowing through HuBERT
        waveforms = waveforms.detach()
        lengths = torch.full((waveforms.shape[0],), waveforms.shape[1], device=device, dtype=torch.long)
        gen_logits, gen_scores = self._hubert_logits_and_scores(waveforms, lengths, sample_rate, device)

        ref_waveforms = ref_waveforms
        ref_lengths = torch.full((ref_waveforms.shape[0],), ref_waveforms.shape[1], device=device, dtype=torch.long)
        _, ref_scores = self._hubert_logits_and_scores(ref_waveforms, ref_lengths, 22050, device)

        gen_scores = gen_scores.view(-1)
        ref_scores = ref_scores.view(-1)
        mse_scores = (gen_scores - ref_scores).pow(2)
        reward_to_minimize = tts_loss_per_sample.detach() + mse_scores
        advantages = -reward_to_minimize

        self.log("rl_ref_score_mean", ref_scores.mean(), on_step=True, on_epoch=True)
        self.log("rl_reward_to_minimize", reward_to_minimize.mean(), on_step=True, on_epoch=True)
        self.log("rl_mse_gen_ref", mse_scores.mean(), on_step=True, on_epoch=True)

        reinforce_loss = -(self._rl_log_probs * advantages.detach()).mean()
        entropy_bonus = -self.rl_entropy_coef * self._rl_entropy.mean()
        total = reinforce_loss + entropy_bonus

        self.log("rl_advantage_mean", advantages.mean().detach(), on_step=True, on_epoch=True)
        self.log("rl_entropy_mean", self._rl_entropy.mean().detach(), on_step=True, on_epoch=True)
        self.log("rl_reinforce_loss", reinforce_loss.detach(), on_step=True, on_epoch=True)
        self.log("rl_entropy_bonus", entropy_bonus.detach(), on_step=True, on_epoch=True)
        return total

    def configure_optimizers(self):
        from torch.optim import Adam

        params = [p for p in self.parameters() if p.requires_grad]
        # Learning rate must be set via config (e.g. model.learning_rate or optimizer.init_args.lr) or argparse (--learning_rate).
        lr = getattr(self.hparams, "learning_rate", None)
        if lr is None:
            raise ValueError(
                "learning_rate must be set via config (model.learning_rate or optimizer.init_args.lr) or CLI (--model.learning_rate=1e-4) or argparse (--learning_rate)"
            )
        return Adam(params, lr=lr)


def plot_spectrogram_to_numpy(spectrogram) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Channels")
    fig.tight_layout()

    return fig


def plot_alignment_to_numpy(alignment, info=None) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Encoder timestep")
    fig.tight_layout()
    return fig


def plot_gate_outputs_to_numpy(gate_targets, gate_outputs) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(
        range(len(gate_targets)),
        gate_targets,
        alpha=0.5,
        color="green",
        marker="+",
        s=1,
        label="target",
    )
    ax.scatter(
        range(len(gate_outputs)),
        gate_outputs,
        alpha=0.5,
        color="red",
        marker=".",
        s=1,
        label="predicted",
    )

    ax.set_xlabel("Frames (Green target, Red predicted)")
    ax.set_ylabel("Gate State")
    fig.tight_layout()

    return fig
