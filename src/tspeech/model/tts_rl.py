from pathlib import Path
from typing import Final, Optional

import tspeech._torchvision_first  # noqa: F401

import torch
import torchaudio.functional as ta_f
import yaml
from torch import nn, Tensor
from torch.nn import functional as F

import lightning as pl
from tspeech.model.bert_gst_encoder import BERTEncoder
from tspeech.model.rl_gst_policy_option1 import RLGSTPolicy
from tspeech.model.htmodel import HTModel
from tspeech.vocoder import HiFiGANVocoder
from tspeech.model.tts import TTSModel

# Match ``generate_and_score_tactron`` (HiFi-GAN output vs HuBERT input).
HIFI_GAN_SAMPLE_RATE: Final[int] = 22050
HUBERT_SAMPLE_RATE: Final[int] = 16000
# Tacotron mel / STFT rate when no ``tacotron_config_path`` (typical ``config.yaml`` default).
_DEFAULT_TACOTRON_SAMPLE_RATE: Final[int] = 16000


class TTSRLModel(pl.LightningModule):
    def __init__(
        self,
        tts_checkpoint_path: str,
        vocoder_checkpoint_dir: str,
        hubert_checkpoint_path: Optional[str] = None,
        bert_model_name: str = "bert-base-uncased",
        hubert_model_name: str = "facebook/hubert-base-ls960",
        rl_temperature: float = 1,
        rl_entropy_coef: float = 0.005,
        save_audio_dir: Optional[str] = None,
        save_audio_every_n_steps: int = 100,
        tacotron_config_path: Optional[str] = None,
    ):
        super().__init__()

        # Always load the base TTS module onto CPU first. The parent Lightning module
        # (or calling script) can move the full model to GPU later.
        #
        # This avoids accidental CUDA OOM during checkpoint deserialization when a caller
        # intends to run on CPU (or when GPU memory is fragmented).
        self.tts = TTSModel.load_from_checkpoint(tts_checkpoint_path, map_location="cpu")

        if self.tts.gst is None:
            raise Exception("oh no")

        if tacotron_config_path:
            cfg_path = Path(tacotron_config_path).expanduser()
            if not cfg_path.is_file():
                raise FileNotFoundError(f"Tacotron config YAML not found: {cfg_path}")
            with cfg_path.open() as f:
                tacotron_cfg = yaml.safe_load(f)
            tacotron_sample_rate = int(tacotron_cfg["data"]["sample_rate"])
        else:
            tacotron_sample_rate = int(_DEFAULT_TACOTRON_SAMPLE_RATE)

        self._tacotron_sample_rate = tacotron_sample_rate
        self._vocoder_sample_rate = int(HIFI_GAN_SAMPLE_RATE)
        self._hubert_sample_rate = int(HUBERT_SAMPLE_RATE)
        if tacotron_sample_rate == self._vocoder_sample_rate:
            self.vocoder_mel_time_scale = 1.0
        else:
            self.vocoder_mel_time_scale = float(self._vocoder_sample_rate) / float(tacotron_sample_rate)

        self.rl_entropy_coef = rl_entropy_coef
        self.save_audio_dir = save_audio_dir
        self.save_audio_every_n_steps = save_audio_every_n_steps

        # BERT encoder that maps text → pooled embeddings.
        self.bert_gst_encoder = BERTEncoder(
            bert_model_name=bert_model_name,
            freeze_bert=True,
        )

        # RL policy over GST token weights, driven by BERT embeddings.
        self.rl_policy = RLGSTPolicy(
            bert_hidden_size=self.bert_gst_encoder.bert.config.hidden_size,
            gst_token_num=10,
            temperature=rl_temperature,
        )

        # HuBERT trustworthiness classifier (frozen).
        self.tw_classifier = HTModel.load_from_checkpoint(
            hubert_checkpoint_path,
            map_location="cpu",
            hubert_model_name=hubert_model_name,
            trainable_layers=0,
        )
        for p in self.tw_classifier.parameters():
            p.requires_grad = False
        self.tw_classifier.eval()

        # Vocoder for converting mels to waveforms during RL scoring.
        # Required for RL training (used to generate waveforms for HuBERT scoring).
        self.vocoder = HiFiGANVocoder(checkpoint_dir=vocoder_checkpoint_dir)
        for p in self.vocoder.parameters():
            p.requires_grad = False
        self.vocoder.eval()

        # In RL mode, Tacotron2 + GST should act as fixed, pre-trained components.
        # Freeze their parameters so only the RL head and related modules train.
        for p in self.tts.parameters():
            p.requires_grad = False
        self.tts.eval()
        

    def _save_audio_if_needed(
        self,
        waveforms: Tensor,
        batch_idx: int,
        length_samples: Optional[Tensor] = None,
        *,
        sample_rate: Optional[int] = None,
        filename_suffix: str = "",
    ) -> None:
        if not self.save_audio_dir:
            return
        if not self.save_audio_every_n_steps or int(self.save_audio_every_n_steps) <= 0:
            return
        if batch_idx % int(self.save_audio_every_n_steps) != 0:
            return
        trainer = getattr(self, "trainer", None)
        if trainer is not None and not getattr(trainer, "is_global_zero", True):
            return

        write_sr = int(self._vocoder_sample_rate) if sample_rate is None else int(sample_rate)

        import numpy as np
        from pathlib import Path
        import soundfile as sf

        out_dir = Path(self.save_audio_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        wav = waveforms.detach().float().cpu()
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        max_t = wav.shape[1]
        max_to_save = min(5, int(wav.shape[0]))
        for i in range(max_to_save):
            w = wav[i].numpy()
            if length_samples is not None and i < length_samples.shape[0]:
                n = int(length_samples[i].item())
                n = max(1, min(n, max_t))
                w = w[:n]
            w = w / (np.abs(w).max() + 1e-8)
            name = f"epoch_{self.current_epoch}_step_{batch_idx}_sample_{i}{filename_suffix}.wav"
            sf.write(str(out_dir / name), w, write_sr)

    def _rescale_mel_time_for_vocoder(self, mel: Tensor) -> Tensor:
        """Rescale mel along time before vocoder (matches ``vocoder_mel_time_scale``)."""
        s = float(self.vocoder_mel_time_scale)
        if abs(s - 1.0) <= 1e-6:
            return mel
        return F.interpolate(
            mel.transpose(1, 2),
            scale_factor=s,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def _waveform_for_hubert(self, wav: Tensor) -> Tensor:
        """Resample vocoder output to HuBERT input rate (16 kHz)."""
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        wav = wav.float()
        return ta_f.resample(
            wav,
            orig_freq=self._vocoder_sample_rate,
            new_freq=self._hubert_sample_rate,
        )

    _SILENCE_LOG_MEL: float = -11.5

    @staticmethod
    def _mel_frames_from_pred_gate(gate: Tensor, max_t: int) -> Tensor:
        """First predicted stop (logit < 0) → mel frame count; matches ``generate_and_score_tactron``."""
        g = gate.squeeze(-1) if gate.dim() == 3 else gate
        _, T = g.shape
        gate_end = g < 0
        has_end = gate_end.any(dim=1)
        end_idx = gate_end.int().argmax(dim=1)
        mel_frames = torch.where(
            has_end,
            end_idx + 1,
            torch.full_like(end_idx, T, dtype=torch.long),
        )
        return mel_frames.clamp(min=1, max=max_t)

    @staticmethod
    def _mel_frames_from_target_gate(gate: Tensor, mel_lens: Tensor) -> Tensor:
        """Target gate (1=continue, 0=stop) → mel frame count; capped by ``mel_spectrogram_len``."""
        g = gate.squeeze(-1) if gate.dim() == 3 else gate
        B, T = g.shape
        gate_end = g < 0.5
        has_end = gate_end.any(dim=1)
        end_idx = gate_end.int().argmax(dim=1)
        mel_frames = torch.where(
            has_end,
            end_idx + 1,
            torch.full_like(end_idx, T, dtype=torch.long),
        )
        cap = mel_lens.to(device=g.device, dtype=torch.long).reshape(B).clamp(min=1)
        return torch.minimum(mel_frames, cap).clamp(min=1)

    def _apply_mel_gate_mask(self, mel: Tensor, n_frames: Tensor) -> Tensor:
        """Replace mel from the stop frame onward with silence (vocoder log-mel floor)."""
        B, T, _ = mel.shape
        # Make n broadcastable against t=(1, T, 1): n must be (B, 1, 1), not (B, 1).
        n = n_frames.clamp(min=1, max=T).reshape(B, 1, 1)
        t = torch.arange(T, device=mel.device, dtype=torch.long).view(1, T, 1)
        keep = (t < n).to(dtype=mel.dtype)
        sil = torch.full_like(mel, self._SILENCE_LOG_MEL)
        return keep * mel + (1.0 - keep) * sil

    def _wav_samples_from_mel_frames(self, n_mel_frames: Tensor, s: float, seq_len_wav: int) -> Tensor:
        """Mel frame counts → 22.05 kHz vocoder-output sample lengths."""
        out = (n_mel_frames.float() * s).round().long() * 256
        return out.clamp(min=1, max=seq_len_wav)

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
        style: Optional[Tensor] = None,
    ):
        return self.tts(
            chars_idx=chars_idx,
            chars_idx_len=chars_idx_len,
            teacher_forcing=teacher_forcing,
            teacher_forcing_dropout=teacher_forcing_dropout,
            mel_spectrogram=mel_spectrogram,
            mel_spectrogram_len=mel_spectrogram_len,
            speaker_id=speaker_id,
            max_len_override=max_len_override,
            style=style,
        )

    def _sanitize_batch_for_checkpoint(self, batch):
        """
        Clamp indices to the frozen Tacotron checkpoint limits to avoid CUDA
        index-out-of-bounds (e.g., vocab / speaker-id mismatches).
        """
        # Prefer hparams when present, but fall back to the actual module shapes
        # (some Lightning checkpoints don't populate hparams as expected).
        num_chars_hp = None
        speaker_count_hp = None
        try:
            num_chars_hp = self.tts.hparams.get("num_chars")  # type: ignore[assignment]
            speaker_count_hp = self.tts.hparams.get("speaker_count")  # type: ignore[assignment]
        except Exception:
            pass

        if num_chars_hp is None:
            # Encoder embedding uses (num_chars + 1) with padding_idx=0
            num_chars = int(self.tts.tacotron2.encoder.embedding.num_embeddings) - 1
        else:
            num_chars = int(num_chars_hp)

        if speaker_count_hp is None:
            speaker_emb = getattr(self.tts.tacotron2, "speaker_embedding", None)
            speaker_count = int(getattr(speaker_emb, "num_embeddings", 1) or 1)
        else:
            speaker_count = int(speaker_count_hp)

        # chars_idx is padded with 0; valid ids are [0, num_chars-1]
        chars_idx = batch.chars_idx.clamp(min=0, max=num_chars - 1)

        # speaker_id should be [0, speaker_count-1] when speaker tokens are enabled
        speaker_id = batch.speaker_id
        if speaker_id.numel() > 0:
            bad = (speaker_id < 0) | (speaker_id >= speaker_count)
            if bool(bad.any()):
                speaker_id = speaker_id.clone()
                speaker_id[bad] = 0

        # TTSBatch is a NamedTuple (immutable), so return a replaced copy.
        try:
            return batch._replace(chars_idx=chars_idx, speaker_id=speaker_id)
        except Exception:
            # Fallback for unexpected batch types
            batch.chars_idx = chars_idx
            batch.speaker_id = speaker_id
            return batch

    def training_step(self, batch, batch_idx: int) -> Tensor:
        if self.tts.gst is None:
            raise Exception("oh no")
        if self.tw_classifier is None:
            raise ValueError("hubert_checkpoint_path is required for RL training (tw_classifier is None).")

        batch = self._sanitize_batch_for_checkpoint(batch)
        batch_size = batch.chars_idx.shape[0]
        s = float(self.vocoder_mel_time_scale)

        with torch.no_grad():
            n_ref = self._mel_frames_from_target_gate(batch.gate, batch.mel_spectrogram_len)
            mel_ref_in = self._apply_mel_gate_mask(batch.mel_spectrogram, n_ref)
            mel_ref = self._rescale_mel_time_for_vocoder(mel_ref_in)
            wav = self.vocoder(mel_ref)
            seq_len = wav.shape[1]

            wav_lens = self._wav_samples_from_mel_frames(n_ref, s, seq_len)

            wav_hubert = self._waveform_for_hubert(wav) 
            seq_h = wav_hubert.shape[1]
            wav_lens_h = (wav_lens.float() / s).round().long().clamp(min=1, max=seq_h)
            time = torch.arange(seq_h, device=self.device)[None, :]  # (1, T')
            mask = time < wav_lens_h[:, None]  # (B, T')

            tw_scores = torch.sigmoid(self.tw_classifier(wav=wav_hubert, mask=mask)).squeeze(-1)  # type: ignore[operator]

        bert_embeddings = self.bert_gst_encoder(score=tw_scores, text=batch.text)
        (
            gst_weights,
            log_probs,
            z,
            mu_std,
            log_std_mean,
            std_mean,
        ) = self.rl_policy(bert_embeddings, deterministic=not self.training)

        style = torch.bmm(
            gst_weights.unsqueeze(1),
            torch.tanh(self.tts.gst.stl.embed)[None, :, :].expand(
                batch_size * self.tts.gst.stl.num_heads, -1, -1
            ),
        )
        style = (
            style.transpose(0, 1)
            .view(batch_size, self.tts.gst.stl.token_embedding_size)
        )
        style = self.tts.gst.stl.attention.out_proj(style).unsqueeze(1)

        with torch.no_grad():
            mel_spectrogram, mel_spectrogram_post, gate, _ = self(
                chars_idx=batch.chars_idx,
                chars_idx_len=batch.chars_idx_len,
                teacher_forcing=True,
                teacher_forcing_dropout=0,
                speaker_id=batch.speaker_id,
                mel_spectrogram=batch.mel_spectrogram,
                mel_spectrogram_len=batch.mel_spectrogram_len,
                style=style,
            )

            n_pred = self._mel_frames_from_pred_gate(gate, mel_spectrogram_post.shape[1])
            mel_pred = self._apply_mel_gate_mask(mel_spectrogram_post, n_pred)
            #mel_pred = self._rescale_mel_time_for_vocoder(mel_pred_in)
            wav_pred = self.vocoder(mel_pred)
            seq_len_pred = wav_pred.shape[1]
            pred_wav_lengths = self._wav_samples_from_mel_frames(n_pred, s, seq_len_pred)

            wav_pred_hubert = self._waveform_for_hubert(wav_pred)

            seq_hp = wav_pred_hubert.shape[1]
            pred_wav_lengths_h = (pred_wav_lengths.float() / s).round().long().clamp(min=1, max=seq_hp)

            self._save_audio_if_needed(wav_pred, batch_idx, pred_wav_lengths)
            self._save_audio_if_needed(
                wav_pred_hubert,
                batch_idx,
                pred_wav_lengths_h,
                sample_rate=self._hubert_sample_rate,
                filename_suffix="_hubert",
            )
            time_pred = torch.arange(seq_hp, device=self.device)[None, :]  # (1, T'')
            mask_pred = time_pred < pred_wav_lengths_h[:, None]  # (B, T'')
            tw_scores_pred = torch.sigmoid(
                self.tw_classifier(wav=wav_pred_hubert, mask=mask_pred)
            ).squeeze(-1)  # type: ignore[operator]

        gate_loss = F.binary_cross_entropy_with_logits(gate, batch.gate, reduction="none").mean(dim=(1, 2))
        mel_loss = F.mse_loss(mel_spectrogram, batch.mel_spectrogram, reduction="none").mean(dim=(1, 2))
        mel_post_loss = F.mse_loss(mel_spectrogram_post, batch.mel_spectrogram, reduction="none").mean(dim=(1, 2))
        tts_loss = gate_loss + mel_loss + mel_post_loss  # (batch_size,)
        
        abs_diff = (tw_scores - tw_scores_pred).abs()  # (batch_size,)
        score_correct = 1-abs_diff #(abs_diff < 0.1).float()

        #reward = score_correct.mean()
        #reward = -(abs_diff) #- tts_loss.detach() #torch.exp(-abs_diff / 0.1)
        reward = (abs_diff < 0.1).float() * 2 - 1
        advantage = (reward - reward.mean()) / (reward.std() + 1e-8)
        log_prob_per_sample = log_probs.reshape(batch_size, -1).sum(dim=1)  # (batch_size,)
        reinforce_loss = -(log_prob_per_sample * reward.detach()).mean()

        gst_entropy = -(gst_weights * torch.log(gst_weights.clamp_min(1e-8))).sum(dim=1)
        gst_entropy_per_sample = gst_entropy.view(batch_size, self.rl_policy.gst_heads).mean(dim=1)
        entropy_bonus = gst_entropy_per_sample.mean()

        loss = reinforce_loss - self.rl_entropy_coef * entropy_bonus
        #reward = reward #- 0.1 * tts_loss.detach()
        # Log subgroup means only when subgroup exists; otherwise mean() would be NaN.
        # Use a Python int for batch_size so Lightning can weight epoch averages.
        target = (tw_scores > 0.6).float()
        low_mask = (target == 0)
        high_mask = (target == 1)
        if bool(low_mask.any()):
            low_ref_pred_mean = tw_scores_pred[low_mask].mean()
            self.log(
                "low_ref_pred_mean",
                low_ref_pred_mean.detach(),
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=int(low_mask.sum().item()),
            )
        if bool(high_mask.any()):
            high_ref_pred_mean = tw_scores_pred[high_mask].mean()
            self.log(
                "high_ref_pred_mean",
                high_ref_pred_mean.detach(),
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=int(high_mask.sum().item()),
            )


        # Per-head GST diagnostics (monitoring only; does not affect loss).
        gst_w = gst_weights.reshape(batch_size, self.rl_policy.gst_heads, self.rl_policy.gst_token_num)
        max_prob = gst_w.max(dim=2).values
        gst_entropy_heads = gst_entropy.reshape(batch_size, self.rl_policy.gst_heads)
        top2 = torch.topk(gst_w, k=2, dim=2).values
        margin = top2[..., 0] - top2[..., 1]
        top_idx = gst_w.argmax(dim=2)  # (batch, heads)

        z_std = z.detach().std()
        # Same logging contract as training_* metrics so TensorBoardLogger records scalars.
        _bs = batch_size
        for h in range(self.rl_policy.gst_heads):
            counts = torch.bincount(top_idx[:, h], minlength=self.rl_policy.gst_token_num).float()
            probs = counts / counts.sum().clamp_min(1.0)
            usage_entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
            self.log(
                f"head{h}_token_usage_entropy",
                usage_entropy.detach(),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=_bs,
            )

        self.log(
            "gst_mean_max_prob",
            max_prob.mean().detach(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=_bs,
        )
        self.log(
            "gst_entropy",
            gst_entropy_heads.mean().detach(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=_bs,
        )
        self.log(
            "gst_top1_top2_margin",
            margin.mean().detach(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=_bs,
        )


        self.log(
            "training_gate_loss",
            gate_loss.detach().mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )
        self.log(
            "training_mel_loss",
            mel_loss.detach().mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )
        self.log(
            "training_mel_post_loss",
            mel_post_loss.detach().mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )
        self.log(
            "training_tts_loss",
            tts_loss.detach().mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )


        self.log(
            "training_reward",
            reward.detach().mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )
        
        self.log(
            "advantage",
            advantage.mean().detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )

        self.log(
            "score_correct",
            score_correct.detach().mean(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )

        self.log(
            "training_reinforce_loss",
            reinforce_loss.detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )

        # Policy diagnostics
        self.log(
            "training_z_std",
            z_std.detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )
        self.log(
            "training_mu_std",
            mu_std.detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )
        
        self.log(
            "training_log_std_mean",
            log_std_mean.detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )
        self.log(
            "training_std_mean",
            std_mean.detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )

        self.log(
             "training_entropy_bonus",
             entropy_bonus.detach(),
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

        self.log(
            "training_log_prob",
            log_prob_per_sample.detach().mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )

        self.log(
            "training_log_prob_std",
            log_prob_per_sample.detach().std(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        # Log grad/weight histograms every N steps to avoid high CPU and GPU syncs.
        LOG_HISTOGRAM_EVERY_N_STEPS = 500

        trainer = getattr(self, "trainer", None)
        if trainer is None or getattr(trainer, "sanity_checking", False):
            return

        # Skip most steps to reduce overhead
        if (trainer.global_step + 1) % LOG_HISTOGRAM_EVERY_N_STEPS != 0:
            return

        logger = getattr(self, "logger", None)
        experiment = getattr(logger, "experiment", None) if logger is not None else None
        # TensorBoard only; CSVLogger has no add_histogram
        if experiment is None or not hasattr(experiment, "add_histogram"):
            return

        for tag, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad.detach().cpu().float().reshape(-1)
                grad_ok = grad[torch.isfinite(grad)]
                if grad_ok.numel() > 0:
                    try:
                        experiment.add_histogram(f"grad/{tag}", grad_ok, self.global_step)
                    except ValueError:
                        pass

                weight = param.detach().cpu().float().reshape(-1)
                w_ok = weight[torch.isfinite(weight)]
                if w_ok.numel() > 0:
                    try:
                        experiment.add_histogram(f"weight/{tag}", w_ok, self.global_step)
                    except ValueError:
                        pass

    def validation_step(self, batch, batch_idx: int) -> Tensor:
        if self.tts.gst is None:
            raise Exception("oh no")
        if self.tw_classifier is None:
            raise ValueError("hubert_checkpoint_path is required for RL training (tw_classifier is None).")

        batch = self._sanitize_batch_for_checkpoint(batch)
        batch_size = batch.chars_idx.shape[0]
        s = float(self.vocoder_mel_time_scale)

        with torch.no_grad():
            n_ref = self._mel_frames_from_target_gate(batch.gate, batch.mel_spectrogram_len)
            mel_ref = self._apply_mel_gate_mask(batch.mel_spectrogram, n_ref)
            #mel_ref = self._rescale_mel_time_for_vocoder(mel_ref_in)
            wav = self.vocoder(mel_ref)
            seq_len = wav.shape[1]

            wav_lens = self._wav_samples_from_mel_frames(n_ref, s, seq_len)

            wav_hubert = self._waveform_for_hubert(wav)
            seq_h = wav_hubert.shape[1]
            wav_lens_h = (wav_lens.float() / s).round().long().clamp(min=1, max=seq_h)
            time = torch.arange(seq_h, device=self.device)[None, :]  # (1, T')
            mask = time < wav_lens_h[:, None]  # (B, T')

            tw_scores = torch.sigmoid(self.tw_classifier(wav=wav_hubert, mask=mask)).squeeze(-1)  # type: ignore[operator]

        # Deterministic policy for evaluation
        bert_embeddings = self.bert_gst_encoder(score=tw_scores, text=batch.text)
        gst_weights, _, _, _, _, _ = self.rl_policy(bert_embeddings, deterministic=True)
        
        style = torch.bmm(
            gst_weights.unsqueeze(1),
            torch.tanh(self.tts.gst.stl.embed)[None, :, :].expand(
                batch_size * self.tts.gst.stl.num_heads, -1, -1
            ),
        )
        style = (
            style.transpose(0, 1)
            .view(batch_size, self.tts.gst.stl.token_embedding_size)
        )
        style = self.tts.gst.stl.attention.out_proj(style).unsqueeze(1)


        with torch.no_grad():
            mel_spectrogram, mel_spectrogram_post, gate, _ = self(
                chars_idx=batch.chars_idx,
                chars_idx_len=batch.chars_idx_len,
                teacher_forcing=True,
                teacher_forcing_dropout=0,
                speaker_id=batch.speaker_id,
                mel_spectrogram=batch.mel_spectrogram,
                mel_spectrogram_len=batch.mel_spectrogram_len,
                style=style,
            )

            n_pred = self._mel_frames_from_pred_gate(gate, mel_spectrogram_post.shape[1])
            mel_pred = self._apply_mel_gate_mask(mel_spectrogram_post, n_pred)
            #mel_pred = self._rescale_mel_time_for_vocoder(mel_pred_in)
            wav_pred = self.vocoder(mel_pred)
            seq_len_pred = wav_pred.shape[1]

            pred_wav_lengths = self._wav_samples_from_mel_frames(n_pred, s, seq_len_pred)

            wav_pred_hubert = self._waveform_for_hubert(wav_pred)
            seq_hp = wav_pred_hubert.shape[1]
            pred_wav_lengths_h = (pred_wav_lengths.float() / s).round().long().clamp(min=1, max=seq_hp)
            time_pred = torch.arange(seq_hp, device=self.device)[None, :]  # (1, T'')
            mask_pred = time_pred < pred_wav_lengths_h[:, None]  # (B, T'')

            tw_scores_pred = torch.sigmoid(
                self.tw_classifier(wav=wav_pred_hubert, mask=mask_pred)
            ).squeeze(-1)  # type: ignore[operator]

        gate_loss = F.binary_cross_entropy_with_logits(gate, batch.gate)
        mel_loss = F.mse_loss(mel_spectrogram, batch.mel_spectrogram)
        mel_post_loss = F.mse_loss(mel_spectrogram_post, batch.mel_spectrogram)
        tts_loss = gate_loss + mel_loss + mel_post_loss

        val_score_loss = (tw_scores - tw_scores_pred).abs().mean(-1)
        val_loss = val_score_loss + tts_loss

        self.log(
            "val_gate_loss",
            gate_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )
        self.log(
            "val_mel_loss",
            mel_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )
        self.log(
            "val_mel_post_loss",
            mel_post_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )
        self.log(
            "val_tts_loss",
            tts_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )
        self.log(
            "val_score_loss",
            val_score_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )
        return val_loss
