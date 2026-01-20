from typing import List, Optional
from pathlib import Path

import lightning as pl
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from tspeech.data.tts import TTSBatch
from tspeech.model.tacotron2 import Tacotron2
from tspeech.model.tacotron2 import GST
from tspeech.model.bert_gst_encoder import BERTGSTEncoder
from tspeech.model.gst_with_weights import GSTWithWeights
from tspeech.model.rl_gst_policy_option1 import RLGSTPolicy, RLGSTRewardFunction


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
        speaker_tokens_enabled: bool = False,
        speaker_count: int = 1,
        max_len_override: Optional[int] = None,
        use_bert_gst: bool = True,
        bert_model_name: str = "bert-base-uncased",
        freeze_bert: bool = True,
        use_hubert_classifier: bool = True,
        hubert_model_name: str = "facebook/hubert-base-ls960",
        hubert_checkpoint_path: Optional[str] = None,
        use_rl_training: bool = False,
        rl_temperature: float = 1.0,
        rl_entropy_coef: float = 0.01,
        use_vocoder: bool = True,
        vocoder_checkpoint_dir: Optional[str] = None,
        save_audio_dir: Optional[str] = None,
        save_audio_every_n_steps: int = 100,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.speaker_tokens = speaker_tokens_enabled
        self.max_len_override = max_len_override
        self.use_bert_gst = use_bert_gst
        self.use_hubert_classifier = use_hubert_classifier
        self.use_rl_training = use_rl_training
        self.rl_entropy_coef = rl_entropy_coef
        self.use_vocoder = use_vocoder
        self.save_audio_dir = save_audio_dir
        self.save_audio_every_n_steps = save_audio_every_n_steps
        
        # Create audio output directory if specified
        if self.save_audio_dir:
            Path(self.save_audio_dir).mkdir(parents=True, exist_ok=True)
        
        # Enable manual optimization when using multiple optimizers (RL training)
        if use_rl_training:
            self.automatic_optimization = False
        
        # Initialize vocoder for RL training
        if use_rl_training and use_vocoder:
            from tspeech.vocoder import HiFiGANVocoder
            
            # Default to UNIVERSAL_V1 checkpoint if not specified
            if vocoder_checkpoint_dir is None:
                # Resolve relative to this file's location
                vocoder_checkpoint_dir = Path(__file__).parent / "UNIVERSAL_V1"
            
            self.vocoder = HiFiGANVocoder(
                checkpoint_dir=str(vocoder_checkpoint_dir),
                sample_rate=22050,
                device="cpu",
            )
            print(f"✓ HiFi-GAN vocoder initialized for RL training (CPU mode) from {vocoder_checkpoint_dir}")
        else:
            self.vocoder = None

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
        )

        # BERT-based GST encoder (replaces mel-based GST)
        if use_bert_gst:
            self.bert_gst_encoder = BERTGSTEncoder(
                bert_model_name=bert_model_name,
                gst_token_num=10,
                freeze_bert=freeze_bert,
            )
            self.gst_with_weights = GSTWithWeights(
                token_num=10,
                token_embedding_size=encoded_dim,
            )
            
            # RL Policy network for GST weights optimization
            if use_rl_training:
                bert_hidden_size = self.bert_gst_encoder.bert.config.hidden_size
                self.rl_policy = RLGSTPolicy(
                    bert_hidden_size=bert_hidden_size,
                    gst_token_num=10,
                    temperature=rl_temperature,
                )
                self.reward_function = RLGSTRewardFunction(
                    trustworthiness_weight=1.0,
                    use_baseline=True,
                )
        else:
            # Fallback to original mel-based GST
            self.gst = GST(out_dim=encoded_dim)
        
        # HubERT trustworthiness classifier
        if use_hubert_classifier:
            from tspeech.hubert.model.htmodel import HTModel
            
            # Load from checkpoint
            self.hubert_classifier = HTModel.load_from_checkpoint(
                hubert_checkpoint_path,
                hubert_model_name=hubert_model_name,
                trainable_layers=0,  # Use frozen for inference
            )
            # Freeze for inference
            for param in self.hubert_classifier.parameters():
                param.requires_grad = False
        
        # Freeze Tacotron2 + GST tokens when using RL training
        if use_rl_training:
            for param in self.tacotron2.parameters():
                param.requires_grad = False
            
            if use_bert_gst:
                for param in self.bert_gst_encoder.parameters():
                    param.requires_grad = False
                for param in self.gst_with_weights.parameters():
                    param.requires_grad = False
                for param in self.rl_policy.parameters():
                    param.requires_grad = True
            
            print("✓ Frozen Tacotron2 + GST tokens for RL training")
            print("  Only RL policy will be trained")

    def forward(
        self,
        chars_idx: Tensor,
        chars_idx_len: Tensor,
        teacher_forcing: bool = True,
        mel_spectrogram: Optional[Tensor] = None,
        mel_spectrogram_len: Optional[Tensor] = None,
        speaker_id: Optional[Tensor] = None,
        max_len_override: Optional[int] = None,
        text: Optional[List[str]] = None,
        return_trustworthiness: bool = False,
    ):
        # Generate GST style embedding
        if self.use_bert_gst:
            # Use BERT to generate GST weights from text
            if text is None:
                raise ValueError("text is required when use_bert_gst=True")
            
            # Get GST weights (either from deterministic encoder or RL policy)
            if self.use_rl_training and self.training:
                # Use RL policy during training
                bert_embeddings = self.bert_gst_encoder.get_bert_embeddings(text)
                gst_weights, log_probs, entropy = self.rl_policy(
                    bert_embeddings,
                    deterministic=False,
                )
                # Store for REINFORCE loss computation
                self._rl_log_probs = log_probs
                self._rl_entropy = entropy
            else:
                # Use deterministic BERT encoder (inference or non-RL training)
                gst_weights = self.bert_gst_encoder(text)  # (batch, 10)
                self._rl_log_probs = None
                self._rl_entropy = None
            
            style = self.gst_with_weights(gst_weights)
            seq_len = chars_idx.shape[1]
            style = style.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            if mel_spectrogram is None or mel_spectrogram_len is None:
                raise ValueError("mel_spectrogram and mel_spectrogram_len are required when use_bert_gst=False")
            style = self.gst(mel_spectrogram, mel_spectrogram_len)
            seq_len = chars_idx.shape[1]
            style = style.unsqueeze(1).expand(-1, seq_len, -1)

        mel_output, mel_postnet, gate, alignment = self.tacotron2(
            chars_idx=chars_idx,
            chars_idx_len=chars_idx_len,
            teacher_forcing=teacher_forcing,
            mel_spectrogram=mel_spectrogram,
            mel_spectrogram_len=mel_spectrogram_len,
            speaker_id=speaker_id,
            max_len_override=max_len_override,
            encoded_extra=style,
        )
        
        if return_trustworthiness and self.use_hubert_classifier:
            waveforms = self.vocoder(mel_postnet, sample_rate=22050).to(self.device)
            
            import torchaudio
            resampler = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000).to(self.device)
            waveforms_16k = resampler(waveforms)
            
            attention_mask = torch.ones(waveforms_16k.shape[0], waveforms_16k.shape[1], dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                trustworthiness_logits = self.hubert_classifier(wav=waveforms_16k, mask=attention_mask)
                trustworthiness_score = torch.sigmoid(trustworthiness_logits).squeeze(-1)
        else:
            trustworthiness_score = None
        
        if return_trustworthiness:
            return mel_output, mel_postnet, gate, alignment, trustworthiness_score
        else:
            return mel_output, mel_postnet, gate, alignment

    def validation_step(self, batch: TTSBatch, batch_idx):
        mel_spectrogram, mel_spectrogram_post, gate, alignment = self(
            chars_idx=batch.chars_idx,
            chars_idx_len=batch.chars_idx_len,
            teacher_forcing=True,
            speaker_id=batch.speaker_id,
            mel_spectrogram=batch.mel_spectrogram,
            mel_spectrogram_len=batch.mel_spectrogram_len,
            text=batch.text,
        )

        gate_loss = F.binary_cross_entropy_with_logits(gate, batch.gate)
        mel_loss = F.mse_loss(mel_spectrogram, batch.mel_spectrogram)
        mel_post_loss = F.mse_loss(mel_spectrogram_post, batch.mel_spectrogram)

        loss = gate_loss + mel_loss + mel_post_loss
        self.log("val_mel_loss", loss, on_step=False, on_epoch=True)

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

        self.log("val_loss", loss, on_step=False, on_epoch=True)

        out["loss"] = loss
        return out

    def training_step(self, batch, batch_idx):
        mel_spectrogram, mel_spectrogram_post, gate, alignment = self(
            chars_idx=batch.chars_idx,
            chars_idx_len=batch.chars_idx_len,
            teacher_forcing=True,
            speaker_id=batch.speaker_id,
            mel_spectrogram=batch.mel_spectrogram,
            mel_spectrogram_len=batch.mel_spectrogram_len,
            text=batch.text,
        )

        # Standard TTS losses
        gate_loss = F.binary_cross_entropy_with_logits(gate, batch.gate)
        mel_loss = F.mse_loss(mel_spectrogram, batch.mel_spectrogram)
        mel_post_loss = F.mse_loss(mel_spectrogram_post, batch.mel_spectrogram)
        
        tts_loss = gate_loss + mel_loss + mel_post_loss
        
        if self.use_rl_training:
            waveforms = self.vocoder(mel_spectrogram_post, sample_rate=22050)
            self._save_audio_if_needed(waveforms, batch_idx)
            rl_loss = self.compute_rl_loss(waveforms, sample_rate=22050)
            
            optimizer = self.optimizers()
            optimizer.zero_grad()
            rl_loss.backward()
            optimizer.step()
            
            loss = rl_loss
            rl_loss_for_logging = rl_loss
        else:
            optimizer = self.optimizers()
            optimizer.zero_grad()
            tts_loss.backward()
            optimizer.step()
            
            loss = tts_loss
            rl_loss_for_logging = None
        
        self._log_training_metrics(gate_loss, mel_loss, mel_post_loss, rl_loss_for_logging, loss)
        return {"loss": loss.detach()}
    
    def _log_training_metrics(self, gate_loss, mel_loss, mel_post_loss, rl_loss, total_loss):
        self.log("training_gate_loss", gate_loss.detach(), on_step=True, on_epoch=True)
        self.log("training_mel_loss", mel_loss.detach(), on_step=True, on_epoch=True)
        self.log("training_mel_post_loss", mel_post_loss.detach(), on_step=True, on_epoch=True)
        if rl_loss is not None:
            self.log("training_rl_loss", rl_loss.detach(), on_step=True, on_epoch=True)
        self.log("training_loss", total_loss.detach(), on_step=True, on_epoch=True)
    
    def _save_audio_if_needed(self, waveforms: Tensor, batch_idx: int):
        if not (self.save_audio_dir and batch_idx % self.save_audio_every_n_steps == 0):
            return
        
        import soundfile as sf
        import numpy as np
        
        for i, waveform in enumerate(waveforms):
            audio_np = waveform.detach().cpu().numpy()
            max_val = np.abs(audio_np).max()
            if max_val > 0:
                audio_np = audio_np / (max_val + 1e-8)
            
            audio_path = Path(self.save_audio_dir) / f"epoch_{self.current_epoch}_step_{batch_idx}_sample_{i}.wav"
            sf.write(str(audio_path), audio_np, 22050)
    
    def compute_rl_loss(self, waveforms: Tensor, sample_rate: int = 16000) -> Tensor:
        """Compute REINFORCE loss for RL training."""
        waveforms = waveforms.to(self.device)
        batch_size = waveforms.shape[0]
        device = waveforms.device
        
        if sample_rate != 16000:
            import torchaudio
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(device)
            waveforms = resampler(waveforms)
        
        seq_len = waveforms.shape[1]
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        
        with torch.no_grad():
            trustworthiness_logits = self.hubert_classifier(wav=waveforms, mask=attention_mask)
        
        rewards, advantages = self.reward_function.compute_reward(trustworthiness_logits)
        reinforce_loss = -(self._rl_log_probs * advantages.detach()).mean()
        entropy_bonus = -self.rl_entropy_coef * self._rl_entropy.mean()
        total_rl_loss = reinforce_loss + entropy_bonus
        
        self.log("rl_reward_mean", rewards.mean().detach(), on_step=True, on_epoch=True)
        self.log("rl_advantage_mean", advantages.mean().detach(), on_step=True, on_epoch=True)
        self.log("rl_entropy_mean", self._rl_entropy.mean().detach(), on_step=True, on_epoch=True)
        
        return total_rl_loss

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
            text=batch.text,
        )

        return mel_spectrogram, mel_spectrogram_post, gate, alignment, text
    
    def configure_optimizers(self):
        from torch.optim import Adam
        
        if self.use_rl_training:
            rl_params = [p for n, p in self.named_parameters() if p.requires_grad and 'rl_policy' in n]
            if len(rl_params) == 0:
                raise ValueError("No trainable RL policy parameters found!")
            print(f"  RL optimizer: {len(rl_params)} trainable parameters")
            return Adam(rl_params, lr=1e-4)
        else:
            tts_params = [p for n, p in self.named_parameters() if p.requires_grad]
            print(f"  TTS optimizer: {len(tts_params)} trainable parameters")
            return Adam(tts_params, lr=1e-3)
