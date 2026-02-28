from typing import Optional, Final

import lightning as pl
import matplotlib
from torch import nn
import torch
from torch import Tensor
from torch.nn import functional as F

from tspeech.data.tts import TTSBatch
from tspeech.model.tacotron2 import Tacotron2
from tspeech.model.tacotron2 import GST
from tspeech.model.bert_gst_encoder import BERTEncoder
from tspeech.model.rl_gst_policy_option1 import RLGSTPolicy
from tspeech.model.htmodel import HTModel
from tspeech.vocoder import HiFiGANVocoder
from tspeech.model.tts import TTSModel


class TTSRLModel(pl.LightningModule):
    def __init__(
        self,
        tts_checkpoint_path: str,
        vocoder_checkpoint_dir: str,
        hubert_checkpoint_path: str,
        bert_model_name: str = "bert-base-uncased",
        hubert_model_name: str = "facebook/hubert-base-ls960",
        rl_temperature: float = 1.0,
        rl_entropy_coef: float = 0.01,
        save_audio_dir: Optional[str] = None,
        save_audio_every_n_steps: int = 100,
    ):
        super().__init__()

        self.tts = TTSModel.load_from_checkpoint(tts_checkpoint_path)

        if self.tts.gst is None:
            raise Exception("oh no")

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

    def training_step(self, batch, batch_idx: int) -> Tensor:
        if self.tts.gst is None:
            raise Exception("oh no")

        batch_size = batch.chars_idx.shape[0]

        with torch.no_grad():
            wav = self.vocoder(batch.mel_spectrogram)
            seq_len = wav.shape[1]

            # Approximate per-sample waveform lengths from mel lengths so the HuBERT
            # attention mask only covers real (unpadded) audio.
            wav_lens = batch.mel_spectrogram_len * 256

            time = torch.arange(seq_len, device=self.device)[None, :]  # (1, T)
            mask = time < wav_lens[:, None]  # (B, T)

            tw_scores = torch.sigmoid(self.tw_classifier(wav=wav, mask=mask)).squeeze(
                -1
            )

        bert_embeddings = self.bert_gst_encoder(score=tw_scores, text=batch.text)
        gst_weights, log_probs, entropy = self.rl_policy(
            bert_embeddings, deterministic=not self.training
        )
        style = self.tts.gst.stl.attention.out_proj(
            torch.matmul(gst_weights, torch.tanh(self.tts.gst.stl.embed)).reshape(
                batch_size, 1, -1
            )
        )

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

            wav_pred = self.vocoder(mel_spectrogram_post)
            seq_len_pred = wav_pred.shape[1]

            pred_wav_lengths = (gate.squeeze(-1) < 0).int().argmax(axis=1)
            
            # If the gate fails to predict the end of the clip, assume it takes the whole wav length
            pred_wav_lengths[pred_wav_lengths == 0] = gate.shape[1] - 1

            # Multiply by 256 to calculate final wav output
            pred_wav_lengths *= 256

            time_pred = torch.arange(seq_len_pred, device=self.device)[
                None, :
            ]  # (1, T')
            mask_pred = time_pred < pred_wav_lengths[:, None]  # (B, T')
            tw_scores_pred = torch.sigmoid(
                self.tw_classifier(wav=wav_pred, mask=mask_pred)
            ).squeeze(-1)

        tts_loss = (
            F.binary_cross_entropy_with_logits(gate, batch.gate)
            + F.mse_loss(mel_spectrogram, batch.mel_spectrogram)
            + F.mse_loss(mel_spectrogram_post, batch.mel_spectrogram)
        )

        score_loss = F.mse_loss(tw_scores, tw_scores_pred)
        reward = -(score_loss + tts_loss)
        reinforce_loss = (log_probs.reshape(batch_size, -1) * reward.detach()).mean()
        # entropy_bonus = self.rl_entropy_coef * entropy.mean()

        loss = reinforce_loss  # + entropy_bonus

        self.log(
            "training_tts_loss",
            tts_loss.detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )

        self.log(
            "training_score_loss",
            score_loss.detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )

        self.log(
            "training_reward",
            reward.detach(),
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

        # self.log(
        #     "training_entropy_bonus",
        #     entropy_bonus.detach(),
        #     on_step=True,
        #     on_epoch=True,
        #     sync_dist=True,
        #     batch_size=mel_spectrogram.shape[0],
        # )

        self.log(
            "training_loss",
            loss.detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=mel_spectrogram.shape[0],
        )

        return loss

    def validation_step(self, batch, batch_idx: int) -> Tensor:
        if self.tts.gst is None:
            raise Exception("oh no")

        batch_size = batch.chars_idx.shape[0]

        with torch.no_grad():
            # Reference audio HuBERT scores
            wav = self.vocoder(batch.mel_spectrogram)
            seq_len = wav.shape[1]

            wav_lens = batch.mel_spectrogram_len * 256

            time = torch.arange(seq_len, device=self.device)[None, :]  # (1, T)
            mask = (time < wav_lens[:, None]).long().unsqueeze(1)  # (B, 1, T)

            tw_scores = torch.sigmoid(self.tw_classifier(wav=wav, mask=mask)).squeeze(
                -1
            )

        # Deterministic policy for evaluation
        bert_embeddings = self.bert_gst_encoder(score=tw_scores, text=batch.text)
        gst_weights, _, _ = self.rl_policy(bert_embeddings, deterministic=True)
        style = self.tts.gst.stl.attention.out_proj(
            torch.matmul(gst_weights, torch.tanh(self.tts.gst.stl.embed)).reshape(
                batch_size, 1, -1
            )
        )

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

            wav_pred = self.vocoder(mel_spectrogram_post)
            seq_len_pred = wav_pred.shape[1]

            pred_wav_lengths = (gate.squeeze(-1) < 0).int().argmax(axis=1)
            
            # If the gate fails to predict the end of the clip, assume it takes the whole wav length
            pred_wav_lengths[pred_wav_lengths == 0] = gate.shape[1] - 1

            # Multiply by 256 to calculate final wav output
            pred_wav_lengths *= 256

            time_pred = torch.arange(seq_len_pred, device=self.device)[
                None, :
            ]  # (1, T')
            mask_pred = (time_pred < pred_wav_lengths[:, None])  # (B, T')

            tw_scores_pred = torch.sigmoid(
                self.tw_classifier(wav=wav_pred, mask=mask_pred)
            ).squeeze(-1)

        tts_loss = (
            F.binary_cross_entropy_with_logits(gate, batch.gate)
            + F.mse_loss(mel_spectrogram, batch.mel_spectrogram)
            + F.mse_loss(mel_spectrogram_post, batch.mel_spectrogram)
        )

        score_loss = F.mse_loss(tw_scores, tw_scores_pred)
        val_loss = score_loss + tts_loss

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
            score_loss,
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
