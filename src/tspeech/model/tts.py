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
    ):
        super().__init__()

        self.save_hyperparameters()

        self.speaker_tokens = speaker_tokens_enabled
        self.max_len_override = max_len_override

        self.teacher_forcing_dropout = teacher_forcing_dropout

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
    ):
        style: Tensor | None = None

        if self.gst is not None:
            style = self.gst(mel_spectrogram_style, mel_spectrogram_style_len)

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
        mel_spectrogram, mel_spectrogram_post, gate, alignment = self(
            chars_idx=batch.chars_idx,
            chars_idx_len=batch.chars_idx_len,
            teacher_forcing=True,
            teacher_forcing_dropout=self.teacher_forcing_dropout,
            speaker_id=batch.speaker_id,
            mel_spectrogram=batch.mel_spectrogram,
            mel_spectrogram_len=batch.mel_spectrogram_len,
            mel_spectrogram_style=batch.mel_spectrogram,
            mel_spectrogram_style_len=batch.mel_spectrogram_len,
        )

        gate_loss = F.binary_cross_entropy_with_logits(gate, batch.gate)
        mel_loss = F.mse_loss(mel_spectrogram, batch.mel_spectrogram)
        mel_post_loss = F.mse_loss(mel_spectrogram_post, batch.mel_spectrogram)

        loss = gate_loss + mel_loss + mel_post_loss

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
        )

        return mel_spectrogram, mel_spectrogram_post, gate, alignment, text


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
