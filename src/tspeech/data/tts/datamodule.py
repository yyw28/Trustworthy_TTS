import csv
from os import path
from typing import Final, Optional

import pandas as pd
import torch
from lightning import LightningDataModule
from torch import Tensor, nn
from torch.utils.data import DataLoader

from tspeech.data.tts.dataset import TTSBatch, TTSDataset


def collate_fn(data: list[TTSBatch]) -> TTSBatch:
    speaker_id_all: list[Tensor] = []
    chars_idx_all: list[Tensor] = []
    chars_idx_len_all: list[Tensor] = []
    mel_spectrogram_all: list[Tensor] = []
    mel_spectrogram_len_all: list[Tensor] = []
    gate_all: list[Tensor] = []
    gate_len_all: list[Tensor] = []

    filename: list[str] = []
    text: list[str] = []

    for d in data:
        speaker_id_all.append(d.speaker_id.squeeze(0))
        chars_idx_all.append(d.chars_idx.squeeze(0))
        chars_idx_len_all.append(d.chars_idx_len.squeeze(0))

        mel_spectrogram_all.append(d.mel_spectrogram.squeeze(0))
        mel_spectrogram_len_all.append(d.mel_spectrogram_len.squeeze(0))

        gate_all.append(d.gate.squeeze(0))
        gate_len_all.append(d.gate_len.squeeze(0))

        filename.extend(d.filename)
        text.extend(d.text)

    return TTSBatch(
        speaker_id=torch.tensor(speaker_id_all),
        chars_idx=nn.utils.rnn.pad_sequence(chars_idx_all, batch_first=True),
        chars_idx_len=torch.tensor(chars_idx_len_all),
        mel_spectrogram=nn.utils.rnn.pad_sequence(
            mel_spectrogram_all, batch_first=True
        ),
        mel_spectrogram_len=torch.tensor(mel_spectrogram_len_all),
        gate=nn.utils.rnn.pad_sequence(gate_all, batch_first=True),
        gate_len=torch.tensor(gate_len_all),
        text=text,
        filename=filename,
    )


class TTSDatamodule(LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        csv_train: str,
        csv_val: str,
        csv_test: str,
        batch_size: int,
        num_workers: int,
        num_mels: int,
        sample_rate: int,
        expand_abbreviations: bool,
        allowed_chars: str,
        end_token: Optional[str] = "^",
        silence: int = 0,
        trim: bool = True,
        trim_top_db: int = 60,
        trim_frame_length: int = 2048,
    ):

        super().__init__()

        self.dataset_dir: Final[str] = dataset_dir
        self.num_workers: Final[int] = num_workers
        self.batch_size: Final[int] = batch_size

        self.csv_train: Final[str] = csv_train
        self.csv_val: Final[str] = csv_val
        self.csv_test: Final[str] = csv_test

        self.allowed_chars: Final[str] = allowed_chars
        self.end_token: Final[str | None] = end_token
        self.silence: Final[int] = silence
        self.trim: Final[bool] = trim
        self.trim_top_db: Final[int] = trim_top_db
        self.trim_frame_length: Final[int] = trim_frame_length
        self.expand_abbreviations: Final[bool] = expand_abbreviations
        self.num_mels: Final[int] = num_mels
        self.sample_rate: Final[int] = sample_rate

    def setup(self, stage: str):
        if stage == "fit":
            self._train_dataloader = DataLoader(
                self.__load_dataset(self.csv_train),
                batch_size=self.batch_size,
                collate_fn=collate_fn,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
                num_workers=self.num_workers,
                persistent_workers=True,
            )
            self._val_dataloader = DataLoader(
                self.__load_dataset(self.csv_val),
                batch_size=self.batch_size * 4,
                collate_fn=collate_fn,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=self.num_workers,
                persistent_workers=True,
            )

    def __load_dataset(self, filename: str) -> TTSDataset:
        df = pd.read_csv(
            path.join(self.dataset_dir, filename), delimiter="|", quoting=csv.QUOTE_NONE
        )

        return TTSDataset(
            filenames=list(df.wav),
            texts=list(df.text),
            speaker_ids=list(df.speaker_idx),
            base_dir=self.dataset_dir,
            allowed_chars=self.allowed_chars,
            end_token=self.end_token,
            silence=self.silence,
            trim=self.trim,
            trim_top_db=self.trim_top_db,
            trim_frame_length=self.trim_frame_length,
            expand_abbreviations=self.expand_abbreviations,
            num_mels=self.num_mels,
            sample_rate=self.sample_rate,
        )

    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    def val_dataloader(self) -> DataLoader:
        return self._val_dataloader

    def test_dataloader(self) -> DataLoader:
        return self._test_dataloader
