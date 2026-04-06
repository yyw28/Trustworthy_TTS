import csv
import warnings
from os import path
from typing import Final, Optional

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch import Tensor, nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from tspeech.data.tts.dataset import TTSBatch, TTSDataset

# If train CSV has no explicit score column, try these names (first match wins).
_DEFAULT_REWEIGHT_SCORE_COLUMNS: tuple[str, ...] = (
    "trust_score",
    "ref_tw_score",
    "tw_score",
    "huBERT_score",
    "hubert_score",
    "trustworthy",
)


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
        reweight_train_sampler: bool = False,
        reweight_threshold: float = 0.6,
        reweight_score_column: Optional[str] = None,
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
        self.reweight_train_sampler: Final[bool] = reweight_train_sampler
        self.reweight_threshold: Final[float] = reweight_threshold
        self.reweight_score_column: Final[Optional[str]] = reweight_score_column

    def _read_manifest(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(
            path.join(self.dataset_dir, filename),
            delimiter="|",
            quoting=csv.QUOTE_NONE,
        )

    @staticmethod
    def _series_str_clean(s: pd.Series) -> list[str]:
        """CSV cells missing or mis-parsed (NaN) become ''; everything else str."""
        return ["" if pd.isna(x) else str(x) for x in s]

    def _df_to_dataset(self, df: pd.DataFrame) -> TTSDataset:
        return TTSDataset(
            filenames=self._series_str_clean(df.wav),
            texts=self._series_str_clean(df.text),
            speaker_ids=list(pd.to_numeric(df.speaker_idx, errors="coerce").fillna(0).astype(int)),
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

    def _resolve_reweight_column(self, df: pd.DataFrame) -> Optional[str]:
        if self.reweight_score_column is not None:
            c = self.reweight_score_column
            if c not in df.columns:
                return None
            return c
        for c in _DEFAULT_REWEIGHT_SCORE_COLUMNS:
            if c in df.columns:
                return c
        return None

    def _train_sample_weights(self, df: pd.DataFrame) -> Optional[torch.Tensor]:
        """
        Per-index weights so low vs high trust (split at ``reweight_threshold``) are
        sampled with equal total mass per class (inverse class frequency).
        """
        col = self._resolve_reweight_column(df)
        if col is None:
            return None
        scores = pd.to_numeric(df[col], errors="coerce")
        if scores.isna().all():
            return None
        scores = scores.fillna(scores.median())
        s = scores.to_numpy(dtype=np.float64)
        thr = float(self.reweight_threshold)
        low_mask = s <= thr
        high_mask = ~low_mask
        n_low = int(low_mask.sum())
        n_high = int(high_mask.sum())
        if n_low == 0 or n_high == 0:
            warnings.warn(
                "reweight_train_sampler: all rows on one side of reweight_threshold; "
                "using uniform sampling.",
                UserWarning,
                stacklevel=2,
            )
            return None
        w = np.zeros(len(df), dtype=np.float64)
        w[low_mask] = 1.0 / n_low
        w[high_mask] = 1.0 / n_high
        return torch.from_numpy(w)

    def setup(self, stage: str):
        if stage == "fit":
            df_train = self._read_manifest(self.csv_train)
            train_ds = self._df_to_dataset(df_train)

            train_kw: dict = dict(
                batch_size=self.batch_size,
                collate_fn=collate_fn,
                drop_last=True,
                pin_memory=True,
                num_workers=self.num_workers,
                persistent_workers=True,
            )

            if self.reweight_train_sampler:
                weights = self._train_sample_weights(df_train)
                if weights is None:
                    warnings.warn(
                        "reweight_train_sampler is True but no usable score column was found "
                        f"(tried reweight_score_column={self.reweight_score_column!r} then "
                        f"{_DEFAULT_REWEIGHT_SCORE_COLUMNS}). Using uniform shuffle.",
                        UserWarning,
                        stacklevel=2,
                    )
                    train_kw["shuffle"] = True
                else:
                    sampler = WeightedRandomSampler(
                        weights.double(),
                        num_samples=len(train_ds),
                        replacement=True,
                    )
                    train_kw["sampler"] = sampler
                    train_kw["shuffle"] = False
            else:
                train_kw["shuffle"] = True

            self._train_dataloader = DataLoader(train_ds, **train_kw)
            self._val_dataloader = DataLoader(
                self._df_to_dataset(self._read_manifest(self.csv_val)),
                batch_size=self.batch_size * 4,
                collate_fn=collate_fn,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=self.num_workers,
                persistent_workers=True,
            )

    def __load_dataset(self, filename: str) -> TTSDataset:
        return self._df_to_dataset(self._read_manifest(filename))

    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    def val_dataloader(self) -> DataLoader:
        return self._val_dataloader

    def test_dataloader(self) -> DataLoader:
        return self._test_dataloader
