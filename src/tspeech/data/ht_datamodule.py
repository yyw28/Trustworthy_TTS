import os
from os import path
from typing import Final, Literal

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from tspeech.data.collate_fn import trustworthy_collate_fn
from tspeech.data.tis_dataset import TISDataset


class HTDataModule(LightningDataModule):
    def __init__(
        self,
        datasets: dict[Literal["tis", "synth"], str],
        batch_size: int,
        num_workers: int,
    ):

        super().__init__()

        self.num_workers = num_workers
        self.batch_size: Final[int] = batch_size

        if len(datasets) == 0:
            raise Exception("At least one dataset must be specified")

        self.datasets: list[Dataset] = []

        if "tis" in datasets:
            dataset_dir = datasets["tis"]

            # Some of the wav files are missing. Search the ones that are there and use it to narrow down the dataframe
            wav_files: set[str] = set()
            for _, _, files in os.walk(path.join(dataset_dir, "Speech WAV Files")):
                for file in files:
                    if file.endswith(".wav"):
                        wav_files.add(file.split(".")[0])

            df = pd.read_csv(
                path.join(dataset_dir, "Speech_dataset_characteristics.csv")
            )
            df["Audio_Filename"] = df["Audio_Filename"].str.strip()
            df = df[df["Audio_Filename"].isin(wav_files)]

            self.datasets.append(TISDataset(df=df, dataset_dir=dataset_dir))

    def setup(self, stage: str):
        datasets_train: list[Dataset] = []
        datasets_val: list[Dataset] = []
        datasets_test: list[Dataset] = []

        for dataset in self.datasets:
            dataset_train, dataset_val, dataset_test = random_split(
                dataset,
                [0.8, 0.1, 0.1],
                generator=torch.Generator().manual_seed(42),
            )

            datasets_train.append(dataset_train)
            datasets_val.append(dataset_val)
            datasets_test.append(dataset_test)

        self.dataset_train = ConcatDataset(datasets_train)
        self.dataset_val = ConcatDataset(datasets_val)
        self.dataset_test = ConcatDataset(datasets_test)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            collate_fn=trustworthy_collate_fn,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            collate_fn=trustworthy_collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            collate_fn=trustworthy_collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
