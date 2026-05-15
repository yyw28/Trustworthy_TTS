import tspeech._torchvision_first  # noqa: F401

from lightning.pytorch.cli import LightningCLI
import torch

# simple demo classes for your convenience
from tspeech.model.htmodel import HTModel
from tspeech.data.ht_datamodule import HTDataModule


def cli_main():
    torch.set_float32_matmul_precision("high")
    cli = LightningCLI(HTModel, HTDataModule)


if __name__ == "__main__":
    cli_main()
