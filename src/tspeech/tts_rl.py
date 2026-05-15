import tspeech._torchvision_first  # noqa: F401

import torch
from lightning.pytorch.cli import LightningCLI

from tspeech.data.tts import TTSDatamodule
from tspeech.model.tts_rl import TTSRLModel

# Which train/val CSVs are used is printed once from ``TTSDatamodule.setup`` when ``fit`` runs
# (look for the line starting with ``TTSDatamodule: train=...``).


def cli_main():
    torch.set_float32_matmul_precision("high")
    LightningCLI(TTSRLModel, TTSDatamodule, seed_everything_default=42)


if __name__ == "__main__":
    cli_main()
