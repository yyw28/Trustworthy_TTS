import torch
from lightning.pytorch.cli import LightningCLI

from tspeech.data.tts import TTSDatamodule
from tspeech.model.tts_rl import TTSRLModel


def cli_main():
    torch.set_float32_matmul_precision("high")
    cli = LightningCLI(TTSRLModel, TTSDatamodule)


if __name__ == "__main__":
    cli_main()
