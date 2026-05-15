"""Utility to warm-start the RL TTS model from a Tacotron2+GST checkpoint.

This script:
- Builds a `TTSModel` using a JSON config (typically `config/tts-rl.json`)
- Loads a *Tacotron2+GST-only* checkpoint (no RL/BERT/HuBERT modules)
- Copies all parameters whose names and shapes match into the new model
- Saves a new checkpoint that is fully compatible with the RL TTSModel

Usage example:

    python -m tspeech.make_rl_warmstart_checkpoint \
        --config config/tts-rl.json \
        --tacotron_ckpt /Users/yuwenyu/Desktop/tts_v2/Trustworthy_TTS_v3/checkpoint-epoch=37.ckpt \
        --output_ckpt /Users/yuwenyu/Desktop/tts_v2/Trustworthy_TTS_v3/checkpoint-rl-warmstart.ckpt

Then train with:

    tts fit --config config/tts-rl.json \
      --ckpt_path /Users/yuwenyu/Desktop/tts_v2/Trustworthy_TTS_v3/checkpoint-rl-warmstart.ckpt \
      --trainer.max_epochs=3 \
      --model.hubert_checkpoint_path=/Users/yuwenyu/Desktop/tts_v2/Trustworthy_TTS_v3/hubert-checkpoint-epoch=44-validation_f1=0.84615.ckpt \
      --model.vocoder_checkpoint_dir=/Users/yuwenyu/Desktop/tts_v2/Trustworthy_TTS_v3/UNIVERSAL_V1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch

import lightning.pytorch as pl
from tspeech.model.tts import TTSModel


def build_model_from_config(
    config_path: str,
    *,
    hubert_checkpoint_path: str | None = None,
    vocoder_checkpoint_dir: str | None = None,
) -> TTSModel:
    """Instantiate `TTSModel` from a Lightning JSON config."""
    path = Path(config_path)
    with path.open("r") as f:
        cfg: Dict[str, Any] = json.load(f)

    model_args = dict(cfg.get("model", {}))
    if not model_args:
        raise ValueError(f"No 'model' section found in config: {config_path}")
    if hubert_checkpoint_path is not None:
        model_args["hubert_checkpoint_path"] = hubert_checkpoint_path
    if vocoder_checkpoint_dir is not None:
        model_args["vocoder_checkpoint_dir"] = vocoder_checkpoint_dir

    return TTSModel(**model_args)


def make_warmstart_checkpoint(
    config_path: str,
    tacotron_ckpt_path: str,
    output_ckpt_path: str,
    *,
    hubert_checkpoint_path: str | None = None,
    vocoder_checkpoint_dir: str | None = None,
) -> None:
    """Create an RL-compatible checkpoint warm-started from Tacotron2+GST weights."""
    print(f"Loading RL TTS model from config: {config_path}")
    model = build_model_from_config(
        config_path,
        hubert_checkpoint_path=hubert_checkpoint_path,
        vocoder_checkpoint_dir=vocoder_checkpoint_dir,
    )
    model_state = model.state_dict()

    tacotron_ckpt_path = str(tacotron_ckpt_path)
    print(f"Loading Tacotron2+GST checkpoint: {tacotron_ckpt_path}")
    ckpt = torch.load(tacotron_ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    loaded_keys = []
    skipped_missing = []
    skipped_shape = []

    for key, value in state_dict.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                model_state[key] = value
                loaded_keys.append(key)
            else:
                skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
        else:
            skipped_missing.append(key)

    print(f"Loaded {len(loaded_keys)} keys from Tacotron checkpoint into RL model.")
    if skipped_shape:
        print(f"Skipped {len(skipped_shape)} keys due to shape mismatch:")
        for name, old_shape, new_shape in skipped_shape[:10]:
            print(f"  - {name}: checkpoint {old_shape} vs model {new_shape}")
        if len(skipped_shape) > 10:
            print(f"  ... and {len(skipped_shape) - 10} more with shape mismatches.")

    if skipped_missing:
        print(f"Skipped {len(skipped_missing)} keys not present in RL model state_dict "
              f"(this is expected for optimizer, scheduler, etc.).")

    # Load the merged state dict back into the model to ensure consistency.
    model.load_state_dict(model_state)

    output_path = Path(output_ckpt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create optimizer state dict that matches what Lightning CLI will create.
    # We create it AFTER loading weights so parameter groups match.
    optimizer = model.configure_optimizers()
    optimizers = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]
    if isinstance(optimizers[0], tuple):
        optimizers = [o for o, _ in optimizers]
    
    # Get optimizer state dicts - these will be fresh (no momentum) but have correct structure.
    optimizer_states = []
    for opt in optimizers:
        opt_state = opt.state_dict()
        # Ensure state dict has the right structure but empty state (fresh start).
        # Clear any momentum/state to ensure compatibility.
        opt_state["state"] = {}
        optimizer_states.append(opt_state)
    
    lr_schedulers: list = []  # config has no lr_scheduler

    # Save in Lightning checkpoint format so trainer can resume without KeyError.
    new_ckpt = {
        "state_dict": model.state_dict(),
        "hyper_parameters": getattr(model, "hparams", {}),
        "pytorch-lightning_version": pl.__version__,
        "epoch": 0,
        "global_step": 0,
        "optimizer_states": optimizer_states,
        "lr_schedulers": lr_schedulers,
    }

    torch.save(new_ckpt, output_path)
    print(f"Saved RL warm-start checkpoint to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an RL TTS warm-start checkpoint from a Tacotron2+GST checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/tts-rl.json",
        help="Path to RL TTS JSON config (default: config/tts-rl.json).",
    )
    parser.add_argument(
        "--tacotron_ckpt",
        type=str,
        required=True,
        help="Path to Tacotron2+GST checkpoint (e.g., checkpoint-epoch=37.ckpt).",
    )
    parser.add_argument(
        "--output_ckpt",
        type=str,
        required=True,
        help="Path to write the new RL warm-start checkpoint.",
    )
    parser.add_argument(
        "--hubert_checkpoint_path",
        type=str,
        default=None,
        help="HuBERT checkpoint path (required if config has use_rl_training=True).",
    )
    parser.add_argument(
        "--vocoder_checkpoint_dir",
        type=str,
        default=None,
        help="Vocoder checkpoint directory (optional).",
    )

    args = parser.parse_args()

    make_warmstart_checkpoint(
        config_path=args.config,
        tacotron_ckpt_path=args.tacotron_ckpt,
        output_ckpt_path=args.output_ckpt,
        hubert_checkpoint_path=args.hubert_checkpoint_path,
        vocoder_checkpoint_dir=args.vocoder_checkpoint_dir,
    )


if __name__ == "__main__":
    main()

