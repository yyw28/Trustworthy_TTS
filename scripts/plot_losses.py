#!/usr/bin/env python3
"""
Plot Lightning/TensorBoard scalar losses from an events file directory.

Example:
  python scripts/plot_losses.py \
    --logdir lightning_logs/tts_rl/version_5 \
    --out loss_curves_version5.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def _load_scalars(logdir: Path) -> dict[str, pd.DataFrame]:
    ea = event_accumulator.EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    out: dict[str, pd.DataFrame] = {}
    for tag in tags:
        scalars = ea.Scalars(tag)
        out[tag] = pd.DataFrame(
            [{"step": s.step, "wall_time": s.wall_time, "value": s.value} for s in scalars]
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="Lightning log directory containing events.out.tfevents.*",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="loss_curves.png",
        help="Output image path (png/pdf/etc.)",
    )
    parser.add_argument(
        "--csv_out",
        type=str,
        default=None,
        help="Optional: write extracted scalars to this CSV (long format: tag, step, value).",
    )
    args = parser.parse_args()

    logdir = Path(args.logdir)
    if not logdir.exists():
        raise FileNotFoundError(f"logdir not found: {logdir}")

    scalars_by_tag = _load_scalars(logdir)
    if not scalars_by_tag:
        raise RuntimeError(f"No scalar tags found in {logdir}")

    # Prefer step-level tags when available.
    preferred_train = [
        "training_loss_step",
        "training_reinforce_loss_step",
        "training_tts_loss_step",
        "training_score_loss_step",
        "training_reward_step",
    ]
    preferred_val = ["val_loss", "val_tts_loss", "val_score_loss"]

    train_tags = [t for t in preferred_train if t in scalars_by_tag]
    val_tags = [t for t in preferred_val if t in scalars_by_tag]

    if not train_tags and not val_tags:
        # Fallback: plot whatever scalar tags exist.
        train_tags = sorted(list(scalars_by_tag.keys()))

    fig, axes = plt.subplots(
        2 if val_tags else 1,
        1,
        figsize=(10, 7 if val_tags else 5),
        sharex=False,
        constrained_layout=True,
    )
    # plt.subplots returns either a single Axes or a numpy array of Axes.
    if hasattr(axes, "ravel"):
        axes = list(axes.ravel())
    else:
        axes = [axes]

    ax0 = axes[0]
    for tag in train_tags:
        df = scalars_by_tag[tag]
        ax0.plot(df["step"], df["value"], label=tag)
    ax0.set_title("Training scalars")
    ax0.set_xlabel("step")
    ax0.set_ylabel("value")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best")

    if val_tags:
        ax1 = axes[1]
        for tag in val_tags:
            df = scalars_by_tag[tag]
            ax1.plot(df["step"], df["value"], marker="o", linewidth=1.5, label=tag)
        ax1.set_title("Validation scalars")
        ax1.set_xlabel("step")
        ax1.set_ylabel("value")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best")

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")

    if args.csv_out:
        rows = []
        for tag, df in scalars_by_tag.items():
            if df.empty:
                continue
            tmp = df[["step", "value"]].copy()
            tmp.insert(0, "tag", tag)
            rows.append(tmp)
        pd.concat(rows, ignore_index=True).to_csv(args.csv_out, index=False)
        print(f"Saved scalars CSV to {args.csv_out}")


if __name__ == "__main__":
    main()

