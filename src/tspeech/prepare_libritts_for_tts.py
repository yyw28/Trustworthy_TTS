#!/usr/bin/env python3
"""
Prepare LibriTTS dataset for TTS training.

Creates train.csv and val.csv with pipe-separated rows:
wav|text|speaker_idx
"""
import argparse
import os
from pathlib import Path
from typing import Optional


def _read_text_for_wav(wav_path: Path) -> Optional[str]:
    candidates = [
        wav_path.with_suffix(".normalized.txt"),
        wav_path.with_suffix(".original.txt"),
        wav_path.with_suffix(".txt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                return text
    return None


def _collect_rows(libritts_dir: Path, split_subdir: str) -> list[tuple[str, str, int]]:
    base = libritts_dir / split_subdir
    if not base.exists():
        raise FileNotFoundError(f"LibriTTS split not found: {base}")

    rows: list[tuple[str, str, int]] = []
    for root, _, files in os.walk(base):
        for filename in files:
            if not filename.endswith(".wav"):
                continue
            wav_path = Path(root) / filename
            text = _read_text_for_wav(wav_path)
            if not text:
                continue
            rel_wav = str(wav_path.relative_to(libritts_dir))
            speaker_id = wav_path.parts[-3]
            try:
                speaker_idx = int(speaker_id)
            except ValueError:
                speaker_idx = 0
            rows.append((rel_wav, text, speaker_idx))
    return rows


def _write_csv(rows: list[tuple[str, str, int]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for wav, text, speaker_idx in rows:
            f.write(f"{wav}|{text}|{speaker_idx}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LibriTTS CSVs for TTS training")
    parser.add_argument("--libritts_dir", required=True, help="Path to LibriTTS root directory")
    parser.add_argument("--output_dir", required=True, help="Directory to write CSVs")
    parser.add_argument("--train_split", default="train-clean-360", help="LibriTTS train split directory")
    parser.add_argument("--val_split", default="dev-clean", help="LibriTTS val split directory")
    args = parser.parse_args()

    libritts_dir = Path(args.libritts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _collect_rows(libritts_dir, args.train_split)
    val_rows = _collect_rows(libritts_dir, args.val_split)

    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    _write_csv(train_rows, train_csv)
    _write_csv(val_rows, val_csv)

    print(f"Wrote train.csv ({len(train_rows)} rows) -> {train_csv}")
    print(f"Wrote val.csv ({len(val_rows)} rows) -> {val_csv}")


if __name__ == "__main__":
    main()
