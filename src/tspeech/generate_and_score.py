#!/usr/bin/env python3
"""
Generate audio from text with a trained RL TTS checkpoint and score with HuBERT.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.preprocessing import OrdinalEncoder

# Ensure local src/ is on PYTHONPATH when running as a script.
project_src = Path(__file__).resolve().parents[1]
if str(project_src) not in sys.path:
    sys.path.insert(0, str(project_src))

from tspeech.data.tts.dataset import _expand_abbreviations
from tspeech.model.tts import TTSModel


def _read_texts_from_csv(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path, delimiter="|", header=None, names=["wav", "text", "speaker_idx"])
    return df["text"].astype(str).tolist()


def _build_encoder(allowed_chars: str, end_token: str) -> OrdinalEncoder:
    chars = sorted(set(allowed_chars))
    encoder = OrdinalEncoder()
    encoder.fit([[x] for x in chars + [end_token]])
    return encoder


def _normalize_text(text: str, allowed_chars: str) -> str:
    lowered = text.lower()
    allowed_set = set(allowed_chars)
    return "".join(ch for ch in lowered if ch in allowed_set)


def _encode_text(encoder: OrdinalEncoder, text: str, end_token: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    chars = list(text + end_token)
    chars_idx = encoder.transform([[x] for x in chars]) + 1  # 0 is padding
    chars_idx = torch.tensor(chars_idx, dtype=torch.int64, device=device).squeeze(1)
    chars_idx = chars_idx.unsqueeze(0)  # (1, seq_len)
    chars_idx_len = torch.IntTensor([chars_idx.shape[1]]).to(device)
    return chars_idx, chars_idx_len


def _default_max_len(text: str) -> int:
    return max(50, len(text) * 10)


def _resolve_wav_path(wav_path: str, base_dir: Optional[Path]) -> Path:
    candidate = Path(wav_path)
    if candidate.is_absolute():
        return candidate
    if base_dir is None:
        raise ValueError("Relative wav path requires --tts_data_dir")
    return base_dir / candidate


def _build_style_mel(
    wav_path: Path,
    sample_rate: int,
    top_db: int,
    frame_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    wav, sr = torchaudio.load(str(wav_path))
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav.squeeze(0)

    wav_np, _ = librosa.effects.trim(
        wav.cpu().numpy(),
        top_db=top_db,
        frame_length=frame_length,
    )
    wav = torch.tensor(wav_np)

    melspectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        f_min=0.0,
        f_max=8000.0,
        n_mels=80,
        power=1.0,
        mel_scale="slaney",
        norm="slaney",
        center=True,
    )
    mel_spectrogram_style = melspectrogram(wav).swapaxes(0, 1)
    mel_spectrogram_style = torch.log(torch.clamp(mel_spectrogram_style, min=1e-5)).unsqueeze(0)
    mel_spectrogram_style_len = torch.IntTensor([mel_spectrogram_style.shape[1]])
    return mel_spectrogram_style, mel_spectrogram_style_len


def _trim_waveforms(waveforms: torch.Tensor, top_db: int = 40, frame_length: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    waveforms_np = waveforms.detach().cpu().numpy()
    trimmed = []
    lengths = []
    for wav in waveforms_np:
        wav_trim, _ = librosa.effects.trim(wav, top_db=top_db, frame_length=frame_length)
        trimmed.append(wav_trim)
        lengths.append(len(wav_trim))

    max_len = max(lengths) if lengths else 0
    padded = np.zeros((len(trimmed), max_len), dtype=waveforms_np.dtype)
    for i, wav in enumerate(trimmed):
        padded[i, : len(wav)] = wav

    return torch.tensor(padded), torch.tensor(lengths)


def _noise_gate(waveforms: torch.Tensor, gate_db: float = -40.0) -> torch.Tensor:
    # Simple per-sample RMS gate.
    if waveforms.numel() == 0:
        return waveforms
    rms = torch.sqrt(torch.mean(waveforms ** 2, dim=1, keepdim=True) + 1e-12)
    gate = (20 * torch.log10(rms + 1e-12)) > gate_db
    return waveforms * gate


def _score_with_hubert(waveforms: torch.Tensor, lengths: torch.Tensor, device: torch.device, hubert_checkpoint: str):
    from tspeech.model.htmodel import HTModel

    hubert = HTModel.load_from_checkpoint(
        hubert_checkpoint,
        hubert_model_name="facebook/hubert-base-ls960",
        trainable_layers=0,
    ).to(device)
    hubert.eval()
    for param in hubert.parameters():
        param.requires_grad = False

    resampler = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000).to(device)
    waveforms_16k = resampler(waveforms.to(device))
    lengths_16k = (lengths.float() * 16000 / 22050).long().clamp(min=1).to(device)
    seq_len = waveforms_16k.shape[1]
    attention_mask = (
        torch.arange(seq_len, device=device)[None, :] < lengths_16k[:, None]
    ).long()
    logits = hubert(wav=waveforms_16k, mask=attention_mask)
    return torch.sigmoid(logits).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audio and score with HuBERT")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/Users/yuwenyu/Desktop/tts_v2/Trustworthy_TTS/lightning_logs/tts_rl/checkpoint-epoch=0-step=201.ckpt",
        help="Path to trained TTS RL checkpoint",
    )
    parser.add_argument(
        "--hubert_checkpoint",
        type=str,
        default="/Users/yuwenyu/Desktop/tts_v2/Trustworthy_v2_first/lightning_logs/ht-finetune/version_1/checkpoints/epoch=37-step=8740.ckpt",
        help="Path to HuBERT checkpoint",
    )
    parser.add_argument(
        "--vocoder_checkpoint_dir",
        type=str,
        default="/Users/yuwenyu/Desktop/tts_v2/Trustworthy_TTS/UNIVERSAL_V1",
        help="HiFi-GAN checkpoint directory",
    )
    parser.add_argument(
        "--tts_data_dir",
        type=str,
        default=None,
        help="Base directory for relative wav paths in encoder CSVs (optional)",
    )
    parser.add_argument("--text", action="append", default=None, help="Text to synthesize (repeatable)")
    parser.add_argument("--texts_file", type=str, default=None, help="Path to text file (one line per text)")
    parser.add_argument(
        "--encoder_csv",
        type=str,
        default=None,
        help="CSV used to build text encoder (pipe-delimited). Defaults to tis_tts_data/train.csv if present.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="If set, sample this many texts from encoder_csv (or default train.csv)",
    )
    parser.add_argument("--output_dir", type=str, default="./generated_audio", help="Output directory")
    parser.add_argument("--device", type=str, default="mps", help="Device for TTS model (mps/cpu/cuda)")
    parser.add_argument("--expand_abbreviations", action="store_true", default=True, help="Expand abbreviations")
    parser.add_argument("--end_token", type=str, default="^", help="End token")
    parser.add_argument(
        "--allowed_chars",
        type=str,
        default="!'(),.:;? \\-abcdefghijklmnopqrstuvwxyz",
        help="Allowed characters for text normalization",
    )
    parser.add_argument("--max_len", type=int, default=None, help="Override max mel length")
    parser.add_argument("--trim_top_db", type=int, default=40, help="Trim threshold in dB (higher trims more)")
    parser.add_argument("--noise_gate_db", type=float, default=-40.0, help="RMS noise gate in dB")
    args = parser.parse_args()

    texts: list[str] = []
    samples: list[dict] = []
    if args.text:
        texts.extend([t for t in args.text if t.strip()])
    if args.texts_file:
        texts_path = Path(args.texts_file)
        texts.extend([line.strip() for line in texts_path.read_text().splitlines() if line.strip()])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_csv: Optional[Path] = Path(args.encoder_csv) if args.encoder_csv else None
    if encoder_csv is None:
        default_csv = Path("tis_tts_data/train.csv")
        if default_csv.exists():
            encoder_csv = default_csv

    if args.num_samples is not None:
        if encoder_csv is None:
            raise ValueError("--num_samples requires --encoder_csv or tis_tts_data/train.csv")
        df = pd.read_csv(encoder_csv, delimiter="|", header=None, names=["wav", "text", "speaker_idx"])
        if len(df) < args.num_samples:
            raise ValueError(f"Not enough texts in {encoder_csv} to sample {args.num_samples}")
        df = df.sample(args.num_samples, random_state=42)
        samples = df.to_dict(orient="records")
        texts = [row["text"] for row in samples]

    if not texts:
        raise ValueError("Provide --text/--texts_file or use --num_samples with a CSV")

    if args.expand_abbreviations:
        texts = [_expand_abbreviations(t) for t in texts]
    texts = [_normalize_text(t, args.allowed_chars) for t in texts]
    if samples:
        for idx, row in enumerate(samples):
            row["text"] = texts[idx]

    encoder = _build_encoder(args.allowed_chars, args.end_token)

    device = torch.device(args.device if torch.backends.mps.is_available() or args.device != "mps" else "cpu")

    base_dir = Path(args.tts_data_dir) if args.tts_data_dir else None

    model = TTSModel.load_from_checkpoint(
        args.checkpoint,
        map_location=device,
        use_hubert_classifier=True,
        hubert_checkpoint_path=args.hubert_checkpoint,
        use_vocoder=False,
        use_rl_training=False,
        vocoder_checkpoint_dir=args.vocoder_checkpoint_dir,
        strict=False,
    ).to(device).eval()

    from tspeech.vocoder import HiFiGANVocoder
    vocoder = HiFiGANVocoder(
        checkpoint_dir=str(args.vocoder_checkpoint_dir),
        sample_rate=22050,
        device="cpu",
    )

    results = []
    with torch.no_grad():
        for idx, text in enumerate(texts):
            chars_idx, chars_idx_len = _encode_text(encoder, text, args.end_token, device)
            max_len_override = args.max_len if args.max_len is not None else _default_max_len(text)

            mel_out, mel_postnet, gate, alignment, score = model(
                chars_idx=chars_idx,
                chars_idx_len=chars_idx_len,
                teacher_forcing=False,
                max_len_override=max_len_override,
                text=[text],
                return_trustworthiness=True,
            )

            waveform = vocoder(mel_postnet, sample_rate=22050).cpu()
            waveform, lengths = _trim_waveforms(waveform, top_db=args.trim_top_db)
            waveform = _noise_gate(waveform, gate_db=args.noise_gate_db)
            if score is None and args.hubert_checkpoint:
                score = _score_with_hubert(waveform, lengths, device, args.hubert_checkpoint).cpu()
                score = score[0]
            wav_path = output_dir / f"sample_{idx:03d}.wav"
            torchaudio.save(str(wav_path), waveform[:1], 22050)

            results.append(
                {
                    "text": text,
                    "wav_path": str(wav_path),
                    "trustworthiness_score": float(score.item()) if score is not None else None,
                }
            )

    results_path = output_dir / "scores.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {len(results)} clips to {output_dir}")
    print(f"Scores: {results_path}")


if __name__ == "__main__":
    main()
