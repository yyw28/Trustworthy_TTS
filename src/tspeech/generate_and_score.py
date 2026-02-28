#!/usr/bin/env python3
"""
Generate audio from text with a trained RL TTS checkpoint and score with HuBERT.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from sklearn.preprocessing import OrdinalEncoder

# Ensure local src/ is on PYTHONPATH when running as a script.
project_src = Path(__file__).resolve().parents[1]
if str(project_src) not in sys.path:
    sys.path.insert(0, str(project_src))

# Import torchvision before Lightning to avoid circular import (torchvision.extension)
try:
    import torchvision  # noqa: F401
except Exception:
    pass

# Inline _expand_abbreviations to avoid tspeech.data.tts import chain (Lightning -> torchmetrics -> torchvision)
_ABBREVIATIONS = [
    (re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"),
        ("co", "company"), ("jr", "junior"), ("maj", "major"), ("gen", "general"),
        ("drs", "doctors"), ("rev", "reverend"), ("lt", "lieutenant"), ("hon", "honorable"),
        ("sgt", "sergeant"), ("capt", "captain"), ("esq", "esquire"), ("ltd", "limited"),
        ("col", "colonel"), ("ft", "fort"),
    ]
]


def _expand_abbreviations(text: str) -> str:
    for regex, replacement in _ABBREVIATIONS:
        text = re.sub(regex, replacement, text)
    return text


from tspeech.model.tts import TTSModel
from tspeech.model.tts_rl import TTSRLModel


def _read_texts_from_csv(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path, delimiter="|", header=None, names=["wav", "text", "speaker_idx"])
    return df["text"].astype(str).tolist()


def _read_libritts_csv(csv_path: Path) -> list[dict]:
    """Read LibriTTS CSV (wav|speaker_id|text|duration|speaker_idx with header)."""
    df = pd.read_csv(csv_path, delimiter="|", header=0, names=["wav", "speaker_id", "text", "duration", "speaker_idx"])
    return df.to_dict(orient="records")


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
        default="lightning_logs/tts_rl/version_2/checkpoints/checkpoint-epoch=2-step=40164.ckpt",
        help="Path to trained TTS RL checkpoint",
    )
    parser.add_argument(
        "--tts_checkpoint_path",
        type=str,
        default="/workplace/checkpoints/checkpoints/checkpoint-epoch=37.ckpt",
        help="Base TTS checkpoint (required to bootstrap TTSRLModel; can be same as warmstart)",
    )
    parser.add_argument(
        "--hubert_checkpoint",
        type=str,
        default="/workplace/checkpoints/checkpoints/hubert-checkpoint-epoch=44-validation_f1=0.84615.ckpt",
        help="Path to HuBERT checkpoint",
    )
    parser.add_argument(
        "--vocoder_checkpoint_dir",
        type=str,
        default="/workplace/checkpoints/checkpoints/UNIVERSAL_V1",
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
        "--libritts_csv",
        type=str,
        default=None,
        help="LibriTTS CSV (wav|speaker_id|text|duration|speaker_idx with header). Use all rows or --num_samples.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="If set, sample this many texts from encoder_csv/libritts_csv (or default train.csv)",
    )
    parser.add_argument("--output_dir", type=str, default="./generated_audio", help="Output directory")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for TTS model (auto/cuda/mps/cpu). 'auto' uses cuda if available, else mps, else cpu.",
    )
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

    libritts_csv: Optional[Path] = Path(args.libritts_csv) if args.libritts_csv else None
    if libritts_csv is not None:
        if not libritts_csv.exists():
            raise FileNotFoundError(f"LibriTTS CSV not found: {libritts_csv}")
        samples = _read_libritts_csv(libritts_csv)
        if args.num_samples is not None:
            if len(samples) < args.num_samples:
                raise ValueError(f"Not enough rows in {libritts_csv} to sample {args.num_samples}")
            import random
            samples = random.sample(samples, args.num_samples)
        texts = [str(row["text"]).strip().strip('"') for row in samples]

    else:
        encoder_csv: Optional[Path] = Path(args.encoder_csv) if args.encoder_csv else None
        if encoder_csv is None:
            default_csv = Path("tis_tts_data/train.csv")
            if default_csv.exists():
                encoder_csv = default_csv

        if args.num_samples is not None:
            if encoder_csv is None:
                raise ValueError("--num_samples requires --encoder_csv or --libritts_csv")
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

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using device: {device}")
    else:
        device = torch.device(args.device)

    base_dir = Path(args.tts_data_dir) if args.tts_data_dir else None

    # Load TTSRLModel (RL checkpoints require bootstrap TTS + init args)
    model = TTSRLModel.load_from_checkpoint(
        args.checkpoint,
        map_location=device,
        tts_checkpoint_path=args.tts_checkpoint_path,
        vocoder_checkpoint_dir=args.vocoder_checkpoint_dir,
        hubert_checkpoint_path=args.hubert_checkpoint,
        strict=False,
    ).to(device).eval()

    vocoder = model.vocoder

    def _output_basename(idx: int, row: Optional[dict]) -> str:
        if row and "wav" in row:
            return Path(row["wav"]).stem
        return f"sample_{idx:03d}"

    results = []
    with torch.no_grad():
        for idx, text in enumerate(texts):
            chars_idx, chars_idx_len = _encode_text(encoder, text, args.end_token, device)
            max_len_override = args.max_len if args.max_len is not None else _default_max_len(text)

            # Get style from RL policy: dummy score (0.5) -> BERT -> policy -> GST style
            dummy_score = torch.tensor([0.5], device=device, dtype=torch.float32).expand(chars_idx.shape[0])
            bert_embeddings = model.bert_gst_encoder(score=dummy_score, text=[text])
            gst_weights, _, _ = model.rl_policy(bert_embeddings, deterministic=True)
            batch_size = chars_idx.shape[0]
            style = model.tts.gst.stl.attention.out_proj(
                torch.matmul(gst_weights, torch.tanh(model.tts.gst.stl.embed)).reshape(
                    batch_size, 1, -1
                )
            )

            # speaker_id: use from libritts row if available, else 0
            row = samples[idx] if idx < len(samples) else None
            speaker_idx = int(row["speaker_idx"]) if row and "speaker_idx" in row else 0
            speaker_id = torch.tensor([speaker_idx], dtype=torch.long, device=device)

            mel_out, mel_postnet, gate, alignment = model(
                chars_idx=chars_idx,
                chars_idx_len=chars_idx_len,
                teacher_forcing_dropout=0.0,
                teacher_forcing=False,
                max_len_override=max_len_override,
                speaker_id=speaker_id,
                style=style,
            )
            score = None  # Computed below via HuBERT if needed

            waveform = vocoder(mel_postnet).cpu()
            waveform, lengths = _trim_waveforms(waveform, top_db=args.trim_top_db)
            waveform = _noise_gate(waveform, gate_db=args.noise_gate_db)
            if score is None and args.hubert_checkpoint:
                score = _score_with_hubert(waveform, lengths, device, args.hubert_checkpoint).cpu()
                score = score[0]
            basename = _output_basename(idx, row)
            wav_path = output_dir / f"{basename}.wav"
            sf.write(str(wav_path), waveform[0].cpu().numpy(), 22050)

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
