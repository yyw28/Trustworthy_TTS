#!/usr/bin/env python3
"""
Generate audio with a trained TTSRLModel checkpoint (RL GST policy + frozen Tacotron+GST).

Requires ``--libritts_csv`` (or ``--text`` with ``--style_wav``) so each utterance has a
reference wav for trust scores (same as training/validation).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional, TypedDict

import yaml

# Ensure local `src/` is on PYTHONPATH when running as a script.
project_src = Path(__file__).resolve().parents[1]
if str(project_src) not in sys.path:
    sys.path.insert(0, str(project_src))

import tspeech._torchvision_first  # noqa: F401

import librosa
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as ta_f
import unidecode
from sklearn.preprocessing import OrdinalEncoder
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from tspeech.data.tts.dataset import _expand_abbreviations
from tspeech.model.rl_gst_policy_option1 import RLGSTPolicy
from tspeech.model.tts_rl import TTSRLModel

HIFI_GAN_SR = 22050
HUBERT_SR = 16000


class LibriTTSRow(TypedDict, total=False):
    wav: str
    speaker_id: str
    text: str
    duration: str
    speaker_idx: str


def _preprocess_transcripts_like_dataset(
    raw_texts: list[str],
    allowed_chars: str,
    end_token: Optional[str],
    expand_abbreviations: bool,
) -> list[str]:
    """Match ``TTSDataset`` text pipeline (unidecode, allowed_chars filter, abbreviations, end token)."""
    allowed_chars_re = re.compile(f"[^{re.escape(allowed_chars)}]+")
    out: list[str] = []
    for t in raw_texts:
        t = allowed_chars_re.sub("", unidecode.unidecode(t).lower())
        if expand_abbreviations:
            t = _expand_abbreviations(t)
        if end_token is not None:
            t = t + end_token
        out.append(t)
    return out


def _make_char_encoder(allowed_chars: str, end_token: Optional[str]) -> OrdinalEncoder:
    """Same OrdinalEncoder setup as ``TTSDataset`` (fixed vocab from allowed_chars + end_token)."""
    chars = sorted(set(allowed_chars + (end_token or "")))
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if end_token is None:
        enc.fit([[x] for x in chars])
    else:
        enc.fit([[x] for x in chars + [end_token]])
    return enc


def _encode_batch_ordinal(
    encoder: OrdinalEncoder,
    texts: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode already-processed transcripts (each includes end token if used)."""
    seqs: list[torch.Tensor] = []
    for text in texts:
        arr = encoder.transform([[ch] for ch in text]).astype("int64").ravel() + 1
        seqs.append(torch.tensor(arr, dtype=torch.int64))
    lens = torch.tensor([s.numel() for s in seqs], dtype=torch.int64, device=device)
    x = pad_sequence(seqs, batch_first=True, padding_value=0).to(device=device)
    return x, lens


def _read_libritts_pipe_csv(path: Path) -> list[LibriTTSRow]:
    rows: list[LibriTTSRow] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if line.lower().startswith("wav|"):
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        wav, speaker_id, text, duration, speaker_idx = parts[:5]
        rows.append(
            {
                "wav": wav.strip(),
                "speaker_id": speaker_id.strip(),
                "text": text.strip().strip('"'),
                "duration": duration.strip(),
                "speaker_idx": speaker_idx.strip(),
            }
        )
    return rows


def _resolve_wav_path(wav_path: str, base_dir: Optional[Path]) -> str:
    p = Path(wav_path)
    if p.is_absolute():
        return str(p)
    if base_dir is None:
        raise ValueError("Relative wav paths require --libritts_dir")
    return str(base_dir / p)


def _reference_wav_to_mel_like_dataset(
    wav_path: str,
    sample_rate: int,
    n_mels: int,
    trim: bool,
    trim_top_db: int,
    trim_frame_length: int,
    silence: int,
) -> tuple[torch.Tensor, int]:
    """Match ``TTSDataset`` reference mel (librosa load + trim + mel)."""
    wav_np, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
    wav = torch.tensor(wav_np, dtype=torch.float32)
    if trim:
        trimmed, _ = librosa.effects.trim(
            wav.numpy(),
            top_db=trim_top_db,
            frame_length=trim_frame_length,
        )
        wav = torch.tensor(trimmed, dtype=torch.float32)
    wav = F.pad(wav, (0, silence))
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        f_min=0.0,
        f_max=8000.0,
        n_mels=n_mels,
        power=1.0,
        mel_scale="slaney",
        norm="slaney",
        center=True,
    )
    mel = mel_tf(wav).swapaxes(0, 1)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel, mel.shape[0]


def _batch_reference_mels(
    rows: list[LibriTTSRow],
    indices: range,
    base_dir: Path,
    sample_rate: int,
    n_mels: int,
    device: torch.device,
    trim: bool,
    trim_top_db: int,
    trim_frame_length: int,
    silence: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Padded mel batch (B, T_max, n_mels) and lengths (B)."""
    mels: list[torch.Tensor] = []
    lens: list[int] = []
    for j in indices:
        if j >= len(rows) or not rows[j].get("wav"):
            raise ValueError("Each CSV row must have a wav path for RL reference mels.")
        ref_path = _resolve_wav_path(str(rows[j]["wav"]), base_dir)
        mel, ln = _reference_wav_to_mel_like_dataset(
            ref_path,
            sample_rate=sample_rate,
            n_mels=n_mels,
            trim=trim,
            trim_top_db=trim_top_db,
            trim_frame_length=trim_frame_length,
            silence=silence,
        )
        mels.append(mel)
        lens.append(ln)
    max_t = max(lens)
    b = len(mels)
    batch = torch.zeros(b, max_t, n_mels, device=device, dtype=torch.float32)
    for i, mel in enumerate(mels):
        t = mel.shape[0]
        batch[i, :t, :] = mel.to(device)
    mel_lens = torch.tensor(lens, dtype=torch.int64, device=device)
    return batch, mel_lens


def _rl_gst_weights_deterministic(rl_policy: RLGSTPolicy, bert_embeddings: torch.Tensor) -> torch.Tensor:
    """Mean-action (mu) → softmax; reproducible eval without rsample."""
    batch_size = bert_embeddings.shape[0]
    h = rl_policy.trunk(bert_embeddings)
    mu = rl_policy.mu_head(h).view(batch_size * rl_policy.gst_heads, rl_policy.gst_token_num)
    return F.softmax(mu / rl_policy.temperature, dim=1)


def _load_tts_rl_model(
    rl_ckpt_path: str,
    device: torch.device,
    tts_checkpoint_path: str,
    vocoder_checkpoint_dir: str,
    hubert_checkpoint_path: str,
    rl_temperature: float,
    tts_config_path: str,
) -> TTSRLModel:
    """Load TTSRLModel; supports checkpoints without Lightning ``hyper_parameters``."""
    ckpt = torch.load(rl_ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters") or {}

    if isinstance(hp, dict) and hp.get("tts_checkpoint_path"):
        model = TTSRLModel.load_from_checkpoint(
            rl_ckpt_path,
            map_location="cpu",
            tts_config_path=tts_config_path,
        )
    else:
        model = TTSRLModel(
            tts_checkpoint_path=tts_checkpoint_path,
            vocoder_checkpoint_dir=vocoder_checkpoint_dir,
            hubert_checkpoint_path=hubert_checkpoint_path,
            rl_temperature=rl_temperature,
            tts_config_path=tts_config_path,
        )
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            print(f"load_state_dict missing keys (first 10): {missing[:10]}")
        if unexpected:
            print(f"load_state_dict unexpected keys (first 10): {unexpected[:10]}")

    return model.to(device).eval()


def _style_from_gst_weights(
    gst_weights: torch.Tensor,
    tts_model,
    batch_size: int,
) -> torch.Tensor:
    stl = tts_model.tts.gst.stl
    style = torch.bmm(
        gst_weights.unsqueeze(1),
        torch.tanh(stl.embed)[None, :, :].expand(batch_size * stl.num_heads, -1, -1),
    )
    style = style.transpose(0, 1).view(batch_size, stl.token_embedding_size)
    return stl.attention.out_proj(style).unsqueeze(1)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate with TTSRLModel checkpoint (RL GST)")
    p.add_argument(
        "--checkpoint",
        required=True,
        help="RL Lightning checkpoint, e.g. checkpoint-epoch=2-step=10041.ckpt",
    )
    p.add_argument(
        "--tts_checkpoint_path",
        default="/workplace/checkpoints/checkpoints/checkpoint-epoch=85-val_loss=0.25344.ckpt",
        help="Frozen Tacotron+GST ckpt (same as training); required if RL ckpt has no hyper_parameters",
    )
    p.add_argument(
        "--vocoder_checkpoint_dir",
        default="/workplace/checkpoints/checkpoints/UNIVERSAL_V1",
        help="HiFi-GAN checkpoint dir (UNIVERSAL_V1); same as training",
    )
    p.add_argument(
        "--hubert_checkpoint_path",
        default="/workplace/checkpoints/checkpoints/hubert-checkpoint-epoch=44-validation_f1=0.84615.ckpt",
        help="HuBERT trustworthiness ckpt (same as training)",
    )
    p.add_argument(
        "--rl_temperature",
        type=float,
        default=1.0,
        help="Policy temperature (must match training; config/tts-rl.json often uses 1.0)",
    )
    p.add_argument(
        "--config",
        required=True,
        help="YAML file containing the config used for the frozen Tacotron (as in generate_and_score_tactron.py)",
    )
    p.add_argument(
        "--reference_sample_rate",
        type=int,
        default=22050,
        help="SR for reference wav→mel (must match RL TTSDatamodule data.sample_rate; often 22050 while Tacotron YAML is 16 kHz)",
    )
    p.add_argument("--libritts_dir", default="/workplace/LibriTTS", help="Base directory for LibriTTS wav paths")
    p.add_argument(
        "--libritts_csv",
        default=None,
        help="Optional LibriTTS CSV: wav|speaker_id|text|duration|speaker_idx",
    )
    p.add_argument(
        "--style_wav",
        default=None,
        help="With --text/--texts_file only: reference wav for trust scores (absolute or relative to --libritts_dir)",
    )
    p.add_argument("--text", action="append", default=None, help="Text to synthesize (repeatable)")
    p.add_argument("--texts_file", default=None, help="Text file (one line per text)")
    p.add_argument("--output_dir", default="./tacotron_rl_test_audio")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size for faster generation on GPU; lower if OOM")
    p.add_argument("--speaker_idx", type=int, default=0)
    p.add_argument("--end_token", default="^")
    p.add_argument("--allowed_chars", default="!'(),.:;? \\-abcdefghijklmnopqrstuvwxyz")
    p.add_argument("--max_len", type=int, default=800)
    p.add_argument(
        "--no_expand_abbreviations",
        action="store_true",
        help="Disable TTSDataset-style abbreviation expansion (training uses expansion by default)",
    )
    p.add_argument(
        "--no_trim_reference",
        action="store_true",
        help="Disable librosa trim on reference wavs (TTSDataset trims by default)",
    )
    p.add_argument("--trim_top_db", type=int, default=40)
    p.add_argument("--trim_frame_length", type=int, default=1024)
    p.add_argument("--silence", type=int, default=0, help="Pad this many samples at end of ref wav before mel")
    p.add_argument("--device", default="auto", help="auto/cuda/cpu")
    p.add_argument(
        "--no_scores",
        action="store_true",
        help="Skip *_16k.wav and scores.json (tw_classifier on generated audio)",
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(args.libritts_dir) if args.libritts_dir else None
    rows: list[LibriTTSRow] = []
    texts: list[str] = []
    if args.libritts_csv:
        rows = _read_libritts_pipe_csv(Path(args.libritts_csv))
        texts = [r["text"] for r in rows if r.get("text")]
    else:
        if args.text:
            texts.extend([t.strip() for t in args.text if t.strip()])
        if args.texts_file:
            texts.extend([t.strip() for t in Path(args.texts_file).read_text().splitlines() if t.strip()])
        if not args.style_wav:
            raise ValueError("Without --libritts_csv, pass --style_wav for reference trust scores.")

    if not texts:
        raise ValueError("Provide --text/--texts_file or --libritts_csv")

    with open(args.config) as infile:
        yaml_cfg = yaml.safe_load(infile)
    tacotron_sample_rate = int(yaml_cfg["data"]["sample_rate"])

    end_tok: Optional[str] = args.end_token if args.end_token else None
    transcripts = _preprocess_transcripts_like_dataset(
        texts,
        args.allowed_chars,
        end_tok,
        expand_abbreviations=not args.no_expand_abbreviations,
    )
    char_encoder = _make_char_encoder(args.allowed_chars, end_tok)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    model = _load_tts_rl_model(
        args.checkpoint,
        device,
        tts_checkpoint_path=args.tts_checkpoint_path,
        vocoder_checkpoint_dir=args.vocoder_checkpoint_dir,
        hubert_checkpoint_path=args.hubert_checkpoint_path,
        rl_temperature=float(args.rl_temperature),
        tts_config_path=args.config,
    )

    print(
        f"YAML data.sample_rate (frozen Tacotron)={tacotron_sample_rate}; "
        f"reference_sample_rate={int(args.reference_sample_rate)}; "
        f"vocoder output SR={int(model.VOCODER_SAMPLE_RATE)} (expect {HIFI_GAN_SR})",
        flush=True,
    )

    tts = model.tts
    n_mels = int(tts.hparams["num_mels"])
    num_chars = int(tts.hparams["num_chars"])
    speaker_count = int(tts.hparams.get("speaker_count", 1))

    for t in transcripts:
        ids = char_encoder.transform([[ch] for ch in t]).astype("int64").ravel() + 1
        # Tacotron embedding is size num_chars+1 with padding_idx=0; valid char ids are 1..num_chars.
        if (ids > num_chars).any() or (ids < 1).any():
            raise ValueError(
                f"Character id out of range for num_chars={num_chars} in text {t[:80]!r}..."
            )

    scores: list[dict] = []
    bs = max(1, int(args.batch_size))
    ref_sr = int(args.reference_sample_rate)
    sr_vocoder = int(model.VOCODER_SAMPLE_RATE)

    def _out_name(i: int) -> str:
        if i < len(rows) and rows[i].get("wav"):
            ref_wav_str = str(rows[i]["wav"])
            name = ref_wav_str.replace("\\", "_").replace("/", "_")
            return name if name.lower().endswith(".wav") else (name + ".wav")
        return f"sample_{i:03d}.wav"

    with torch.inference_mode():
        trim_ref = not args.no_trim_reference
        for start in range(0, len(transcripts), bs):
            end = min(start + bs, len(transcripts))
            batch_transcripts = transcripts[start:end]
            bsz = end - start

            if rows:
                ref_mel, ref_mel_len = _batch_reference_mels(
                    rows,
                    range(start, end),
                    base_dir,
                    ref_sr,
                    n_mels,
                    device,
                    trim=trim_ref,
                    trim_top_db=int(args.trim_top_db),
                    trim_frame_length=int(args.trim_frame_length),
                    silence=int(args.silence),
                )
            else:
                ref_path = _resolve_wav_path(str(args.style_wav), base_dir)
                mel0, ln0 = _reference_wav_to_mel_like_dataset(
                    ref_path,
                    sample_rate=ref_sr,
                    n_mels=n_mels,
                    trim=trim_ref,
                    trim_top_db=int(args.trim_top_db),
                    trim_frame_length=int(args.trim_frame_length),
                    silence=int(args.silence),
                )
                mel0 = mel0.to(device)
                ref_mel = mel0.unsqueeze(0).expand(bsz, -1, -1).contiguous()
                ref_mel_len = torch.full((bsz,), ln0, dtype=torch.int64, device=device)

            ref_mel_voc = model._rescale_mel_time_for_vocoder(ref_mel)
            wav_ref = model.vocoder(ref_mel_voc)
            seq_len = wav_ref.shape[1]
            wav_lens = model._wav_samples_from_mel_frames(
                ref_mel_len,
                float(model.vocoder_mel_time_scale),
                seq_len,
            )
            time = torch.arange(seq_len, device=device)[None, :]
            mask = (time < wav_lens[:, None]).long()

            tw_scores = torch.sigmoid(model.tw_classifier(wav=wav_ref, mask=mask)).squeeze(-1)

            bert_embeddings = model.bert_gst_encoder(score=tw_scores, text=batch_transcripts)
            gst_weights = _rl_gst_weights_deterministic(model.rl_policy, bert_embeddings)
            style = _style_from_gst_weights(gst_weights, model, bsz)

            speaker_idxs: list[int] = []
            for j in range(start, end):
                sp = int(args.speaker_idx)
                if rows and j < len(rows) and rows[j].get("speaker_idx"):
                    sp = int(rows[j]["speaker_idx"])
                if sp < 0 or sp >= speaker_count:
                    sp = 0
                speaker_idxs.append(sp)
            speaker = torch.tensor(speaker_idxs, dtype=torch.int64, device=device)

            x, x_len = _encode_batch_ordinal(char_encoder, batch_transcripts, device=device)

            _, mel_post, gate, _ = model.tts(
                chars_idx=x,
                chars_idx_len=x_len,
                teacher_forcing_dropout=0.0,
                teacher_forcing=False,
                speaker_id=speaker,
                max_len_override=int(args.max_len),
                style=style,
            )

            mel_for_vocoder = model._rescale_mel_time_for_vocoder(mel_post)

            wav = model.vocoder(mel_for_vocoder)
            if wav.dim() == 3:
                wav = wav.squeeze(1)

            mel_frames = TTSRLModel._mel_frames_from_pred_gate(gate, gate.shape[1])
            pred_len = model._wav_samples_from_mel_frames(
                mel_frames,
                float(model.vocoder_mel_time_scale),
                wav.shape[-1],
            )
            pred_len = pred_len.clamp(min=256, max=wav.shape[-1])

            for bi, j in enumerate(range(start, end)):
                out_name = _out_name(j)
                out_path = out_dir / out_name
                n = int(pred_len[bi].item())
                wav_0 = wav[bi, :n].detach().cpu().numpy()
                sf.write(str(out_path), wav_0, sr_vocoder)

                if not args.no_scores:
                    seg = wav[bi : bi + 1, :n]
                    wav16_seg = ta_f.resample(
                        seg,
                        orig_freq=sr_vocoder,
                        new_freq=HUBERT_SR,
                    )
                    wav16_0 = wav16_seg.squeeze(0)
                    out_name_16k = (
                        out_name[:-4] + "_16k.wav"
                        if out_name.lower().endswith(".wav")
                        else out_name + "_16k.wav"
                    )
                    out_path_16k = out_dir / out_name_16k
                    sf.write(str(out_path_16k), wav16_0.detach().cpu().numpy(), HUBERT_SR)

                    mask_s = torch.ones((1, wav16_0.shape[0]), device=device, dtype=torch.long)
                    score = torch.sigmoid(
                        model.tw_classifier(wav=wav16_0.unsqueeze(0), mask=mask_s)
                    ).squeeze(-1)[0].item()
                    scores.append(
                        {
                            "wav_path_orig": str(out_path),
                            "wav_path_16k": str(out_path_16k),
                            "generated_trustworthiness_score": float(score),
                        }
                    )

    if scores:
        scores_path = out_dir / "scores.json"
        scores_path.write_text(json.dumps(scores, indent=2))
        print(f"Saved scores to {scores_path}")

    print(f"Saved {len(transcripts)} wavs to {out_dir}")


if __name__ == "__main__":
    main()

