#!/usr/bin/env python3
"""
Generate audio with a trained TTSRLModel checkpoint (RL GST policy + frozen Tacotron+GST).

Requires ``--libritts_csv`` (or ``--text`` with ``--style_wav``) so each utterance has a
reference wav for trust scores (same as training/validation).
"""

from __future__ import annotations

import argparse
import json
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


def _normalize_text(text: str, allowed_chars: str) -> str:
    """Match ``generate_and_score_tactron.py`` / Libri-style inference (lowercase + allowed set)."""
    allowed = set(allowed_chars)
    return "".join(ch for ch in text.lower() if ch in allowed)


def _prepare_text_lines(
    raw_texts: list[str],
    allowed_chars: str,
    expand_abbreviations: bool,
) -> list[str]:
    """Abbreviations (optional) + same normalization as Tacotron generation script."""
    out: list[str] = []
    for t in raw_texts:
        if expand_abbreviations:
            t = _expand_abbreviations(t)
        out.append(_normalize_text(t, allowed_chars))
    return out


def _build_char_to_id_from_texts(texts: list[str], end_token: str) -> dict[str, int]:
    """Dynamic vocab from the current texts + end token (same as ``generate_and_score_tactron.py``)."""
    chars_set: set[str] = set()
    for t in texts:
        chars_set.update(t)
    if end_token:
        chars_set.add(end_token)
    chars = sorted(chars_set)
    return {ch: i + 1 for i, ch in enumerate(chars)}  # 0 reserved for padding


def _encode_batch(
    char_to_id: dict[str, int],
    texts: list[str],
    end_token: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode lines **without** trailing end token; append ``end_token`` like Tacotron script."""
    suffix = end_token if end_token else ""
    seqs = [
        torch.tensor([char_to_id[ch] for ch in (t + suffix)], dtype=torch.int64)
        for t in texts
    ]
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
) -> TTSRLModel:
    """Load TTSRLModel; supports checkpoints without Lightning ``hyper_parameters``."""
    ckpt = torch.load(rl_ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters") or {}

    if isinstance(hp, dict) and hp.get("tts_checkpoint_path"):
        model = TTSRLModel.load_from_checkpoint(
            rl_ckpt_path,
            map_location="cpu",
        )
    else:
        model = TTSRLModel(
            tts_checkpoint_path=tts_checkpoint_path,
            vocoder_checkpoint_dir=vocoder_checkpoint_dir,
            hubert_checkpoint_path=hubert_checkpoint_path,
            rl_temperature=rl_temperature,
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
    p.add_argument(
        "--allowed_chars",
        default="!'(),.:;? \\-0123456789abcdefghijklmnopqrstuvwxyz",
        help="Allowed symbols after lowercasing (same filter as generate_and_score_tactron.py); "
        "char ids are built dynamically from this run's texts.",
    )
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
    scale_tacotron_hifi_gan = (
        1.0 if tacotron_sample_rate == HIFI_GAN_SR else HIFI_GAN_SR / float(tacotron_sample_rate)
    )

    end_str = args.end_token if args.end_token else ""
    texts_norm = _prepare_text_lines(
        texts,
        args.allowed_chars,
        expand_abbreviations=not args.no_expand_abbreviations,
    )
    transcripts = [t + end_str for t in texts_norm]
    char_to_id = _build_char_to_id_from_texts(texts_norm, end_str)

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
    )

    print(
        f"YAML data.sample_rate (frozen Tacotron)={tacotron_sample_rate}; "
        f"mel→vocoder scale={scale_tacotron_hifi_gan:.6g}; "
        f"reference_sample_rate={int(args.reference_sample_rate)}; "
        f"vocoder output SR={HIFI_GAN_SR}",
        flush=True,
    )

    tts = model.tts
    n_mels = int(tts.hparams["num_mels"])
    num_chars = int(tts.hparams["num_chars"])
    speaker_count = int(tts.hparams.get("speaker_count", 1))

    max_id = max(char_to_id.values()) if char_to_id else 0
    if max_id >= num_chars:
        raise ValueError(
            f"Text encoder produced max id {max_id} but model.num_chars={num_chars}. "
            "Adjust --allowed_chars / transcripts so the dynamic vocab fits the checkpoint, "
            "or use training data that matches Tacotron's character table."
        )

    scores: list[dict] = []
    bs = max(1, int(args.batch_size))
    ref_sr = int(args.reference_sample_rate)
    sr_vocoder = HIFI_GAN_SR

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

            # Reference mels use --reference_sample_rate (match RL datamodule); same as training_step
            # ``wav_lens = mel_len * 256`` when mels are already HiFi-GAN–compatible.
            wav_ref = model.vocoder(ref_mel)
            seq_len = wav_ref.shape[1]
            wav_lens = ref_mel_len * 256
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

            batch_lines = texts_norm[start:end]
            x, x_len = _encode_batch(char_to_id, batch_lines, end_str, device=device)

            _, mel_post, gate, _ = model.tts(
                chars_idx=x,
                chars_idx_len=x_len,
                teacher_forcing_dropout=0.0,
                teacher_forcing=False,
                speaker_id=speaker,
                max_len_override=int(args.max_len),
                style=style,
            )

            # Tacotron mel frames are at tacotron_sample_rate; stretch to HiFi-GAN rate like
            # ``generate_and_score_tactron.py``.
            if scale_tacotron_hifi_gan != 1.0:
                mel_bn_t = F.interpolate(
                    mel_post.transpose(1, 2),
                    scale_factor=scale_tacotron_hifi_gan,
                    mode="linear",
                    align_corners=False,
                )
                mel_for_vocoder = mel_bn_t.transpose(1, 2)
            else:
                mel_for_vocoder = mel_post

            wav = model.vocoder(mel_for_vocoder)
            if wav.dim() == 3:
                wav = wav.squeeze(1)

            gate_end = gate.squeeze(-1) < 0
            has_end = gate_end.any(dim=1)
            end_idx = gate_end.int().argmax(dim=1)
            mel_frames = torch.where(
                has_end, end_idx + 1, torch.full_like(end_idx, gate.shape[1])
            )
            pred_len = (
                mel_frames.float() * scale_tacotron_hifi_gan
            ).round().long() * 256
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
                    # gst_weights: (B*heads, tokens) → 8 heads × 10 token weights per utterance (nested list)
                    H = model.rl_policy.gst_heads
                    Tn = model.rl_policy.gst_token_num
                    gst_w_per_head = gst_weights.view(bsz, H, Tn)[bi].detach().cpu().tolist()
                    scores.append(
                        {
                            "wav_path_orig": str(out_path),
                            "wav_path_16k": str(out_path_16k),
                            "generated_trustworthiness_score": float(score),
                            "gst_weights": gst_w_per_head,
                        }
                    )

    if scores:
        scores_path = out_dir / "scores.json"
        scores_path.write_text(json.dumps(scores, indent=2))
        print(f"Saved scores to {scores_path}")

    print(f"Saved {len(transcripts)} wavs to {out_dir}")


if __name__ == "__main__":
    main()

