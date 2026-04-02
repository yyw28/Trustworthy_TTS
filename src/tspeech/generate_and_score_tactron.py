#!/usr/bin/env python3
"""
Generate audio with Tacotron2+GST and HiFi-GAN (style from a reference wav).

- Batched inference (--batch_size) for faster GPU generation
- Optional HuBERT trustworthiness scoring (--hubert_checkpoint); writes scores.json
- With --libritts_csv, output filenames mirror wav paths with '/' -> '_'

Modeled after ``say.ipynb``.
"""

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

import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as ta_f
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from tspeech.model.tacotron2.hifi_gan import Generator
from tspeech.model.tts import TTSModel

HIFI_GAN_SR = 22050
HUBERT_SR = 16000

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class LibriTTSRow(TypedDict, total=False):
    wav: str
    speaker_id: str
    text: str
    duration: str
    speaker_idx: str


def _normalize_text(text: str, allowed_chars: str) -> str:
    allowed = set(allowed_chars)
    return "".join(ch for ch in text.lower() if ch in allowed)


def _build_char_to_id_from_texts(texts: list[str], end_token: str) -> dict[str, int]:
    chars_set: set[str] = set()
    for t in texts:
        chars_set.update(t)
    chars_set.add(end_token)
    chars = sorted(chars_set)
    return {ch: i + 1 for i, ch in enumerate(chars)}  # 0 reserved for padding


def _encode_batch(
    char_to_id: dict[str, int],
    texts: list[str],
    end_token: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    seqs = [torch.tensor([char_to_id[ch] for ch in (t + end_token)], dtype=torch.int64) for t in texts]
    lens = torch.tensor([s.numel() for s in seqs], dtype=torch.int64, device=device)
    x = pad_sequence(seqs, batch_first=True, padding_value=0).to(device=device)
    return x, lens


def _mel_spectrogram(wav: torch.Tensor, sr: int, n_mels: int) -> torch.Tensor:
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
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
    )(wav)
    return torch.log(torch.clamp(melspec, min=1e-5)).transpose(0, 1)


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
        return p
    if base_dir is None:
        raise ValueError("Relative wav paths require --libritts_dir")
    return str(base_dir / p)


def _load_hifigan_generator(checkpoint_dir: str, device: torch.device) -> Generator:
    ckpt_dir = Path(checkpoint_dir)
    config_path = ckpt_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing HiFi-GAN config: {config_path}")

    config = AttrDict(json.loads(config_path.read_text()))
    generator_files = sorted(ckpt_dir.glob("g_*"))
    if not generator_files:
        raise FileNotFoundError(f"No HiFi-GAN generator checkpoint (g_*) found in {ckpt_dir}")
    generator_path = generator_files[0]

    generator = Generator(config)
    state = torch.load(generator_path, map_location="cpu")
    state_dict = state["generator"] if isinstance(state, dict) and "generator" in state else state
    generator.load_state_dict(state_dict, strict=False)
    generator.remove_weight_norm()
    return generator.to(device).eval()


def main() -> None:
    p = argparse.ArgumentParser(description="Generate test samples with Tacotron2+GST checkpoint")
    p.add_argument("--checkpoint", required=True, help="Tacotron+GST checkpoint, e.g. checkpoint-epoch=85-....ckpt")
    p.add_argument("--config", required=True, help="YAML file containing the config used to train the checkpoint")
    p.add_argument("--vocoder_checkpoint_dir", required=True, help="HiFi-GAN checkpoint dir (UNIVERSAL_V1)")
    p.add_argument("--hubert_checkpoint", default=None, help="Optional HuBERT trustworthiness checkpoint to score generated audio")

    p.add_argument("--libritts_dir", default="/workplace/LibriTTS", help="Base directory for LibriTTS wav paths")
    p.add_argument("--libritts_csv", default=None, help="Optional LibriTTS CSV: wav|speaker_id|text|duration|speaker_idx")

    p.add_argument("--style_wav", default=None, help="Reference wav to compute GST style from (absolute or relative to --libritts_dir)")
    p.add_argument("--text", action="append", default=None, help="Text to synthesize (repeatable)")
    p.add_argument("--texts_file", default=None, help="Text file (one line per text)")
    p.add_argument("--output_dir", default="./tacotron_gst_test_audio")
    p.add_argument("--limit", type=int, default=None, help="If set, only synthesize the first N items")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size for faster generation on GPU")

    p.add_argument("--speaker_idx", type=int, default=0)
    p.add_argument("--end_token", default="^")
    p.add_argument("--allowed_chars", default="!'(),.:;? \\-abcdefghijklmnopqrstuvwxyz")
    p.add_argument("--max_len", type=int, default=800)

    p.add_argument("--device", default="auto", help="auto/cuda/cpu")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model config file
    with open(args.config) as infile:
        config = yaml.safe_load(infile)
    tacotron_sample_rate = config["data"]["sample_rate"]

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

    if not texts:
        raise ValueError("Provide --text/--texts_file or --libritts_csv")

    if args.limit is not None:
        n_lim = int(args.limit)
        if n_lim > 0:
            texts = texts[:n_lim]
            if rows:
                rows = rows[:n_lim]

    texts = [_normalize_text(t, args.allowed_chars) for t in texts]
    char_to_id = _build_char_to_id_from_texts(texts, args.end_token)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    model = TTSModel.load_from_checkpoint(args.checkpoint, map_location="cpu").to(device).eval()
    if model.gst is None:
        raise ValueError("Loaded checkpoint has gst disabled (model.gst is None).")

    tacotron = model.tacotron2
    gst = model.gst
    n_mels = int(model.hparams["num_mels"])
    num_chars = int(model.hparams["num_chars"])
    speaker_count = int(model.hparams.get("speaker_count", 1))

    max_id = max(char_to_id.values())
    if max_id >= num_chars:
        raise ValueError(
            f"Text encoder produced id {max_id} but model.num_chars={num_chars}. "
            "This usually means the char vocabulary used at inference doesn't match training. "
            "Try adjusting --allowed_chars / normalization, or build the vocab from the same dataset used in training."
        )

    generator = _load_hifigan_generator(args.vocoder_checkpoint_dir, device=device)

    hubert = None
    if args.hubert_checkpoint:
        from tspeech.model.htmodel import HTModel

        hubert = HTModel.load_from_checkpoint(
            args.hubert_checkpoint,
            map_location="cpu",
            hubert_model_name="facebook/hubert-base-ls960",
            trainable_layers=0,
        ).to(device)
        hubert.eval()
        for p_ in hubert.parameters():
            p_.requires_grad = False

    scores: list[dict] = []

    style_wav_path: Optional[str] = args.style_wav
    if style_wav_path is None:
        if not rows:
            raise ValueError("--style_wav is required unless --libritts_csv is provided")
        style_wav_path = rows[0].get("wav")
        if not style_wav_path:
            raise ValueError("Could not pick a style wav from --libritts_csv; pass --style_wav explicitly")

    # Compute the reference audio style embedding
    audio_np, sr = sf.read(
        _resolve_wav_path(str(style_wav_path), base_dir),
        dtype="float32",
        always_2d=True,
    )
    style_wav = torch.from_numpy(audio_np).transpose(0, 1)
    # If necessary, convert stereo to mono by averaging the two channels
    if style_wav.shape[0] > 1:
        style_wav = style_wav.mean(dim=0, keepdim=True)
    # If necessary, resample to the Tacotron sample rate
    if sr != tacotron_sample_rate:
        style_wav = torchaudio.functional.resample(
            style_wav, orig_freq=sr, new_freq=tacotron_sample_rate
        )
    style_mel = (
        _mel_spectrogram(style_wav.squeeze(0), sr=tacotron_sample_rate, n_mels=n_mels)
        .unsqueeze(0)
        .to(device)
    )
    style_mel_len = torch.tensor([style_mel.shape[1]], dtype=torch.int64, device=device)
    # Compute the style embedding
    style = gst(style_mel, style_mel_len)

    def _out_name(i: int) -> str:
        if i < len(rows) and rows[i].get("wav"):
            ref_wav_str = str(rows[i]["wav"])
            name = ref_wav_str.replace("\\", "_").replace("/", "_")
            return name if name.lower().endswith(".wav") else (name + ".wav")
        return f"sample_{i:03d}.wav"

    bs = max(1, int(args.batch_size))

    # The following scale ratio exists because of a mismatch between different components
    # of Tacotron and the sample rates they expect to receive as input.
    #
    # Currently, our trained Tacotron instance outputs spectrograms with a sample rate of 16000 Hz,
    # but this could change in the future. HiFi-GAN expects and outputs a sample rate of 22050 Hz.
    scale_tacotron_hifi_gan = (
        1 if tacotron_sample_rate == HIFI_GAN_SR else HIFI_GAN_SR / tacotron_sample_rate
    )

    with torch.inference_mode():
        for start in range(0, len(texts), bs):
            end = min(start + bs, len(texts))
            batch_texts = texts[start:end]

            speaker_idxs: list[int] = []
            for j in range(start, end):
                sp = int(args.speaker_idx)
                if j < len(rows) and rows[j].get("speaker_idx"):
                    sp = int(rows[j]["speaker_idx"])
                if sp < 0 or sp >= speaker_count:
                    sp = 0
                speaker_idxs.append(sp)
            speaker = torch.tensor(speaker_idxs, dtype=torch.int64, device=device)

            x, x_len = _encode_batch(char_to_id, batch_texts, args.end_token, device=device)

            _, mel_post, gate, _ = tacotron(
                chars_idx=x,
                chars_idx_len=x_len,
                teacher_forcing_dropout=0.0,
                teacher_forcing=False,
                speaker_id=speaker,
                max_len_override=int(args.max_len),
                encoded_extra=style.expand((end - start, -1, -1)),
            )

            mel_b80t = F.interpolate(
                mel_post.transpose(1, 2),
                scale_factor=scale_tacotron_hifi_gan,
                mode="linear",
                align_corners=False,
            )

            wav = generator(mel_b80t)
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
                sf.write(str(out_path), wav_0, HIFI_GAN_SR)

                if hubert is not None:
                    # Same trim as saved wav, then resample to 16 kHz for HuBERT (no separate mel undo).
                    seg = wav[bi : bi + 1, :n]
                    wav16_seg = ta_f.resample(
                        seg,
                        orig_freq=HIFI_GAN_SR,
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

                    mask = torch.ones((1, wav16_0.shape[0]), device=device, dtype=torch.long)
                    score = torch.sigmoid(hubert(wav=wav16_0.unsqueeze(0), mask=mask)).squeeze(-1)[0].item()
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

    print(f"Saved {len(texts)} wavs to {out_dir}")


if __name__ == "__main__":
    main()
