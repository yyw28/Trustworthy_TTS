import re
from os import path
from typing import NamedTuple, Optional

import librosa
import torch
from sklearn.preprocessing import OrdinalEncoder
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram


class TTSBatch(NamedTuple):
    speaker_id: Tensor

    chars_idx: Tensor
    chars_idx_len: Tensor

    mel_spectrogram: Tensor
    mel_spectrogram_len: Tensor

    gate: Tensor
    gate_len: Tensor

    filename: list[str]
    text: list[str]


_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def _expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


class TTSDataset(Dataset):
    def __init__(
        self,
        filenames: list[str],
        texts: list[str],
        base_dir: str,
        allowed_chars: str,
        num_mels: int,
        sample_rate: int,
        speaker_ids: Optional[list[str]] = None,
        end_token: Optional[str] = "^",
        silence: int = 0,
        trim: bool = True,
        trim_top_db: int = 60,
        trim_frame_length: int = 2048,
        expand_abbreviations=False,
    ):
        super().__init__()

        if end_token is not None and end_token in allowed_chars:
            raise Exception("end_token cannot be in allowed_chars!")

        self.sample_rate = sample_rate

        # Simple assignments
        self.filenames = filenames
        self.end_token = end_token
        if end_token is None:
            print("Dataset: Not using an end token")
        else:
            print(f"Dataset: Using end token {end_token}")

        self.trim = trim
        self.trim_top_db = trim_top_db
        self.trim_frame_length = trim_frame_length
        if trim:
            print(
                f"Dataset: Trimming silence with top db {trim_top_db} and frame length {trim_frame_length}"
            )
        else:
            print("Dataset: Not trimming silence from input audio files")

        print(f"Dataset: Adding {silence} frames of silence to the end of each clip")

        self.silence = silence

        self.base_dir = base_dir

        print(f"Dataset: Allowed characters {allowed_chars}")

        # Preprocessing step - ensure textual data only contains allowed characters
        # allowed_chars_re = re.compile(f"[^{allowed_chars}]+")
        # texts = [
        #     allowed_chars_re.sub("", unidecode.unidecode(t).lower()) for t in texts
        # ]

        if expand_abbreviations:
            print("Dataset: Expanding abbreviations in input text...")
            texts = [_expand_abbreviations(t) for t in texts]
        if end_token is not None:
            texts = [t + end_token for t in texts]

        chars_set = set()
        for t in texts:
            chars_set = chars_set.union(set(t))
        chars = list(chars_set)
        chars.sort()

        self.texts = texts

        self.speaker_ids = speaker_ids

        # Preprocessing step - create an ordinal encoder to transform textual data to a
        # tensor of integers
        self.encoder = OrdinalEncoder()
        if end_token is None:
            self.encoder.fit([[x] for x in chars])
        else:
            self.encoder.fit([[x] for x in chars + [end_token]])

        # Create a Torchaudio MelSpectrogram generator
        self.melspectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            f_min=0.0,
            f_max=8000.0,
            n_mels=num_mels,
            power=1.0,
            mel_scale="slaney",
            norm="slaney",
            center=True,
        )

    def _load_audio(self, file_path: str) -> Tensor:
        # Prefer torchcodec when available (fast). If it fails to load (e.g. FFmpeg or
        # version mismatch), fall back to librosa, which does not depend on torchcodec.
        try:
            from torchcodec.decoders import AudioDecoder  # type: ignore

            with open(file_path, "rb") as infile:
                decoder = AudioDecoder(infile.read(), sample_rate=self.sample_rate)
                wav = decoder.get_all_samples().data

            # Ensure mono 1D float tensor
            if wav.ndim == 2 and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.squeeze(0).to(dtype=torch.float32)
            return wav
        except Exception:
            pass

        # Fallback: decode with librosa (no torchcodec usage)
        wav_np, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        wav = torch.tensor(wav_np, dtype=torch.float32)
        return wav

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i: int) -> TTSBatch:
        # Audio preprocessing -----------------------------------------------------------
        # Load the audio file and squeeze it to a 1D Tensor
        filename = self.filenames[i]
        wav = self._load_audio(path.join(self.base_dir, filename))

        # Trim if necessary
        if self.trim:
            wav_np, _ = librosa.effects.trim(
                wav.numpy(),
                top_db=self.trim_top_db,
                frame_length=self.trim_frame_length,
            )
            wav = torch.tensor(wav_np)
        wav = F.pad(wav, (0, self.silence))

        # Create the Mel spectrogram and save its length
        mel_spectrogram = self.melspectrogram(wav).swapaxes(0, 1)
        mel_spectrogram = torch.log(torch.clamp(mel_spectrogram, min=1e-5))
        mel_spectrogram_len = torch.IntTensor([len(mel_spectrogram)])

        # Create gate output indicating whether the TTS model should continue producing Mel
        # spectrogram frames
        gate = torch.ones(len(mel_spectrogram), 1)
        gate[-1] = 0.0
        gate_len = torch.IntTensor([len(gate)])

        # Text preprocessing ------------------------------------------------------------
        text = self.texts[i]

        # Encode the text
        chars_idx = self.encoder.transform([[x] for x in text])

        # Index 0 is for padding, so increment all characters by 1
        chars_idx += 1

        # Transform to a tensor and remove the extra dimension necessary for the OrdinalEncoder
        chars_idx = torch.tensor(chars_idx, dtype=torch.int64).squeeze(-1)
        chars_idx_len = torch.tensor([len(chars_idx)], dtype=torch.int64)

        return TTSBatch(
            speaker_id=(
                torch.tensor(
                    [self.speaker_ids[i]] if self.speaker_ids is not None else [],
                    dtype=torch.int64,
                )
            ),
            chars_idx=chars_idx.unsqueeze(0),
            chars_idx_len=chars_idx_len,
            mel_spectrogram=mel_spectrogram.unsqueeze(0),
            mel_spectrogram_len=mel_spectrogram_len,
            gate=gate.unsqueeze(0),
            gate_len=gate_len,
            text=[text],
            filename=[filename],
        )
