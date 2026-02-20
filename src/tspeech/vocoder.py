"""HiFi-GAN vocoder wrapper for TTS generation."""
import json
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn

from tspeech.model.tacotron2.hifi_gan import Generator


class AttrDict(dict):
    """Dictionary that allows attribute access."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class HiFiGANVocoder(nn.Module):
    """
    HiFi-GAN vocoder wrapper for converting mel spectrograms to waveforms.
    
    Usage:
        vocoder = HiFiGANVocoder(checkpoint_dir="UNIVERSAL_V1")
        waveform = vocoder(mel_spectrogram)  # (batch, mel_frames, n_mels) -> (batch, samples) at 22050 Hz
    """

    def __init__(self, checkpoint_dir: str):
        """
        Initialize HiFi-GAN vocoder.
        
        Parameters
        ----------
        checkpoint_dir : str
            Directory containing config.json and generator checkpoint (e.g., g_02500000)
        """
        super().__init__()

        self.checkpoint_dir = self._resolve_checkpoint_dir(checkpoint_dir)

        # Load config
        config_path = self.checkpoint_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path} (checkpoint_dir={self.checkpoint_dir})"
            )
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        self.config = AttrDict(config_dict)
        
        # Find generator checkpoint
        generator_files = list(self.checkpoint_dir.glob("g_*"))
        if not generator_files:
            raise FileNotFoundError(f"No generator checkpoint found in {self.checkpoint_dir}")
        
        # Use the first generator checkpoint found (or could sort by name)
        generator_path = sorted(generator_files)[0]
        
        # Initialize generator
        self.generator = Generator(self.config)
        
        # Load checkpoint
        checkpoint = torch.load(generator_path)
        if isinstance(checkpoint, dict) and "generator" in checkpoint:
            state_dict = checkpoint["generator"]
        else:
            state_dict = checkpoint
        
        self.generator.load_state_dict(state_dict, strict=False)
        
        # Remove weight norm for inference
        self.generator.remove_weight_norm()
        self.generator.eval()

        print(f"✓ HiFi-GAN vocoder loaded from {generator_path}")

    @staticmethod
    def _resolve_checkpoint_dir(checkpoint_dir: str) -> Path:
        raw_path = Path(checkpoint_dir)
        candidates = []

        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(raw_path)
            candidates.append(Path.cwd() / raw_path)
            project_root = Path(__file__).resolve().parents[2]
            candidates.append(project_root / raw_path)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return raw_path
    
    def __call__(self, mel_spectrogram: Tensor) -> Tensor:
        """
        Convert mel spectrogram to waveform.
        
        Parameters
        ----------
        mel_spectrogram : Tensor
            Mel spectrogram of shape (batch, mel_frames, n_mels)
            Values should be in log scale
            
        Returns
        -------
        Tensor
            Waveform of shape (batch, samples) at 22050 Hz
        """
        # Convert from (batch, mel_frames, n_mels) to (batch, n_mels, mel_frames) for HiFi-GAN
        mel_spectrogram = mel_spectrogram.transpose(1, 2)
        
        # Ensure mel is in log scale (clamp to avoid numerical issues)
        mel_spectrogram = torch.clamp(mel_spectrogram, min=-11.5, max=2.0)
        
        # Generate waveform
        with torch.no_grad():
            waveform = self.generator(mel_spectrogram)
        
        # Squeeze channel dimension: (batch, 1, samples) -> (batch, samples)
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        
        return waveform
