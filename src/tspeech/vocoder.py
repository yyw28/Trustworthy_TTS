"""HiFi-GAN vocoder for converting mel spectrograms to waveforms."""
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResBlock(nn.Module):
    """Residual block for HiFi-GAN."""
    
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(
                channels, channels, kernel_size, 1,
                dilation=dilation[i], padding=(kernel_size - 1) * dilation[i] // 2
            )
            for i in range(len(dilation))
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(
                channels, channels, kernel_size, 1,
                dilation=1, padding=(kernel_size - 1) // 2
            )
            for _ in range(len(dilation))
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class Generator(nn.Module):
    """HiFi-GAN generator."""
    
    def __init__(
        self,
        initial_channel: int,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        num_mels: int = 80,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        
        # Initial convolution
        self.conv_pre = nn.Conv1d(
            num_mels, upsample_initial_channel, 7, 1, padding=3
        )
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2 ** i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k, u, padding=(k - u) // 2
                )
            )
            self.noise_convs.append(
                nn.Conv1d(
                    upsample_initial_channel // (2 ** (i + 1)),
                    upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=1
                )
            )
        
        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))
        
        # Final convolution
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3)
        
    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = self.noise_convs[i](torch.randn_like(x))
            x = x + xs
            for j in range(self.num_kernels):
                x = self.resblocks[i * self.num_kernels + j](x)
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x


class HiFiGANVocoder:
    """HiFi-GAN vocoder wrapper."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        config_path: Optional[str] = None,
        device: str = "cpu",
        sample_rate: int = 22050,
    ):
        """
        Initialize HiFi-GAN vocoder.
        
        Parameters
        ----------
        checkpoint_dir : str
            Directory containing generator checkpoint (g_*.pth or g_* file)
        config_path : Optional[str]
            Path to config.json (if None, looks for config.json in checkpoint_dir)
        device : str
            Device to run on ("cpu", "cuda", "mps")
        sample_rate : int
            Sample rate of output audio (should match config)
        """
        self.device = device
        self.sample_rate = sample_rate
        
        checkpoint_dir = Path(checkpoint_dir)
        
        # Resolve relative paths relative to project root or current working directory
        if not checkpoint_dir.is_absolute():
            # Try relative to current file's parent (src/tspeech/)
            possible_path = Path(__file__).parent.parent.parent / checkpoint_dir
            if possible_path.exists():
                checkpoint_dir = possible_path
            # Otherwise use as-is (relative to cwd)
        
        # Load config
        if config_path is None:
            config_path = checkpoint_dir / "config.json"
        else:
            config_path = Path(config_path)
            if not config_path.is_absolute():
                config_path = checkpoint_dir / config_path
            
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Find generator checkpoint
        generator_files = list(checkpoint_dir.glob("g_*"))
        if not generator_files:
            raise FileNotFoundError(
                f"Generator checkpoint not found in {checkpoint_dir}. "
                f"Expected file matching pattern 'g_*'"
            )
        generator_path = generator_files[0]
        
        # Initialize generator
        self.generator = Generator(
            initial_channel=config.get("upsample_initial_channel", 512),  # Parameter name kept for API compatibility
            resblock_kernel_sizes=config["resblock_kernel_sizes"],
            resblock_dilation_sizes=config["resblock_dilation_sizes"],
            upsample_rates=config["upsample_rates"],
            upsample_initial_channel=config["upsample_initial_channel"],
            upsample_kernel_sizes=config["upsample_kernel_sizes"],
            num_mels=config["num_mels"],
        )
        
        # Load checkpoint
        try:
            checkpoint = torch.load(generator_path, map_location=device)
        except Exception as e:
            # Try loading as a state dict directly
            try:
                checkpoint = torch.load(generator_path, map_location=device, weights_only=False)
            except:
                raise RuntimeError(f"Failed to load checkpoint from {generator_path}: {e}")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "generator" in checkpoint:
                state_dict = checkpoint["generator"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                # Assume it's already a state dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove "module." prefix if present (from DataParallel)
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }
        
        self.generator.load_state_dict(state_dict, strict=False)
        self.generator.to(device)
        self.generator.eval()
        
    def __call__(self, mel_spectrogram: Tensor, sample_rate: Optional[int] = None) -> Tensor:
        """
        Convert mel spectrogram to waveform.
        
        Parameters
        ----------
        mel_spectrogram : Tensor
            Mel spectrogram of shape (batch, num_mels, time) or (num_mels, time)
        sample_rate : Optional[int]
            Sample rate (ignored, kept for compatibility)
            
        Returns
        -------
        Tensor
            Waveform of shape (batch, samples) or (samples,)
        """
        # Handle single spectrogram
        if mel_spectrogram.dim() == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        mel_spectrogram = mel_spectrogram.to(self.device)
        
        with torch.no_grad():
            waveform = self.generator(mel_spectrogram)
            # Remove channel dimension: (batch, 1, samples) -> (batch, samples)
            waveform = waveform.squeeze(1)
        
        if squeeze_output:
            waveform = waveform.squeeze(0)
        
        return waveform
