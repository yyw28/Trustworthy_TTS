"""HubERT-based trustworthiness classifier for TTS output evaluation."""
from typing import Optional

import torch
from torch import Tensor

from tspeech.hubert.model.htmodel import HTModel


class TrustClassifier:
    """
    Wrapper for HubERT trustworthiness classifier.
    
    Evaluates the trustworthiness of generated speech waveforms.
    """
    
    def __init__(
        self,
        hubert_model_name: str = "facebook/hubert-base-ls960",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize trustworthiness classifier.
        
        Parameters
        ----------
        hubert_model_name : str
            HuggingFace HubERT model name
        checkpoint_path : Optional[str]
            Path to trained HubERT checkpoint (if None, uses pretrained HubERT only)
        device : Optional[str]
            Device to run on (cuda, mps, cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        if checkpoint_path:
            # Load trained classifier
            self.model = HTModel.load_from_checkpoint(
                checkpoint_path,
                hubert_model_name=hubert_model_name,
                trainable_layers=0,  # Use frozen for inference
            )
        else:
            # Use pretrained HubERT with untrained classifier head
            self.model = HTModel(
                hubert_model_name=hubert_model_name,
                trainable_layers=0,
            )
        
        # Freeze all parameters for inference
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model = self.model.to(self.device)
    
    def evaluate(self, waveform: Tensor, sample_rate: int = 16000) -> Tensor:
        """
        Evaluate trustworthiness of waveform.
        
        Parameters
        ----------
        waveform : Tensor
            Audio waveform of shape (batch, samples) or (samples,)
            Should be mono, 16 kHz
        sample_rate : int
            Sample rate of input waveform (will be resampled to 16 kHz if needed)
            
        Returns
        -------
        Tensor
            Trustworthiness scores of shape (batch, 1) or (1,)
            Values are logits (use sigmoid to get probabilities)
        """
        # Ensure waveform is on correct device
        waveform = waveform.to(self.device)
        
        # Handle single waveform
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Resample to 16 kHz if needed
        if sample_rate != 16000:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000,
            ).to(self.device)
            waveform = resampler(waveform)
        
        # Create attention mask (all ones for full sequences)
        batch_size, seq_len = waveform.shape
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=self.device)
        
        # Evaluate
        with torch.no_grad():
            trustworthiness_logits = self.model(wav=waveform, mask=attention_mask)
        
        return trustworthiness_logits
    
    def evaluate_batch(self, waveforms: list[Tensor], sample_rate: int = 16000) -> Tensor:
        """
        Evaluate trustworthiness of multiple waveforms.
        
        Parameters
        ----------
        waveforms : list[Tensor]
            List of audio waveforms, each of shape (samples,)
        sample_rate : int
            Sample rate of input waveforms
            
        Returns
        -------
        Tensor
            Trustworthiness scores of shape (len(waveforms), 1)
        """
        # Pad waveforms to same length
        max_len = max(w.shape[0] for w in waveforms)
        padded = []
        for w in waveforms:
            if w.dim() == 1:
                w = w.unsqueeze(0)
            pad_len = max_len - w.shape[1]
            if pad_len > 0:
                w = torch.nn.functional.pad(w, (0, pad_len))
            padded.append(w)
        
        batch_waveform = torch.cat(padded, dim=0)
        return self.evaluate(batch_waveform, sample_rate=sample_rate)








