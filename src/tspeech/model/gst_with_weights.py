"""GST module that accepts pre-computed weights (e.g., from BERT)."""
import torch
from torch import Tensor, nn


class GSTWithWeights(nn.Module):
    """
    GST module that uses pre-computed weights (e.g., from BERT text encoder).
    
    Instead of computing weights from mel spectrograms via attention,
    this module directly uses provided weights to compute weighted sum of GST tokens.
    
    Flow:
        GST weights (batch, 10) × GST token embeddings (10, token_dim) 
        → Weighted sum → Style embedding (batch, token_dim)
    """
    
    def __init__(self, token_num: int = 10, token_embedding_size: int = 512):
        """
        Initialize GST with weights.
        
        Parameters
        ----------
        token_num : int
            Number of GST tokens (default: 10)
        token_embedding_size : int
            Dimension of GST token embeddings (should match encoded_dim)
        """
        super().__init__()
        
        # GST token embeddings (learnable parameters)
        # Shape: (token_num, token_embedding_size)
        self.gst_tokens = nn.Parameter(
            torch.FloatTensor(token_num, token_embedding_size)
        )
        
        nn.init.normal_(self.gst_tokens, mean=0, std=0.5)
        self.token_num = token_num
        self.token_embedding_size = token_embedding_size
    
    def forward(self, weights: Tensor) -> Tensor:
        """
        Compute style embedding from GST weights.
        
        Parameters
        ----------
        weights : Tensor
            GST weights of shape (batch_size, token_num)
            Each row should sum to 1.0 (softmax probabilities)
            
        Returns
        -------
        Tensor
            Style embedding of shape (batch_size, token_embedding_size)
            Computed as weighted sum: weights @ gst_tokens
        """
        # weights: (batch, token_num)
        # gst_tokens: (token_num, token_embedding_size)
        # Output: (batch, token_embedding_size)
        style_embed = torch.matmul(weights, self.gst_tokens)
        
        return style_embed







