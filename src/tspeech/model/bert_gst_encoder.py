"""BERT-based text encoder for generating GST weights."""
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import BertModel, AutoTokenizer


class BERTGSTEncoder(nn.Module):
    """
    BERT text encoder that outputs 10-dimensional GST weights via softmax.
    
    Flow:
        Text → BERT → Pooled features → Linear → 10-dim logits → Softmax → GST weights
    """
    
    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        gst_token_num: int = 10,
        freeze_bert: bool = True,
    ):
        """
        Initialize BERT GST encoder.
        
        Parameters
        ----------
        bert_model_name : str
            HuggingFace BERT model name
        gst_token_num : int
            Number of GST tokens (default: 10)
        freeze_bert : bool
            Whether to freeze BERT parameters (default: True)
        """
        super().__init__()
        
        # Load BERT model and tokenizer
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        
        # Freeze BERT if requested
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Get BERT hidden size
        bert_hidden_size = self.bert.config.hidden_size
        
        # Linear layer to map BERT output to GST weight logits
        # Output: 10-dim logits (one per GST token)
        self.gst_weight_projection = nn.Linear(bert_hidden_size, gst_token_num)
        
        self.gst_token_num = gst_token_num
    
    def get_bert_embeddings(self, text: list[str]) -> Tensor:
        """
        Get BERT embeddings for text (used by RL policy).
        
        Parameters
        ----------
        text : list[str]
            List of text strings (batch)
            
        Returns
        -------
        Tensor
            BERT embeddings of shape (batch_size, bert_hidden_size)
        """
        # Tokenize text
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        # Move to same device as BERT model
        device = next(self.bert.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        # Get BERT embeddings
        with torch.no_grad() if not any(p.requires_grad for p in self.bert.parameters()) else torch.enable_grad():
            bert_outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        # Mean pooling over sequence (masked)
        pooled_output = bert_outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        masked_output = pooled_output * mask_expanded
        pooled = masked_output.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)  # (batch, hidden_size)
        
        return pooled
    
    def forward(self, text: list[str]) -> Tensor:
        """
        Encode text and generate GST weights.
        
        Parameters
        ----------
        text : list[str]
            List of text strings (batch)
            
        Returns
        -------
        Tensor
            GST weights of shape (batch_size, gst_token_num)
            Each row sums to 1.0 (softmax probabilities)
        """
        # Get BERT embeddings
        pooled = self.get_bert_embeddings(text)
        
        # Project to GST weight logits
        gst_logits = self.gst_weight_projection(pooled)  # (batch, gst_token_num)
        
        # Apply softmax to get weights (probabilities)
        gst_weights = F.softmax(gst_logits, dim=1)  # (batch, gst_token_num)
        
        return gst_weights
    
    def encode_text(self, text: str) -> Tensor:
        """
        Encode a single text string.
        
        Parameters
        ----------
        text : str
            Input text string
            
        Returns
        -------
        Tensor
            GST weights of shape (gst_token_num,)
        """
        weights = self.forward([text])
        return weights.squeeze(0)  # Remove batch dimension




