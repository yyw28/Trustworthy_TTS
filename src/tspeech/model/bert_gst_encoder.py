"""BERT-based text encoder for GST weights. Text → BERT → linear → softmax → (batch, gst_token_num)."""
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import BertModel, AutoTokenizer


class BERTGSTEncoder(nn.Module):
    """BERT encoder + linear projection to GST weight logits. Outputs softmax weights (batch, gst_token_num)."""

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        gst_token_num: int = 10,
        freeze_bert: bool = True,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.gst_weight_projection = nn.Linear(self.bert.config.hidden_size, gst_token_num)
        self.gst_token_num = gst_token_num

    def get_bert_embeddings(self, text: list[str]) -> Tensor:
        """(batch,) text → (batch, hidden_size) BERT pooled embeddings."""
        enc = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        device = next(self.bert.parameters()).device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Masked mean pooling
        h = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return pooled

    def forward(self, text: list[str]) -> Tensor:
        """(batch,) text → (batch, gst_token_num) GST weights (softmax, sum=1)."""
        pooled = self.get_bert_embeddings(text)
        return F.softmax(self.gst_weight_projection(pooled), dim=1)

    def encode_text(self, text: str) -> Tensor:
        """Single text → (gst_token_num,) GST weights."""
        return self.forward([text]).squeeze(0)
