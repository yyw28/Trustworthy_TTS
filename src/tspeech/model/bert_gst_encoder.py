"""BERT-based text encoder for GST weights. Text → BERT → linear → softmax → (batch, gst_token_num)."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import BertModel, AutoTokenizer


class BERTEncoder(nn.Module):
    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        freeze_bert: bool = True,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.tw = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + 1, self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.Tanh(),
        )

        # self.tw = nn.Sequential(nn.Linear(self.bert.))
        for p in self.bert.parameters():
            p.requires_grad = False

        if not freeze_bert:
            for i in range(-3, 0):
                for p in self.bert.encoder.layer[i].parameters():
                    p.requires_grad = True

    def forward(self, score: Tensor, text: list[str]) -> Tensor:
        tokenized = self.tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].to(self.bert.device)
        attention_mask = tokenized["attention_mask"].to(self.bert.device)

        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Masked mean pooling
        h = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        return self.tw(torch.concat([pooled, score[:, None]], dim=-1))
