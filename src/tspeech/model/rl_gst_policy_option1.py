"""
Reinforcement Learning Policy Network for *soft* GST weights optimization (continuous action).
- You DO sample a continuous 10-dim vector of GST weights (on the simplex).
- Correct REINFORCE: log_prob corresponds to the sampled latent z that produced the weights.

We use a Logistic-Normal policy:
  z ~ Normal(mu(x), sigma(x))
  gst_weights = softmax(z / temperature)
"""

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions import Normal


class RLGSTPolicy(nn.Module):
    """BERT embeddings → Normal(mu, sigma) → z → softmax(z/temp) = GST weights. Returns weights, log_prob(z), entropy."""

    def __init__(
        self,
        bert_hidden_size: int = 768,
        gst_token_num: int = 10,
        gst_heads: int = 8,
        hidden_dim: int = 256,
        temperature: float = 1.0,
        init_log_std: float = -0.5,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.gst_token_num = gst_token_num
        self.gst_heads = gst_heads
        self.temperature = temperature
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(bert_hidden_size, hidden_dim * gst_heads),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * gst_heads, hidden_dim * gst_heads),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.mu_head = nn.Linear(hidden_dim * gst_heads, gst_token_num * gst_heads)
        self.log_std_head = nn.Linear(hidden_dim * gst_heads, gst_token_num * gst_heads)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.constant_(self.log_std_head.bias, init_log_std)

    def forward(
        self,
        bert_embeddings: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """(batch, 768) → (batch, K) weights, (batch,) log_prob or None, (batch,) entropy or None."""
        h = self.trunk(bert_embeddings)
        mu = self.mu_head(h).reshape(-1, self.gst_heads, self.gst_token_num)
        log_std = (
            self.log_std_head(h)
            .clamp(self.log_std_min, self.log_std_max)
            .reshape(-1, self.gst_heads, self.gst_token_num)
        )
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        z = dist.rsample()
        log_probs = dist.log_prob(z).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)

        gst_weights = F.softmax(z / self.temperature, dim=1)
        return gst_weights, log_probs, entropy

    # def get_log_probs(self, bert_embeddings: Tensor, z: Tensor) -> Tensor:
    #     """Log prob of z under current policy (for REINFORCE)."""
    #     h = self.trunk(bert_embeddings)
    #     mu = self.mu_head(h)
    #     log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
    #     std = torch.exp(log_std)
    #     return Normal(mu, std).log_prob(z).sum(dim=1)
