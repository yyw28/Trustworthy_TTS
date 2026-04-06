"""
Reinforcement Learning Policy Network for *soft* GST weights optimization (continuous action).
- You DO sample a continuous 10-dim vector of GST weights (on the simplex).
- Correct REINFORCE: log_prob corresponds to the sampled latent z that produced the weights.

We use a Logistic-Normal policy:
  z ~ Normal(mu(x), sigma(x))
  gst_weights = softmax(z / temperature)
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions import Normal


class RLGSTPolicy(nn.Module):
    """BERT → Normal(mu, sigma) → z → softmax(z/temp) = GST weights.

    Returns ``gst_weights``, ``log_probs`` (per-dim, for REINFORCE), ``z``, and
    scalar diagnostics ``mu_std``, ``log_std_mean``, ``std_mean`` for logging.
    """

    def __init__(
        self,
        bert_hidden_size: int = 1024,
        gst_token_num: int = 10,
        gst_heads: int = 8, #8 GST heads
        hidden_dim: int = 256,
        temperature: float = 0.10,
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
            nn.Linear(bert_hidden_size, hidden_dim * gst_heads), #[1024, 256*8]
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * gst_heads, hidden_dim * gst_heads), #[256*8, 256*8]
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.mu_head = nn.Linear(hidden_dim * gst_heads, gst_token_num * gst_heads) #[256*8, 10*8]
        self.log_std_head = nn.Linear(hidden_dim * gst_heads, gst_token_num * gst_heads) #[256*8, 10*8]

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.constant_(self.log_std_head.bias, init_log_std)

    def forward(self, bert_embeddings: Tensor, deterministic: bool = False):
        batch_size = bert_embeddings.shape[0]
        h = self.trunk(bert_embeddings) #[batch_size, 1024] -> [batch_size, 256*8]
        mu = self.mu_head(h).view(batch_size * self.gst_heads, self.gst_token_num) #[batch_size, 256*8] -> [batch_size*8, 10]
        log_std = (
            self.log_std_head(h)
            .clamp(self.log_std_min, self.log_std_max)
            .view(batch_size * self.gst_heads, self.gst_token_num) #[batch_size, 256*8] -> [batch_size*8, 10]
        )
        std = torch.exp(log_std) #[batch_size*8, 10] -> [batch_size*8, 10]
        dist = Normal(mu, std) #[batch_size*8, 10] -> [batch_size*8, 10]
        if deterministic:
            z = mu
        else:
            z = dist.rsample() #[batch_size*8, 10] -> [batch_size*8, 10]
        log_probs = dist.log_prob(z) #[batch_size*8, 10] -> [batch_size*8, 10]

        gst_weights = F.softmax(z / self.temperature, dim=1) #[batch_size*8, 10] -> [batch_size*8, 10]
        md = mu.detach()
        mu_std = md.std(unbiased=False) if md.numel() > 1 else md.new_zeros(())
        log_std_mean = log_std.detach().mean()
        std_mean = std.detach().mean()
        return gst_weights, log_probs, z, mu_std, log_std_mean, std_mean

    # def get_log_probs(self, bert_embeddings: Tensor, z: Tensor) -> Tensor:
    #     """Log prob of z under current policy (for REINFORCE)."""
    #     h = self.trunk(bert_embeddings)
    #     mu = self.mu_head(h)
    #     log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
    #     std = torch.exp(log_std)
    #     return Normal(mu, std).log_prob(z).sum(dim=1)
