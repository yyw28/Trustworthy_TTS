"""Reinforcement Learning Policy Network for GST Weights Optimization.

Based on the approach from "Reinforcement Learning for Emotional Text-to-Speech Synthesis"
(arXiv:2104.01408v2), adapted for trustworthiness optimization.
"""
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions import Categorical


class RLGSTPolicy(nn.Module):
    """
    Policy network that outputs GST weights using REINFORCE algorithm.
    
    The policy network takes BERT embeddings and outputs a distribution over GST weights.
    During training, we sample from this distribution and use REINFORCE to optimize
    based on trustworthiness rewards from HubERT.
    """
    
    def __init__(
        self,
        bert_hidden_size: int = 768,
        gst_token_num: int = 10,
        hidden_dim: int = 256,
        temperature: float = 1.0,
    ):
        """
        Initialize RL GST Policy network.
        
        Parameters
        ----------
        bert_hidden_size : int
            Hidden size of BERT embeddings (default: 768 for bert-base)
        gst_token_num : int
            Number of GST tokens (default: 10)
        hidden_dim : int
            Hidden dimension for policy network
        temperature : float
            Temperature for sampling (higher = more exploration)
        """
        super().__init__()
        
        self.gst_token_num = gst_token_num
        self.temperature = temperature
        
        # Policy network: BERT embedding -> GST weight logits
        self.policy_network = nn.Sequential(
            nn.Linear(bert_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, gst_token_num),
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        bert_embeddings: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Forward pass through policy network.
        
        Parameters
        ----------
        bert_embeddings : Tensor
            BERT embeddings of shape (batch_size, bert_hidden_size)
        deterministic : bool
            If True, return deterministic (greedy) weights instead of sampling
            
        Returns
        -------
        Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
            - gst_weights: Sampled or deterministic GST weights (batch_size, gst_token_num)
            - log_probs: Log probabilities of sampled actions (for REINFORCE)
            - entropy: Entropy of the distribution (for exploration bonus)
        """
        # Get logits from policy network
        logits = self.policy_network(bert_embeddings)  # (batch_size, gst_token_num)
        
        # Apply temperature scaling
        logits = logits / self.temperature
        
        if deterministic:
            # Greedy selection: take the token with highest probability
            gst_weights = F.softmax(logits, dim=1)
            # For deterministic, we can use argmax or softmax
            # Using softmax for smooth gradients if needed
            return gst_weights, None, None
        
        # Create categorical distribution
        dist = Categorical(logits=logits)
        
        # Sample GST token indices
        sampled_indices = dist.sample()  # (batch_size,)
        
        # Convert to one-hot weights (or use soft weights)
        # Option 1: One-hot (hard selection)
        # gst_weights = F.one_hot(sampled_indices, num_classes=self.gst_token_num).float()
        
        # Option 2: Soft weights using Gumbel-Softmax for differentiable sampling
        # This allows gradients to flow through
        gst_weights = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=1)
        
        # Compute log probabilities for REINFORCE
        log_probs = dist.log_prob(sampled_indices)  # (batch_size,)
        
        # Compute entropy for exploration bonus
        entropy = dist.entropy()  # (batch_size,)
        
        return gst_weights, log_probs, entropy
    
    def get_log_probs(self, bert_embeddings: Tensor, gst_weights: Tensor) -> Tensor:
        """
        Compute log probabilities of given GST weights under the policy.
        
        Parameters
        ----------
        bert_embeddings : Tensor
            BERT embeddings
        gst_weights : Tensor
            GST weights to evaluate
            
        Returns
        -------
        Tensor
            Log probabilities
        """
        logits = self.policy_network(bert_embeddings) / self.temperature
        
        # Convert weights to indices (argmax)
        indices = gst_weights.argmax(dim=1)
        
        dist = Categorical(logits=logits)
        return dist.log_prob(indices)


class RLGSTRewardFunction:
    """
    Reward function for RL training.
    
    Computes rewards based on HubERT trustworthiness scores.
    Can also include other reward components (e.g., speech quality).
    """
    
    def __init__(
        self,
        trustworthiness_weight: float = 1.0,
        baseline_weight: float = 0.0,
        use_baseline: bool = True,
    ):
        """
        Initialize reward function.
        
        Parameters
        ----------
        trustworthiness_weight : float
            Weight for trustworthiness reward component
        baseline_weight : float
            Weight for baseline reward (for variance reduction)
        use_baseline : bool
            Whether to use baseline for variance reduction
        """
        self.trustworthiness_weight = trustworthiness_weight
        self.baseline_weight = baseline_weight
        self.use_baseline = use_baseline
        
        # Running baseline for variance reduction
        self.running_baseline = None
        self.baseline_momentum = 0.9
    
    def compute_reward(
        self,
        trustworthiness_scores: Tensor,
        baseline: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute reward from trustworthiness scores.
        
        Parameters
        ----------
        trustworthiness_scores : Tensor
            Trustworthiness logits/scores from HubERT (batch_size, 1)
        baseline : Optional[Tensor]
            Baseline values for variance reduction
            
        Returns
        -------
        Tuple[Tensor, Tensor]
            - rewards: Computed rewards (batch_size,)
            - advantages: Advantages (rewards - baseline) for policy gradient
        """
        # Convert logits to probabilities
        trustworthiness_probs = torch.sigmoid(trustworthiness_scores).squeeze(-1)  # (batch_size,)
        
        # Reward is the trustworthiness probability
        rewards = self.trustworthiness_weight * trustworthiness_probs
        
        # Update running baseline
        if self.use_baseline:
            if self.running_baseline is None:
                self.running_baseline = rewards.mean().detach()
            else:
                self.running_baseline = (
                    self.baseline_momentum * self.running_baseline
                    + (1 - self.baseline_momentum) * rewards.mean().detach()
                )
            
            # Use provided baseline or running baseline
            if baseline is None:
                baseline = self.running_baseline
            
            # Compute advantages (rewards - baseline)
            advantages = rewards - baseline
        else:
            advantages = rewards
        
        return rewards, advantages
    
    def reset_baseline(self):
        """Reset the running baseline."""
        self.running_baseline = None







