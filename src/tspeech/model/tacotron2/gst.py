from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# Currently based on the Nvidia Mellotron GST code
# https://github.com/NVIDIA/mellotron/blob/master/modules.py


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, ref_enc_filters, n_mel_channels, ref_enc_gru_size):

        super().__init__()
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters

        convs = [
            nn.Conv2d(
                in_channels=filters[i],
                out_channels=filters[i + 1],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=ref_enc_filters[i]) for i in range(K)]
        )

        out_channels = self.calculate_channels(n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=ref_enc_gru_size,
            batch_first=True,
        )
        self.n_mel_channels = n_mel_channels
        self.ref_enc_gru_size = ref_enc_gru_size

    def forward(self, inputs, input_lengths: Optional[Tensor] = None):
        out = inputs.view(inputs.size(0), 1, -1, self.n_mel_channels)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        if input_lengths is not None:
            input_lengths = torch.ceil(input_lengths.float() / 2 ** len(self.convs))
            out = nn.utils.rnn.pack_padded_sequence(
                out, input_lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        self.gru.flatten_parameters()
        _, out = self.gru(out)
        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):
    """
    inputs --- [N, token_embedding_size//2]
    """

    def __init__(
        self,
        token_num: int,
        token_embedding_size: int,
        num_heads: int,
        ref_enc_gru_size: int,
    ):
        super().__init__()
        self.token_embedding_size = token_embedding_size
        self.num_heads = num_heads

        self.embed = nn.Parameter(
            torch.FloatTensor(token_num, token_embedding_size // num_heads)
        )
        d_q = ref_enc_gru_size
        d_k = token_embedding_size // num_heads

        self.W_query = nn.Linear(
            in_features=d_q, out_features=token_embedding_size, bias=False
        )

        self.attention = nn.MultiheadAttention(
            token_embedding_size,
            num_heads=num_heads,
            kdim=d_k,
            vdim=d_k,
            batch_first=True,
        )

        nn.init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs: Tensor) -> Tensor:
        N = inputs.size(0)
        query = inputs.unsqueeze(1)
        keys = (
            torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)
        )  # [N, token_num, token_embedding_size // num_heads]
        style_embed, _ = self.attention(
            query=self.W_query(query),
            key=keys,
            value=keys,
            need_weights=False,
            average_attn_weights=False,
        )

        return style_embed


class GST(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.reference_encoder = ReferenceEncoder(
            ref_enc_filters=[32, 32, 64, 64, 128, 128],
            n_mel_channels=80,
            ref_enc_gru_size=128,
        )
        self.stl: nn.Module = STL(
            token_num=10,
            token_embedding_size=out_dim,
            num_heads=8,
            ref_enc_gru_size=128,
        )

    def forward(self, mels, mel_len, weights=None):
        out = self.reference_encoder(mels, mel_len)
        out = self.stl(out)
        return out
