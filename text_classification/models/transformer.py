from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.parameter import Parameter


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.fc_2(F.relu(self.fc_1(x)))
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.register_buffer("scale", scale)

    def forward(self, q, k, v, mask=None):

        energy = torch.einsum("...ij,...kj->...ik", [q, k]) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.dropout(torch.softmax(energy, dim=-1))
        output = torch.einsum("...qk,...kd->...qd", [attention, v])

        return output, attention


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, num_heads, dropout):
        super().__init__()

        assert hid_dim % num_heads == 0

        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.head_dim = hid_dim // num_heads

        self.q_proj_weight = Parameter(torch.Tensor(hid_dim, hid_dim))  # W_i^Q
        self.k_proj_weight = Parameter(torch.Tensor(hid_dim, hid_dim))  # W_i^K
        self.v_proj_weight = Parameter(torch.Tensor(hid_dim, hid_dim))  # W_i^V
        self.out_proj = Parameter(torch.Tensor(hid_dim, hid_dim))  # W^0

        scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        self.attention = ScaledDotProductAttention(scale, dropout)

        self._init_parameters()

    def _init_parameters(self):
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)
        xavier_uniform_(self.out_proj)

    def forward(self, query, key, value, mask=None):

        b_sz = query.shape[0]

        q = torch.einsum("...ij,jk->...ik", [query, self.q_proj_weight])  # QW_i^Q
        k = torch.einsum("...ij,jk->...ik", [key, self.k_proj_weight])  # KW_i^K
        v = torch.einsum("...ij,jk->...ik", [value, self.v_proj_weight])  # VW_i^V

        q = q.view(b_sz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b_sz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b_sz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        output, attention = self.attention(q, k, v, mask)

        x = output.transpose(1, 2).contiguous()

        x = x.view(b_sz, -1, self.hid_dim)

        x = torch.einsum("...ij,jk->...ik", [x, self.out_proj])

        return x, attention


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim: int, num_heads: int, pf_dim: int, dropout: float):
        super().__init__()

        self.attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.attn = MultiHeadAttentionLayer(hid_dim, num_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, pf_dim, dropout
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):

        _src, _ = self.attn(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.attn_layer_norm(src + self.dropout(_src))

        # position-wise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual connection and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src

class Encoder(nn.Module):
    def __init__(
        self,
        input_size, 
        hid_dim,
        n_layers,
        n_heads,
        pf_dim,
        dropout,
        padding_idx = 0,
        max_length = 284
    ):
        super().__init__()

        self.tok_embedding = nn.Embedding(input_size, hid_dim, padding_idx=padding_idx)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList(
            [EncoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)]
        )

        self.dropout = nn.Dropout(dropout)

        self.register_buffer("scale", torch.sqrt(torch.FloatTensor([hid_dim])))

    def make_mask(self, x):

        x_mask = (x != 0).unsqueeze(1).unsqueeze(2)

        return x_mask

    def forward(self, x):
        
        pos = torch.arange(x.shape[1], device=x.device)
        x_mask = self.make_mask(x)

        x = self.tok_embedding(x) * self.scale
        x = x + self.pos_embedding(pos).expand_as(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, x_mask)

        pooled_output = x.transpose(0,1)[0]
        return x, pooled_output

class TransformerWithClassifierHead(nn.Module):

    r"""
    Transformer with Classification Head

    Reference: `Vaswani et al. (2017). Attention is All you Need. <https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html>`_

    Args:
        input_size: Input size, for most cases size of vocabularly.
        num_class: Number of classes.
        hid_dim: Dimension of the transformer.
        n_layers: Number of encoder layers.
        n_heads: Number of heads used in multi-head attention.
        pf_dim: Dimension of the feed-forward layer.
        dropout: Dropout used in each encoder layer.
        mlp_dim: Dimension of the classification head.
        padding_idx: Index of the padding token.
        max_length: Max length of the transformer.

    Example::

        # for binary classification
        >>> model = TransformerWithClassifierHead(input_size=100, num_class=2)
    """

    def __init__(
        self,
        input_size: int,
        num_class: int,
        hid_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        pf_dim: int = 1024,
        dropout: float = 0.1,
        mlp_dim: int = 256,
        padding_idx: int = 0,
        max_length: int = 284,
    ):
        super().__init__()

        self.encoder = Encoder(input_size, hid_dim, n_layers, n_heads, pf_dim, dropout,
                                padding_idx, max_length)

        self.clf_head = nn.Sequential(
            nn.Linear(hid_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_class),
        )

    def forward(self, batch):

        x, _ = batch
        
        _, cls_output = self.encoder(x)

        x = self.clf_head(cls_output)
        return x
