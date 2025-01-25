import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = einops.rearrange(
            torch.arange(0, max_len, dtype=torch.float), "l -> l 1"
        )

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = einops.rearrange(pe, "l d -> 1 l d")
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1]].requires_grad_(False)
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embeddings, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model))


class AttentionBlock(nn.Module):
    def __init__(self, d_model, dropout):
        super(AttentionBlock, self).__init__()

        d_q = d_k = 7  # (d_q = d_k)
        d_v = 8

        self.dropout = nn.Dropout(dropout)
        self.q_w = nn.Linear(d_model, d_k)  # (d_model, d_k)
        self.k_w = nn.Linear(d_model, d_k)  # (d_model, d_k)
        self.v_w = nn.Linear(d_model, d_v)  # (d_model, d_v)

        self.out = nn.Linear(d_v, d_model)

    def forward(self, q, k, v, mask=None):
        q = self.q_w(q)  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_k)
        k = self.k_w(k)  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_k)
        v = self.v_w(v)  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_v)

        print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")

        socres = torch.einsum(
            "b d i, b e i -> b d e", q, k
        )  # (batch_size, seq_len, seq_len)
        print(f"socres: {socres.shape}")

        if mask is not None:
            socres = socres.masked_fill(mask == 0, -1e9)

        attn = F.softmax(socres, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum(
            "b d e, b e o -> b d o", attn, v
        )  # (batch_size, seq_len, d_v)
        print(out.shape)
        out = self.out(out)

        return out, attn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Residual(nn.Module):
    def __init__(self, features, dropout):
        super(Residual, self).__init__()

        self.norm = LayerNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        features,
        self_attn_block: AttentionBlock,
        feed_forward: FeedForward,
        dropout,
    ):
        super(EncoderBlock, self).__init__()

        self.self_attn_block = self_attn_block
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList(
            [Residual(features, dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connection[0](
            x, lambda x: self.self_attn_block(x, x, x, src_mask)
        )
        x = self.residual_connection[1](x, self.feed_forward)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        features,
        self_attn_block: AttentionBlock,
        feed_forward: FeedForward,
        dropout,
    ):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                EncoderBlock(features, self_attn_block, feed_forward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)

        return x
