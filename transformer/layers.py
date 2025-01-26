import copy
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
    def __init__(self, d_model, dropout, d_k=64, d_v=64):
        super(AttentionBlock, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.q_w = nn.Linear(d_model, d_k)  # (d_model, d_k)
        self.k_w = nn.Linear(d_model, d_k)  # (d_model, d_k)
        self.v_w = nn.Linear(d_model, d_v)  # (d_model, d_v)

        self.out = nn.Linear(d_v, d_model)

    def forward(self, q, k, v, mask=None):
        q = self.q_w(q)  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_k)
        k = self.k_w(k)  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_k)
        v = self.v_w(v)  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_v)

        # print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")

        self.socres = torch.einsum(
            "b d i, b e i -> b d e", q, k
        )  # (batch_size, seq_len, seq_len)
        # print(f"socres: {self.socres.shape}")

        if mask is not None:
            self.socres = self.socres.masked_fill(mask == 0, -1e9)

        self.socres = F.softmax(self.socres, dim=-1)
        self.socres = self.dropout(self.socres)

        out = torch.einsum(
            "b d e, b e o -> b d o", self.socres, v
        )  # (batch_size, seq_len, d_v)

        # print(out.shape)
        out = self.out(out)

        return out


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

    def forward(self, x):
        x = self.residual_connection[0](x, lambda x: self.self_attn_block(x, x, x))
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

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        features,
        self_attn_block: AttentionBlock,
        src_attn_block: AttentionBlock,
        feed_forward: FeedForward,
        dropout,
    ):
        super(DecoderBlock, self).__init__()

        self.self_attn_block = self_attn_block
        self.src_attn_block = src_attn_block
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList(
            [Residual(features, dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, tgt_mask):
        x = self.residual_connection[0](
            x, lambda x: self.self_attn_block(x, x, x, tgt_mask)
        )
        x = self.residual_connection[1](
            x,
            lambda x: self.src_attn_block(x, encoder_output, encoder_output),
        )
        x = self.residual_connection[2](x, self.feed_forward)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers,
        features,
        self_attn_block: AttentionBlock,
        src_attn_block: AttentionBlock,
        feed_forward: FeedForward,
        dropout,
    ):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    features, self_attn_block, src_attn_block, feed_forward, dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = LayerNorm(features)

    def forward(self, x, encoder_output, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask)

        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(ProjectionLayer, self).__init__()

        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # print(x.shape)
        return F.log_softmax(self.linear(x), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: Embeddings,
        tgt_embed: Embeddings,
        src_positional_encoding: PositionalEncoding,
        tgt_positional_encoding: PositionalEncoding,
        generator: ProjectionLayer,
    ):
        super(Transformer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_positional_encoding = src_positional_encoding
        self.tgt_positional_encoding = tgt_positional_encoding
        self.generator = generator

    def encode(self, src):

        src = self.src_positional_encoding(self.src_embed(src))
        return self.encoder(src)

    def decode(self, tgt, encoder_output, tgt_mask):

        tgt = self.tgt_positional_encoding(self.tgt_embed(tgt))
        return self.decoder(tgt, encoder_output, tgt_mask)

    def forward(self, src, tgt, tgt_mask):

        encoder_output = self.encode(src)
        return self.decode(tgt, encoder_output, tgt_mask)

    # def


def make_model(
    src_vocab_size,
    tgt_vocab_size,
    num_layers=6,
    d_model=512,
    d_ff=2048,
    d_k=64,
    d_v=64,
    num_heads=8,
    dropout=0.1,
):
    c = copy.deepcopy

    attn = AttentionBlock(d_model, dropout, d_k, d_v)
    ff = FeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = Transformer(
        encoder=Encoder(
            num_layers,
            d_model,
            c(attn),
            c(ff),
            dropout,
        ),
        decoder=Decoder(
            num_layers,
            d_model,
            c(attn),
            c(attn),
            c(ff),
            dropout,
        ),
        src_embed=Embeddings(src_vocab_size, d_model),
        tgt_embed=Embeddings(tgt_vocab_size, d_model),
        src_positional_encoding=c(position),
        tgt_positional_encoding=c(position),
        generator=ProjectionLayer(d_model, tgt_vocab_size),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


class Inference:
    """Object for holding a model during inference and some utility methods used when training the model."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def greedy_decode(model, src, max_len, start_symbol):
        memory = model.encode(src)
        ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len - 1):
            out = model.decode(
                ys,
                memory,
                Inference.subsequent_mask(ys.shape[1]).type_as(src.data),
            )
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat(
                [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )
        return ys

    @staticmethod
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
            torch.uint8
        )
        return subsequent_mask == 0

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Inference.subsequent_mask(tgt.shape[-1]).type_as(
            tgt_mask.data
        )
        return tgt_mask
