"""
Shared MiniGPT model components used across the learning notebooks.

Architecture (as of medium enhancements):
- ReLU² activation in feed-forward network
- No biases in any Linear layer
- Logit softcapping (tanh, cap=15)
- RMSNorm instead of LayerNorm (no learnable scale/shift)
- QK Norm: Q and K normalized before attention dot product
- RoPE: rotary positional embeddings applied inside attention, no additive pos encoding
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# REFERENCE: Code removed during medium enhancements (kept for learning)
# ---------------------------------------------------------------------------
#
# 1. SinusoidalPositionalEncoding — replaced by RoPE
#
# class SinusoidalPositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, max_len: int = 5000) -> None:
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
#         )
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer("pe", pe)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         seq_len = x.size(1)
#         return x + self.pe[:seq_len, :]
#
# ---------------------------------------------------------------------------
#
# 2. LearnedPositionalEmbedding — replaced by RoPE
#
# class LearnedPositionalEmbedding(nn.Module):
#     def __init__(self, d_model: int, max_len: int) -> None:
#         super().__init__()
#         self.position_embedding = nn.Embedding(max_len, d_model)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         batch_size, seq_len, _ = x.size()
#         positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
#         return x + self.position_embedding(positions)
#
# # Backward-compatible alias
# PositionalEncoding = SinusoidalPositionalEncoding
#
# ---------------------------------------------------------------------------
#
# 3. _build_position_encoder() — part of MiniGPT, removed with positional classes
#
# def _build_position_encoder(self, d_model: int, max_seq_len: int) -> nn.Module:
#     if self.position_encoding_type == "learned":
#         return LearnedPositionalEmbedding(d_model, max_seq_len)
#     if self.position_encoding_type == "sinusoidal":
#         return SinusoidalPositionalEncoding(d_model, max_seq_len)
#     raise ValueError(f"Unsupported position_encoding_type: {self.position_encoding_type}")
#
# ---------------------------------------------------------------------------
#
# 4. position_encoding_type param + self.pos_encoding in MiniGPT.__init__
#    (was used to select sinusoidal vs learned; notebooks used "learned")
#
# def __init__(self, ..., position_encoding_type: str = "sinusoidal") -> None:
#     ...
#     self.position_encoding_type = position_encoding_type.lower()
#     self.pos_encoding = self._build_position_encoder(d_model, max_seq_len)
#
# # And in forward():
# x = self.pos_encoding(x)   # additive position signal injected here, before blocks
#
# ---------------------------------------------------------------------------
#
# 5. nn.LayerNorm instances — replaced by RMSNorm (no learnable params)
#
# In GPTBlock.__init__:
#     self.norm1 = nn.LayerNorm(d_model)   # before attention
#     self.norm2 = nn.LayerNorm(d_model)   # before feed-forward
#
# In GPTBlock.forward:
#     attn_output = self.attention(self.norm1(x))
#     ffn_output  = self.feed_forward(self.norm2(x))
#
# In MiniGPT.__init__:
#     self.ln_final = nn.LayerNorm(d_model)  # final norm before output projection
#
# In MiniGPT.forward:
#     x = self.ln_final(x)
#
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm: normalize by RMS only, no learnable scale or shift."""
    return F.rms_norm(x, (x.size(-1),))


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_rotary_embeddings(
    seq_len: int, head_dim: int, base: int = 10000, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin rotation matrices for RoPE.

    Returns cos, sin each of shape (1, 1, seq_len, head_dim // 2),
    ready to broadcast over (batch, heads, seq, dim).
    """
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)                    # (seq_len, head_dim/2)
    cos = freqs.cos()[None, None, :, :]                 # (1, 1, seq_len, head_dim/2)
    sin = freqs.sin()[None, None, :, :]
    return cos, sin


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embeddings to x (shape: B, heads, T, head_dim)."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation.lower()

        if self.activation not in {"relu", "gelu"}:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        if self.activation == "gelu":
            x = F.gelu(x)
        else:
            x = F.relu(x).square()  # ReLU²: faster convergence, lower loss than plain ReLU
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalMultiHeadAttention(nn.Module):
    """Multi-head causal attention with QK Norm and RoPE."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos_sin: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()

        # Project to Q, K, V and split into heads: (B, heads, T, head_dim)
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # QK Norm: normalize Q and K before dot product to stabilize attention
        Q = rms_norm(Q)
        K = rms_norm(K)

        # RoPE: apply rotary positional embeddings to Q and K
        cos, sin = cos_sin
        Q = apply_rotary_emb(Q, cos, sin)
        K = apply_rotary_emb(K, cos, sin)

        # Scaled dot-product attention with causal mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attn_output = torch.matmul(attention_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.W_o(attn_output)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class GPTBlock(nn.Module):
    """A single GPT transformer block with pre-norm (RMSNorm)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.attention = CausalMultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation=activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos_sin: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # Pre-norm: normalize before attention and FFN, then add residual
        x = x + self.dropout(self.attention(rms_norm(x), cos_sin))
        x = x + self.dropout(self.feed_forward(rms_norm(x)))
        return x


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class MiniGPT(nn.Module):
    """Complete mini GPT model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.activation = activation.lower()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                GPTBlock(d_model, num_heads, d_ff, dropout, activation=self.activation)
                for _ in range(num_layers)
            ]
        )
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Precompute RoPE cos/sin buffers (not saved to checkpoint — recomputed on load)
        head_dim = d_model // num_heads
        cos, sin = precompute_rotary_embeddings(max_seq_len, head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.size()

        x = self.token_embedding(token_ids)   # (B, T, d_model)
        x = self.dropout(x)

        # Slice RoPE buffers to current sequence length
        cos_sin = (self.rope_cos[:, :, :T, :], self.rope_sin[:, :, :T, :])

        for block in self.blocks:
            x = block(x, cos_sin)

        x = rms_norm(x)
        logits = self.output_projection(x)
        logits = 15.0 * torch.tanh(logits / 15.0)  # softcap: smoothly clamp to [-15, 15]
        return logits
