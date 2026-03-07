"""Flax/Linen MSA Transformer with row and column attention.

Adapted from the MSA Transformer architecture (Rao et al., 2021).
Row attention operates along the sequence length (within each sequence),
column attention operates across sequences at each position.

Input:  (N, L, 21) one-hot MSA (20 AA + gap)
Output: (L, d_model) mean-pooled sequence representation
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn


class RowAttention(nn.Module):
    """Multi-head self-attention along L (within each sequence).

    Input:  (N, L, D)
    Output: (N, L, D)
    """
    n_heads: int = 4

    @nn.compact
    def __call__(self, x):
        D = x.shape[-1]
        head_dim = D // self.n_heads
        N, L, _ = x.shape

        # QKV projections
        qkv = nn.Dense(3 * D)(x)  # (N, L, 3D)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Reshape to heads: (N, n_heads, L, head_dim)
        q = q.reshape(N, L, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(N, L, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(N, L, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        # Attention
        scale = jnp.sqrt(jnp.float32(head_dim))
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale  # (N, H, L, L)
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.matmul(attn, v)  # (N, H, L, head_dim)

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(N, L, D)
        return nn.Dense(D)(out)


class ColumnAttention(nn.Module):
    """Multi-head self-attention along N (across sequences at each position).

    Input:  (N, L, D)
    Output: (N, L, D)
    """
    n_heads: int = 4

    @nn.compact
    def __call__(self, x):
        D = x.shape[-1]
        head_dim = D // self.n_heads
        N, L, _ = x.shape

        # Transpose to (L, N, D) for column-wise attention
        x_t = x.transpose(1, 0, 2)  # (L, N, D)

        qkv = nn.Dense(3 * D)(x_t)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(L, N, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(L, N, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(L, N, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        scale = jnp.sqrt(jnp.float32(head_dim))
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.matmul(attn, v)  # (L, H, N, head_dim)

        out = out.transpose(0, 2, 1, 3).reshape(L, N, D)
        out = nn.Dense(D)(out)
        return out.transpose(1, 0, 2)  # back to (N, L, D)


class MSATransformerBlock(nn.Module):
    """One block: RowAttn + LN + ColAttn + LN."""
    n_heads: int = 4

    @nn.compact
    def __call__(self, x):
        # Row attention + residual + LN
        h = nn.LayerNorm()(x + RowAttention(n_heads=self.n_heads)(x))
        # Column attention + residual + LN
        h = nn.LayerNorm()(h + ColumnAttention(n_heads=self.n_heads)(h))
        return h


class MSATransformer(nn.Module):
    """MSA Transformer: embed + row/column attention blocks + mean-pool.

    Input:  (N, L, 21) one-hot MSA
    Output: (L, d_model) mean-pooled representation across sequences
    """
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2

    @nn.compact
    def __call__(self, x):
        # Input embedding: (N, L, 21) -> (N, L, d_model)
        h = nn.Dense(self.d_model)(x)

        # Transformer blocks
        for _ in range(self.n_layers):
            h = MSATransformerBlock(n_heads=self.n_heads)(h)

        # Mean-pool over N (sequences) -> (L, d_model)
        return jnp.mean(h, axis=0)
