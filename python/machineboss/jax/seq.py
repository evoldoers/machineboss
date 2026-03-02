"""Sequence representations for DP algorithms.

TokenSeq: observed token sequence (one-hot emission weights)
PSWMSeq: position-specific weight matrix (log-probability emission weights)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .types import NEG_INF


@dataclass
class TokenSeq:
    """An observed token sequence with one-hot emission weights."""
    tokens: jnp.ndarray  # (L,) int32 token indices (1-based; 0=empty)

    def emission_weights(self, n_tokens: int) -> jnp.ndarray:
        """Return log-space emission weights: (L, n_tokens).

        For each position, weight is 0.0 for the observed token and -inf elsewhere.
        Token index 0 (empty) is never emitted.
        """
        L = len(self.tokens)
        # One-hot: (L, n_tokens) with 0.0 at token position, NEG_INF elsewhere
        weights = jnp.full((L, n_tokens), NEG_INF)
        weights = weights.at[jnp.arange(L), self.tokens].set(0.0)
        return weights

    def __len__(self):
        return len(self.tokens)


@dataclass
class PSWMSeq:
    """Position-specific weight matrix: log-probability emission weights."""
    log_probs: jnp.ndarray  # (L, n_tokens) log-probabilities

    def emission_weights(self, n_tokens: int) -> jnp.ndarray:
        """Return log-space emission weights: (L, n_tokens).

        Directly returns the stored log-probability matrix.
        Token index 0 (empty) column should be NEG_INF.
        """
        return self.log_probs

    def __len__(self):
        return self.log_probs.shape[0]


def wrap_seq(seq, n_tokens: int):
    """Wrap a raw jnp.ndarray as TokenSeq, or pass through TokenSeq/PSWMSeq."""
    if seq is None:
        return None
    if isinstance(seq, (TokenSeq, PSWMSeq)):
        return seq
    return TokenSeq(tokens=seq)


def pad_length(length: int, ratio: float = 1.5) -> int:
    """Round length up to the nearest value in a geometric series.

    Produces a discrete set of padded lengths so that JAX reuses
    compiled kernels instead of recompiling for every distinct length.
    Series: 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, ...

    Args:
        length: actual sequence length
        ratio: geometric ratio (default 1.5)
    Returns:
        Padded length >= length.
    """
    if length <= 4:
        return length
    import math
    # Find smallest k such that round(4 * ratio^k) >= length
    k = math.ceil(math.log(length / 4) / math.log(ratio))
    return max(length, round(4 * ratio ** k))


def pad_token_seq(seq: TokenSeq, padded_len: int) -> tuple[TokenSeq, int]:
    """Pad a TokenSeq to padded_len with empty tokens (0).

    Returns (padded_seq, original_length).
    """
    L = len(seq)
    if L >= padded_len:
        return seq, L
    pad = jnp.zeros(padded_len - L, dtype=jnp.int32)
    padded = jnp.concatenate([seq.tokens, pad])
    return TokenSeq(tokens=padded), L


def pad_pswm_seq(seq: PSWMSeq, padded_len: int) -> tuple[PSWMSeq, int]:
    """Pad a PSWMSeq to padded_len with NEG_INF rows.

    Returns (padded_seq, original_length).
    """
    L = len(seq)
    if L >= padded_len:
        return seq, L
    n_tokens = seq.log_probs.shape[1]
    pad = jnp.full((padded_len - L, n_tokens), NEG_INF)
    padded = jnp.concatenate([seq.log_probs, pad], axis=0)
    return PSWMSeq(log_probs=padded), L
