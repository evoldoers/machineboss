"""Utility functions for JAX DP algorithms."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import NEG_INF


def log_sum_exp_pair(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable log(exp(a) + exp(b))."""
    return jnp.logaddexp(a, b)


def log_sum_exp_vec(x: jnp.ndarray) -> jnp.ndarray:
    """Log-sum-exp over the last axis."""
    return jax.nn.logsumexp(x, axis=-1)


def scatter_logsumexp(values: jnp.ndarray, indices: jnp.ndarray, size: int) -> jnp.ndarray:
    """Scatter values into an array of given size using logsumexp aggregation.

    For each index i, result[i] = logsumexp(values[indices == i]).
    """
    result = jnp.full(size, NEG_INF)
    # Use segment_logsumexp approach: scatter then reduce
    result = result.at[indices].set(
        jnp.logaddexp(result[indices], values)
    )
    return result
