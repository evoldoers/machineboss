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

    Uses max-shift-exp-sum-log pattern for numerical stability and
    correct handling of duplicate indices.
    """
    # Step 1: find max per destination
    max_per_dst = jnp.full(size, NEG_INF).at[indices].max(values)
    # Step 2: shift values by max, exponentiate, sum
    shifted = values - max_per_dst[indices]
    sum_exp = jnp.zeros(size).at[indices].add(jnp.exp(shifted))
    # Step 3: log and unshift
    result = max_per_dst + jnp.log(sum_exp)
    # Where no values were scattered, keep NEG_INF
    result = jnp.where(max_per_dst > NEG_INF + 1, result, NEG_INF)
    return result


def scatter_max(values: jnp.ndarray, indices: jnp.ndarray, size: int) -> jnp.ndarray:
    """Scatter values into an array of given size using max aggregation.

    For each index i, result[i] = max(values[indices == i]).
    """
    return jnp.full(size, NEG_INF).at[indices].max(values)
