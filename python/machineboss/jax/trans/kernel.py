"""Core scatter/gather kernel operations for TransMachine.

All operations: (cell, tm: TransMachine, semiring) -> cell

Uses sparse COO gather/scatter with pre-built masks from TransMachine.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..types import NEG_INF
from ..semiring import LogSemiring
from ..utils import scatter_logsumexp, scatter_max
from .machine import TransMachine


def _scatter_semiring(values: jnp.ndarray, indices: jnp.ndarray,
                      size: int, semiring: LogSemiring) -> jnp.ndarray:
    """Scatter values to indices using semiring aggregation."""
    if semiring.plus is jnp.logaddexp:
        return scatter_logsumexp(values, indices, size)
    else:
        return scatter_max(values, indices, size)


def propagate_silent(cell: jnp.ndarray, tm: TransMachine,
                     semiring: LogSemiring) -> jnp.ndarray:
    """Propagate silent transitions (forward) until convergence.

    Gather from cell[tm.src], mask by silent_mask, scatter to tm.dst.
    Converge via lax.while_loop.
    """
    S = tm.n_states

    def body_fn(carry):
        prev, _ = carry
        vals = prev[tm.src] + tm.log_w
        vals = jnp.where(tm.silent_mask, vals, NEG_INF)
        update = _scatter_semiring(vals, tm.dst, S, semiring)
        new = semiring.plus(cell, update)
        return new, prev

    def cond_fn(carry):
        new, prev = carry
        return jnp.any(jnp.abs(new - prev) > 1e-10)

    init = (cell, jnp.full_like(cell, NEG_INF))
    result, _ = jax.lax.while_loop(cond_fn, body_fn, init)
    return result


def propagate_silent_backward(cell: jnp.ndarray, tm: TransMachine,
                               semiring: LogSemiring) -> jnp.ndarray:
    """Propagate silent transitions (backward) until convergence.

    Gather from cell[tm.dst], scatter to tm.src.
    """
    S = tm.n_states

    def body_fn(carry):
        prev, _ = carry
        vals = tm.log_w + prev[tm.dst]
        vals = jnp.where(tm.silent_mask, vals, NEG_INF)
        update = _scatter_semiring(vals, tm.src, S, semiring)
        new = semiring.plus(cell, update)
        return new, prev

    def cond_fn(carry):
        new, prev = carry
        return jnp.any(jnp.abs(new - prev) > 1e-10)

    init = (cell, jnp.full_like(cell, NEG_INF))
    result, _ = jax.lax.while_loop(cond_fn, body_fn, init)
    return result


def emit_step_forward(cell: jnp.ndarray, prev: jnp.ndarray,
                      tm: TransMachine, mask: jnp.ndarray,
                      emission_w: jnp.ndarray,
                      semiring: LogSemiring) -> jnp.ndarray:
    """Apply emitting transitions from prev to cell (forward).

    Args:
        cell: (S,) current cell being filled
        prev: (S,) predecessor cell
        tm: TransMachine
        mask: (T,) boolean mask selecting which transitions fire
        emission_w: (T,) per-transition emission log-weights
        semiring: LogSemiring
    Returns:
        Updated cell.
    """
    S = tm.n_states
    vals = prev[tm.src] + tm.log_w + emission_w
    vals = jnp.where(mask, vals, NEG_INF)
    update = _scatter_semiring(vals, tm.dst, S, semiring)
    return semiring.plus(cell, update)


def emit_step_backward(cell: jnp.ndarray, future: jnp.ndarray,
                        tm: TransMachine, mask: jnp.ndarray,
                        emission_w: jnp.ndarray,
                        semiring: LogSemiring) -> jnp.ndarray:
    """Apply emitting transitions from future to cell (backward).

    Args:
        cell: (S,) current cell being filled
        future: (S,) successor cell
        tm: TransMachine
        mask: (T,) boolean mask selecting which transitions fire
        emission_w: (T,) per-transition emission log-weights
        semiring: LogSemiring
    Returns:
        Updated cell.
    """
    S = tm.n_states
    vals = tm.log_w + emission_w + future[tm.dst]
    vals = jnp.where(mask, vals, NEG_INF)
    update = _scatter_semiring(vals, tm.src, S, semiring)
    return semiring.plus(cell, update)


def to_matrix(tm: TransMachine, mask: jnp.ndarray,
              emission_w: jnp.ndarray,
              semiring: LogSemiring) -> jnp.ndarray:
    """Scatter transitions into (S, S) matrix.

    Args:
        tm: TransMachine
        mask: (T,) boolean mask selecting which transitions to include
        emission_w: (T,) per-transition emission log-weights
        semiring: LogSemiring
    Returns:
        (S, S) transition matrix [src, dst].
    """
    S = tm.n_states
    vals = tm.log_w + emission_w
    vals = jnp.where(mask, vals, NEG_INF)
    flat_idx = tm.src * S + tm.dst
    flat = _scatter_semiring(vals, flat_idx, S * S, semiring)
    return flat.reshape(S, S)


def emission_weights_1d(tm: TransMachine, seq_emission: jnp.ndarray,
                        is_input: bool) -> jnp.ndarray:
    """Compute per-transition emission weights for 1D DP.

    Args:
        tm: TransMachine
        seq_emission: (n_tokens,) emission log-probs for this position
        is_input: True if consuming input, False if output
    Returns:
        (T,) per-transition emission log-weights.
    """
    tok = tm.in_tok if is_input else tm.out_tok
    return seq_emission[tok]


def emission_weights_2d(tm: TransMachine,
                        in_emission: jnp.ndarray | None,
                        out_emission: jnp.ndarray | None) -> jnp.ndarray:
    """Compute per-transition emission weights for 2D DP match step.

    Args:
        tm: TransMachine
        in_emission: (n_in,) input emission log-probs, or None
        out_emission: (n_out,) output emission log-probs, or None
    Returns:
        (T,) per-transition emission log-weights.
    """
    w = jnp.zeros(tm.n_transitions)
    if in_emission is not None:
        w = w + in_emission[tm.in_tok]
    if out_emission is not None:
        w = w + out_emission[tm.out_tok]
    return w
