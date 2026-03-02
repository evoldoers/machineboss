"""Sparse COO transition kernel for DP algorithms.

Operates on the sparse (COO) representation in JAXMachine:
  log_weights, src_states, dst_states, in_tokens, out_tokens.

All operations are parameterized by a LogSemiring for Forward vs Viterbi.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import JAXMachine, NEG_INF
from .semiring import LogSemiring
from .utils import scatter_logsumexp, scatter_max


def _scatter_semiring(values: jnp.ndarray, indices: jnp.ndarray,
                      size: int, semiring: LogSemiring) -> jnp.ndarray:
    """Scatter values to indices using semiring aggregation."""
    if semiring.plus is jnp.logaddexp:
        return scatter_logsumexp(values, indices, size)
    else:
        return scatter_max(values, indices, size)


def propagate_silent_sparse(cell: jnp.ndarray, machine: JAXMachine,
                            semiring: LogSemiring) -> jnp.ndarray:
    """Propagate silent transitions (in=0, out=0) using sparse COO.

    Forward direction: accumulate into destination states.
    """
    S = machine.n_states
    # Mask for silent transitions
    silent_mask = (machine.in_tokens == 0) & (machine.out_tokens == 0)

    src = machine.src_states
    dst = machine.dst_states
    w = machine.log_weights

    def body_fn(carry):
        prev, _ = carry
        vals = prev[src] + w
        # Zero out non-silent transitions
        vals = jnp.where(silent_mask, vals, NEG_INF)
        update = _scatter_semiring(vals, dst, S, semiring)
        new = semiring.plus(cell, update)
        return new, prev

    def cond_fn(carry):
        new, prev = carry
        return jnp.any(jnp.abs(new - prev) > 1e-10)

    init = (cell, jnp.full_like(cell, NEG_INF))
    result, _ = jax.lax.while_loop(cond_fn, body_fn, init)
    return result


def propagate_silent_backward_sparse(cell: jnp.ndarray, machine: JAXMachine,
                                      semiring: LogSemiring) -> jnp.ndarray:
    """Propagate silent transitions backward using sparse COO.

    Backward direction: accumulate into source states.
    """
    S = machine.n_states
    silent_mask = (machine.in_tokens == 0) & (machine.out_tokens == 0)

    src = machine.src_states
    dst = machine.dst_states
    w = machine.log_weights

    def body_fn(carry):
        prev, _ = carry
        vals = w + prev[dst]
        vals = jnp.where(silent_mask, vals, NEG_INF)
        update = _scatter_semiring(vals, src, S, semiring)
        new = semiring.plus(cell, update)
        return new, prev

    def cond_fn(carry):
        new, prev = carry
        return jnp.any(jnp.abs(new - prev) > 1e-10)

    init = (cell, jnp.full_like(cell, NEG_INF))
    result, _ = jax.lax.while_loop(cond_fn, body_fn, init)
    return result


def emit_step_forward_sparse(cell: jnp.ndarray, prev: jnp.ndarray,
                              machine: JAXMachine,
                              in_tok: int, out_tok: int,
                              semiring: LogSemiring) -> jnp.ndarray:
    """Apply emitting transitions from prev to cell (Forward), sparse COO.

    Selects transitions matching (in_tok, out_tok) and scatters to dst states.
    """
    S = machine.n_states
    mask = (machine.in_tokens == in_tok) & (machine.out_tokens == out_tok)
    vals = prev[machine.src_states] + machine.log_weights
    vals = jnp.where(mask, vals, NEG_INF)
    update = _scatter_semiring(vals, machine.dst_states, S, semiring)
    return semiring.plus(cell, update)


def emit_step_backward_sparse(cell: jnp.ndarray, future: jnp.ndarray,
                                machine: JAXMachine,
                                in_tok: int, out_tok: int,
                                semiring: LogSemiring) -> jnp.ndarray:
    """Apply emitting transitions from future to cell (Backward), sparse COO.

    Selects transitions matching (in_tok, out_tok) and scatters to src states.
    """
    S = machine.n_states
    mask = (machine.in_tokens == in_tok) & (machine.out_tokens == out_tok)
    vals = machine.log_weights + future[machine.dst_states]
    vals = jnp.where(mask, vals, NEG_INF)
    update = _scatter_semiring(vals, machine.src_states, S, semiring)
    return semiring.plus(cell, update)


def emit_step_forward_sparse_pswm(cell: jnp.ndarray, prev: jnp.ndarray,
                                   machine: JAXMachine,
                                   in_emission: jnp.ndarray | None,
                                   out_emission: jnp.ndarray | None,
                                   emit_in: bool, emit_out: bool,
                                   semiring: LogSemiring) -> jnp.ndarray:
    """Apply emitting transitions weighted by PSWM emission (Forward), sparse COO.

    Args:
        cell: (S,) current cell
        prev: (S,) predecessor cell
        machine: JAXMachine
        in_emission: (n_in_tokens,) emission log-probs, or None
        out_emission: (n_out_tokens,) emission log-probs, or None
        emit_in: whether input token is consumed
        emit_out: whether output token is consumed
        semiring: LogSemiring
    """
    S = machine.n_states

    # Build mask: transitions that consume the right type of tokens
    if emit_in and emit_out:
        mask = (machine.in_tokens > 0) & (machine.out_tokens > 0)
    elif emit_in:
        mask = (machine.in_tokens > 0) & (machine.out_tokens == 0)
    elif emit_out:
        mask = (machine.in_tokens == 0) & (machine.out_tokens > 0)
    else:
        mask = (machine.in_tokens == 0) & (machine.out_tokens == 0)

    # Base values: prev[src] + log_weight
    vals = prev[machine.src_states] + machine.log_weights

    # Add emission weights
    if emit_in and in_emission is not None:
        vals = vals + in_emission[machine.in_tokens]
    if emit_out and out_emission is not None:
        vals = vals + out_emission[machine.out_tokens]

    vals = jnp.where(mask, vals, NEG_INF)
    update = _scatter_semiring(vals, machine.dst_states, S, semiring)
    return semiring.plus(cell, update)


def emit_step_backward_sparse_pswm(cell: jnp.ndarray, future: jnp.ndarray,
                                    machine: JAXMachine,
                                    in_emission: jnp.ndarray | None,
                                    out_emission: jnp.ndarray | None,
                                    emit_in: bool, emit_out: bool,
                                    semiring: LogSemiring) -> jnp.ndarray:
    """Apply emitting transitions weighted by PSWM emission (Backward), sparse COO."""
    S = machine.n_states

    if emit_in and emit_out:
        mask = (machine.in_tokens > 0) & (machine.out_tokens > 0)
    elif emit_in:
        mask = (machine.in_tokens > 0) & (machine.out_tokens == 0)
    elif emit_out:
        mask = (machine.in_tokens == 0) & (machine.out_tokens > 0)
    else:
        mask = (machine.in_tokens == 0) & (machine.out_tokens == 0)

    vals = machine.log_weights + future[machine.dst_states]

    if emit_in and in_emission is not None:
        vals = vals + in_emission[machine.in_tokens]
    if emit_out and out_emission is not None:
        vals = vals + out_emission[machine.out_tokens]

    vals = jnp.where(mask, vals, NEG_INF)
    update = _scatter_semiring(vals, machine.src_states, S, semiring)
    return semiring.plus(cell, update)
