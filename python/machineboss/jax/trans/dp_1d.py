"""1D DP algorithms for TransMachine.

For generators (no input, output only) or recognizers (input only, no output).
Supports Forward, Backward, and Viterbi via semiring abstraction.

Two strategies:
- simple: sequential jax.lax.scan, O(L) depth
- optimal: parallel jax.lax.associative_scan, O(log L) depth
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..types import NEG_INF
from ..semiring import LogSemiring, LOGSUMEXP, MAXPLUS
from ..seq import TokenSeq, PSWMSeq, wrap_seq
from .machine import TransMachine
from .kernel import (
    propagate_silent, propagate_silent_backward,
    emit_step_forward, emit_step_backward,
    emission_weights_1d, to_matrix,
)


def _get_seq_info(tm, input_seq, output_seq):
    """Determine active sequence and dimension."""
    if input_seq is None:
        seq = wrap_seq(output_seq, tm.n_out)
        return seq, False
    else:
        seq = wrap_seq(input_seq, tm.n_in)
        return seq, True


# ============================================================
# Simple strategy (sequential scan)
# ============================================================

def _forward_1d_simple(tm, input_seq, output_seq, semiring, *, length=None):
    """1D Forward using jax.lax.scan."""
    S = tm.n_states
    seq, is_input = _get_seq_info(tm, input_seq, output_seq)
    L = len(seq)
    n_tokens = tm.n_in if is_input else tm.n_out
    mask = tm.emit_in_mask if is_input else tm.emit_out_mask

    if L == 0:
        cell = jnp.full(S, NEG_INF).at[0].set(0.0)
        cell = propagate_silent(cell, tm, semiring)
        return cell[S - 1]

    emit = seq.emission_weights(n_tokens)  # (L, n_tokens)

    if length is not None:
        valid = jnp.arange(L) < length
    else:
        valid = jnp.ones(L, dtype=bool)

    cell = jnp.full(S, NEG_INF).at[0].set(0.0)
    cell = propagate_silent(cell, tm, semiring)

    def scan_fn(cell, inputs):
        e, v = inputs
        ew = emission_weights_1d(tm, e, is_input)
        new_cell = emit_step_forward(
            jnp.full(S, NEG_INF), cell, tm, mask, ew, semiring)
        new_cell = propagate_silent(new_cell, tm, semiring)
        return jnp.where(v, new_cell, cell), None

    final_cell, _ = jax.lax.scan(scan_fn, cell, (emit, valid))
    return final_cell[S - 1]


def _backward_1d_simple(tm, input_seq, output_seq, semiring, *, length=None):
    """1D Backward using jax.lax.scan."""
    S = tm.n_states
    seq, is_input = _get_seq_info(tm, input_seq, output_seq)
    L = len(seq)
    n_tokens = tm.n_in if is_input else tm.n_out
    mask = tm.emit_in_mask if is_input else tm.emit_out_mask

    emit = seq.emission_weights(n_tokens) if L > 0 else None

    if length is not None:
        valid = jnp.arange(L) < length
    else:
        valid = jnp.ones(L, dtype=bool) if L > 0 else jnp.array([], dtype=bool)

    term = jnp.full(S, NEG_INF).at[S - 1].set(0.0)
    term = propagate_silent_backward(term, tm, semiring)

    if L == 0:
        return term[None, :]

    def scan_fn(cell, inputs):
        e, v = inputs
        ew = emission_weights_1d(tm, e, is_input)
        new_cell = emit_step_backward(
            jnp.full(S, NEG_INF), cell, tm, mask, ew, semiring)
        new_cell = propagate_silent_backward(new_cell, tm, semiring)
        result = jnp.where(v, new_cell, cell)
        return result, result

    final_cell, bp_rev = jax.lax.scan(scan_fn, term, (emit[::-1], valid[::-1]))
    bp_cells = bp_rev[::-1]
    bp = jnp.concatenate([bp_cells, term[None, :]], axis=0)
    return bp


# ============================================================
# Optimal strategy (parallel prefix scan)
# ============================================================

def _log_identity(S):
    """Identity matrix in log-space."""
    identity = jnp.full((S, S), NEG_INF)
    return identity.at[jnp.arange(S), jnp.arange(S)].set(0.0)


def _silent_closure_matrix(tm, semiring):
    """Compute Kleene star of silent transitions as (S,S) matrix."""
    S = tm.n_states
    identity = _log_identity(S)
    silent_mat = to_matrix(tm, tm.silent_mask, jnp.zeros(tm.n_transitions), semiring)

    def body_fn(carry):
        prev_result, power, _ = carry
        next_power = semiring.mat_mul(power, silent_mat)
        new_result = semiring.plus(prev_result, next_power)
        return new_result, next_power, prev_result

    def cond_fn(carry):
        new_result, _, prev_result = carry
        return jnp.any(jnp.abs(new_result - prev_result) > 1e-10)

    init = (identity, identity, jnp.full_like(identity, NEG_INF))
    result, _, _ = jax.lax.while_loop(cond_fn, body_fn, init)
    return result


def _build_emit_matrix_1d(tm, emission, is_input, semiring):
    """Build (S,S) emission transition matrix for one position."""
    S = tm.n_states
    mask = tm.emit_in_mask if is_input else tm.emit_out_mask
    ew = emission_weights_1d(tm, emission, is_input)
    return to_matrix(tm, mask, ew, semiring)


def _forward_1d_optimal(tm, input_seq, output_seq, semiring, *, length=None):
    """1D Forward using associative_scan for O(log L) depth."""
    S = tm.n_states
    seq, is_input = _get_seq_info(tm, input_seq, output_seq)
    L = len(seq)
    n_tokens = tm.n_in if is_input else tm.n_out
    real_L = length if length is not None else L

    closure = _silent_closure_matrix(tm, semiring)

    if L == 0:
        return closure[0, S - 1]

    emit = seq.emission_weights(n_tokens)
    identity = _log_identity(S)
    is_pswm = isinstance(seq, PSWMSeq)

    def make_transfer(emission_p):
        if is_pswm:
            e = emission_p.at[0].set(NEG_INF)
        else:
            e = emission_p
        emit_mat = _build_emit_matrix_1d(tm, e, is_input, semiring)
        return semiring.mat_mul(closure, semiring.mat_mul(emit_mat, closure))

    transfers = jax.vmap(make_transfer)(emit)

    if length is not None and length < L:
        mask = jnp.arange(L) < length
        transfers = jnp.where(mask[:, None, None], transfers, identity[None, :, :])

    prefix = jax.lax.associative_scan(semiring.mat_mul, transfers, axis=0)

    init = jnp.full(S, NEG_INF).at[0].set(0.0)
    init_closed = semiring.reduce(init[:, None] + closure, axis=0)
    final = semiring.reduce(init_closed[:, None] + prefix[real_L - 1], axis=0)
    return final[S - 1]


def _backward_1d_optimal(tm, input_seq, output_seq, semiring, *, length=None):
    """1D Backward using associative_scan with reverse=True."""
    S = tm.n_states
    seq, is_input = _get_seq_info(tm, input_seq, output_seq)
    L = len(seq)
    n_tokens = tm.n_in if is_input else tm.n_out
    real_L = length if length is not None else L

    closure = _silent_closure_matrix(tm, semiring)
    is_pswm = isinstance(seq, PSWMSeq)

    if L == 0:
        bp = jnp.full((1, S), NEG_INF)
        term = jnp.full(S, NEG_INF).at[S - 1].set(0.0)
        term_closed = semiring.reduce(closure + term[None, :], axis=1)
        bp = bp.at[0].set(term_closed)
        return bp

    emit = seq.emission_weights(n_tokens)
    identity = _log_identity(S)

    def make_transfer(emission_p):
        if is_pswm:
            e = emission_p.at[0].set(NEG_INF)
        else:
            e = emission_p
        emit_mat = _build_emit_matrix_1d(tm, e, is_input, semiring)
        return semiring.mat_mul(closure, semiring.mat_mul(emit_mat, closure))

    transfers = jax.vmap(make_transfer)(emit)

    if length is not None and length < L:
        mask = jnp.arange(L) < length
        transfers = jnp.where(mask[:, None, None], transfers, identity[None, :, :])

    suffix = jax.lax.associative_scan(
        lambda a, b: semiring.mat_mul(b, a),
        transfers, axis=0, reverse=True)

    term = jnp.full(S, NEG_INF).at[S - 1].set(0.0)
    term_closed = semiring.reduce(closure + term[None, :], axis=1)

    bp_emit = semiring.reduce(suffix[:real_L] + term_closed[None, None, :], axis=2)

    bp = jnp.full((real_L + 1, S), NEG_INF)
    bp = bp.at[:real_L].set(bp_emit)
    bp = bp.at[real_L].set(term_closed)
    return bp


# ============================================================
# Public API
# ============================================================

def _auto_pad_1d(tm, input_seq, output_seq, length):
    """Auto-pad the active 1D sequence for JIT compilation cache efficiency.

    When length is not explicitly provided, pads the active sequence to a
    geometric-series bucket so JAX reuses compiled kernels.

    Returns (input_seq, output_seq, length).
    """
    from ..seq import pad_length, pad_token_seq, pad_pswm_seq

    if length is not None:
        return input_seq, output_seq, length

    if input_seq is None:
        seq = wrap_seq(output_seq, tm.n_out)
        L = len(seq)
        padded_L = pad_length(L)
        if padded_L > L:
            if isinstance(seq, PSWMSeq):
                seq, orig_L = pad_pswm_seq(seq, padded_L)
            else:
                seq, orig_L = pad_token_seq(seq, padded_L)
            return None, seq, orig_L
        return None, seq, None
    else:
        seq = wrap_seq(input_seq, tm.n_in)
        L = len(seq)
        padded_L = pad_length(L)
        if padded_L > L:
            if isinstance(seq, PSWMSeq):
                seq, orig_L = pad_pswm_seq(seq, padded_L)
            else:
                seq, orig_L = pad_token_seq(seq, padded_L)
            return seq, None, orig_L
        return seq, None, None


def forward_1d(tm: TransMachine, input_seq=None, output_seq=None,
               semiring: LogSemiring = LOGSUMEXP, *,
               strategy: str = 'auto', length: int | None = None,
               auto_pad: bool = True) -> float:
    """1D Forward algorithm.

    Args:
        tm: TransMachine
        input_seq: token sequence or None (generator)
        output_seq: token sequence or None (recognizer)
        semiring: LOGSUMEXP (default) or MAXPLUS
        strategy: 'simple', 'optimal', or 'auto'
        length: real sequence length for padded sequences
        auto_pad: pad to geometric bucket for JIT cache reuse (default True)
    """
    if auto_pad:
        input_seq, output_seq, length = _auto_pad_1d(
            tm, input_seq, output_seq, length)
    if strategy == 'simple':
        return _forward_1d_simple(tm, input_seq, output_seq, semiring, length=length)
    elif strategy == 'optimal':
        return _forward_1d_optimal(tm, input_seq, output_seq, semiring, length=length)
    else:
        return _forward_1d_optimal(tm, input_seq, output_seq, semiring, length=length)


def backward_1d(tm: TransMachine, input_seq=None, output_seq=None,
                semiring: LogSemiring = LOGSUMEXP, *,
                strategy: str = 'auto', length: int | None = None,
                auto_pad: bool = True) -> jnp.ndarray:
    """1D Backward algorithm. Returns (L+1, S) matrix."""
    if auto_pad:
        input_seq, output_seq, length = _auto_pad_1d(
            tm, input_seq, output_seq, length)
    if strategy == 'simple':
        return _backward_1d_simple(tm, input_seq, output_seq, semiring, length=length)
    elif strategy == 'optimal':
        return _backward_1d_optimal(tm, input_seq, output_seq, semiring, length=length)
    else:
        return _backward_1d_optimal(tm, input_seq, output_seq, semiring, length=length)


def viterbi_1d(tm: TransMachine, input_seq=None, output_seq=None, *,
               strategy: str = 'auto', length: int | None = None,
               auto_pad: bool = True) -> float:
    """1D Viterbi algorithm."""
    return forward_1d(tm, input_seq, output_seq, MAXPLUS,
                      strategy=strategy, length=length, auto_pad=auto_pad)
