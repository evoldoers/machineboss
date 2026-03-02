"""1D DP engine using jax.lax.scan (SIMPLE strategy).

For generators (no input, output only) or recognizers (input only, no output).
Supports Forward, Backward, and Viterbi via semiring abstraction.
Supports both dense and sparse kernels.
Supports both TokenSeq and PSWMSeq via emission_weights().
JIT-compilable: all loops use jax.lax.scan.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import JAXMachine, NEG_INF
from .semiring import LogSemiring
from .seq import TokenSeq, PSWMSeq, wrap_seq
from .kernel_dense import (
    propagate_silent, propagate_silent_backward,
    emit_step_forward, emit_step_backward,
)


def _get_seq_and_trans(machine, input_seq, output_seq):
    """Determine which sequence is active and get the appropriate transition slice.

    For 1D DP, exactly one of input_seq/output_seq is None.
    Returns (seq, is_input) where is_input indicates which dimension is active.
    """
    if input_seq is None:
        seq = wrap_seq(output_seq, machine.n_output_tokens)
        return seq, False
    else:
        seq = wrap_seq(input_seq, machine.n_input_tokens)
        return seq, True


def _precompute_trans_1d(emit, trans_slice, semiring):
    """Precompute weighted transition matrices for all positions.

    Unified approach for both TokenSeq (one-hot) and PSWMSeq (soft weights).

    Args:
        emit: (L, n_tokens) emission weights
        trans_slice: (n_tokens, S, S) transition matrices for active dimension
        semiring: LogSemiring
    Returns:
        (L, S, S) transition matrices for each position.
    """
    def make_one(e):
        e = e.at[0].set(NEG_INF)  # exclude empty token
        return semiring.reduce(e[:, None, None] + trans_slice, axis=0)
    return jax.vmap(make_one)(emit)


# --- Forward ---

def forward_1d_simple(machine: JAXMachine, input_seq, output_seq,
                      semiring: LogSemiring, *, kernel: str = 'dense',
                      length: int | None = None) -> float:
    """1D Forward/Viterbi using jax.lax.scan.

    Args:
        machine: JAXMachine
        input_seq: token sequence or None (generator case)
        output_seq: token sequence or None (recognizer case)
        semiring: LOGSUMEXP for Forward, MAXPLUS for Viterbi
        kernel: 'dense' or 'sparse'
        length: real sequence length (for padded sequences). If None, uses len(seq).
    Returns:
        Log-likelihood or Viterbi score (scalar).
    """
    if kernel == 'sparse':
        return _forward_1d_sparse(machine, input_seq, output_seq, semiring,
                                  length=length)
    return _forward_1d_dense(machine, input_seq, output_seq, semiring,
                             length=length)


def _forward_1d_dense(machine, input_seq, output_seq, semiring, *, length=None):
    """1D Forward using dense tensor and jax.lax.scan."""
    assert machine.log_trans is not None
    S = machine.n_states
    log_trans = machine.log_trans

    seq, is_input = _get_seq_and_trans(machine, input_seq, output_seq)
    L = len(seq)
    n_tokens = machine.n_input_tokens if is_input else machine.n_output_tokens

    # Handle empty sequence
    if L == 0:
        cell = jnp.full(S, NEG_INF).at[0].set(0.0)
        cell = propagate_silent(cell, log_trans[0, 0], semiring)
        return cell[S - 1]

    emit = seq.emission_weights(n_tokens)  # (L, n_tokens)
    silent = log_trans[0, 0]  # (S, S)

    # Precompute per-position transition matrices (unified for TOK and PSWM)
    trans_slice = log_trans[:, 0, :, :] if is_input else log_trans[0, :, :, :]
    trans_all = _precompute_trans_1d(emit, trans_slice, semiring)  # (L, S, S)

    # Validity mask for padding
    if length is not None:
        mask = jnp.arange(L) < length
    else:
        mask = jnp.ones(L, dtype=bool)

    # Init
    cell = jnp.full(S, NEG_INF).at[0].set(0.0)
    cell = propagate_silent(cell, silent, semiring)

    # Scan
    def scan_fn(cell, inputs):
        trans, valid = inputs
        new_cell = emit_step_forward(jnp.full(S, NEG_INF), cell, trans, semiring)
        new_cell = propagate_silent(new_cell, silent, semiring)
        return jnp.where(valid, new_cell, cell), None

    final_cell, _ = jax.lax.scan(scan_fn, cell, (trans_all, mask))
    return final_cell[S - 1]


# --- Backward ---

def backward_1d_simple(machine: JAXMachine, input_seq, output_seq,
                       semiring: LogSemiring, *, kernel: str = 'dense',
                       length: int | None = None) -> jnp.ndarray:
    """1D Backward using jax.lax.scan.

    Args:
        machine: JAXMachine
        input_seq: token sequence or None
        output_seq: token sequence or None
        semiring: LOGSUMEXP or MAXPLUS
        kernel: 'dense' or 'sparse'
        length: real sequence length (for padded sequences). If None, uses len(seq).
    Returns:
        Backward vector array of shape (L+1, S) where L is the sequence length
        (padded length if padding is used).
    """
    if kernel == 'sparse':
        return _backward_1d_sparse(machine, input_seq, output_seq, semiring,
                                   length=length)
    return _backward_1d_dense(machine, input_seq, output_seq, semiring,
                              length=length)


def _backward_1d_dense(machine, input_seq, output_seq, semiring, *, length=None):
    """1D Backward using dense tensor and jax.lax.scan."""
    assert machine.log_trans is not None
    S = machine.n_states
    log_trans = machine.log_trans

    seq, is_input = _get_seq_and_trans(machine, input_seq, output_seq)
    L = len(seq)
    n_tokens = machine.n_input_tokens if is_input else machine.n_output_tokens

    emit = seq.emission_weights(n_tokens)
    silent = log_trans[0, 0]

    trans_slice = log_trans[:, 0, :, :] if is_input else log_trans[0, :, :, :]
    trans_all = _precompute_trans_1d(emit, trans_slice, semiring)  # (L, S, S)

    if length is not None:
        mask = jnp.arange(L) < length
    else:
        mask = jnp.ones(L, dtype=bool)

    # Terminal cell
    term = jnp.full(S, NEG_INF).at[S - 1].set(0.0)
    term = propagate_silent_backward(term, silent, semiring)

    if L == 0:
        return term[None, :]  # (1, S)

    # Backward scan: process positions in reverse
    def scan_fn(cell, inputs):
        trans, valid = inputs
        new_cell = emit_step_backward(jnp.full(S, NEG_INF), cell, trans, semiring)
        new_cell = propagate_silent_backward(new_cell, silent, semiring)
        result = jnp.where(valid, new_cell, cell)
        return result, result

    # Reverse inputs for backward scan
    final_cell, bp_rev = jax.lax.scan(scan_fn, term, (trans_all[::-1], mask[::-1]))

    # bp_rev[k] = bp[L-1-k], unreverse to get bp[p] for p = 0..L-1
    bp_cells = bp_rev[::-1]  # (L, S)

    # Build (L+1, S): bp[0..L-1] from scan, bp[L] = term
    bp = jnp.concatenate([bp_cells, term[None, :]], axis=0)
    return bp


# --- Sparse Forward ---

def _forward_1d_sparse(machine, input_seq, output_seq, semiring, *, length=None):
    """1D Forward using sparse COO transitions and jax.lax.scan."""
    from .kernel_sparse import (
        propagate_silent_sparse, emit_step_forward_sparse_pswm,
    )

    S = machine.n_states
    seq, is_input = _get_seq_and_trans(machine, input_seq, output_seq)
    L = len(seq)
    n_tokens = machine.n_input_tokens if is_input else machine.n_output_tokens

    if L == 0:
        cell = jnp.full(S, NEG_INF).at[0].set(0.0)
        cell = propagate_silent_sparse(cell, machine, semiring)
        return cell[S - 1]

    # Always use emission weights (unified for TOK and PSWM)
    emit = seq.emission_weights(n_tokens)  # (L, n_tokens)

    if length is not None:
        mask = jnp.arange(L) < length
    else:
        mask = jnp.ones(L, dtype=bool)

    cell = jnp.full(S, NEG_INF).at[0].set(0.0)
    cell = propagate_silent_sparse(cell, machine, semiring)

    if is_input:
        def scan_fn(cell, inputs):
            e, valid = inputs
            new_cell = emit_step_forward_sparse_pswm(
                jnp.full(S, NEG_INF), cell, machine,
                e, None, emit_in=True, emit_out=False, semiring=semiring)
            new_cell = propagate_silent_sparse(new_cell, machine, semiring)
            return jnp.where(valid, new_cell, cell), None
    else:
        def scan_fn(cell, inputs):
            e, valid = inputs
            new_cell = emit_step_forward_sparse_pswm(
                jnp.full(S, NEG_INF), cell, machine,
                None, e, emit_in=False, emit_out=True, semiring=semiring)
            new_cell = propagate_silent_sparse(new_cell, machine, semiring)
            return jnp.where(valid, new_cell, cell), None

    final_cell, _ = jax.lax.scan(scan_fn, cell, (emit, mask))
    return final_cell[S - 1]


# --- Sparse Backward ---

def _backward_1d_sparse(machine, input_seq, output_seq, semiring, *, length=None):
    """1D Backward using sparse COO transitions and jax.lax.scan."""
    from .kernel_sparse import (
        propagate_silent_backward_sparse, emit_step_backward_sparse_pswm,
    )

    S = machine.n_states
    seq, is_input = _get_seq_and_trans(machine, input_seq, output_seq)
    L = len(seq)
    n_tokens = machine.n_input_tokens if is_input else machine.n_output_tokens

    emit = seq.emission_weights(n_tokens)

    if length is not None:
        mask = jnp.arange(L) < length
    else:
        mask = jnp.ones(L, dtype=bool)

    term = jnp.full(S, NEG_INF).at[S - 1].set(0.0)
    term = propagate_silent_backward_sparse(term, machine, semiring)

    if L == 0:
        return term[None, :]

    if is_input:
        def scan_fn(cell, inputs):
            e, valid = inputs
            new_cell = emit_step_backward_sparse_pswm(
                jnp.full(S, NEG_INF), cell, machine,
                e, None, emit_in=True, emit_out=False, semiring=semiring)
            new_cell = propagate_silent_backward_sparse(new_cell, machine, semiring)
            result = jnp.where(valid, new_cell, cell)
            return result, result
    else:
        def scan_fn(cell, inputs):
            e, valid = inputs
            new_cell = emit_step_backward_sparse_pswm(
                jnp.full(S, NEG_INF), cell, machine,
                None, e, emit_in=False, emit_out=True, semiring=semiring)
            new_cell = propagate_silent_backward_sparse(new_cell, machine, semiring)
            result = jnp.where(valid, new_cell, cell)
            return result, result

    final_cell, bp_rev = jax.lax.scan(scan_fn, term, (emit[::-1], mask[::-1]))
    bp_cells = bp_rev[::-1]
    bp = jnp.concatenate([bp_cells, term[None, :]], axis=0)
    return bp
