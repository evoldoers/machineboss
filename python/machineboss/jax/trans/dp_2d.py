"""2D DP algorithms for TransMachine.

For transducers with both input and output sequences.
Supports Forward, Backward, and Viterbi via semiring abstraction.

Two strategies:
- simple: outer scan over rows, inner scan over columns
- optimal: anti-diagonal wavefront with vmap
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
    emission_weights_1d, emission_weights_2d,
    to_matrix,
)


# ============================================================
# Simple strategy (nested scans)
# ============================================================

def _forward_2d_simple_matrix(tm, input_seq, output_seq, semiring):
    """2D Forward matrix using nested jax.lax.scan.

    Returns full (Li+1, Lo+1, S) forward matrix.
    """
    S = tm.n_states

    input_seq = wrap_seq(input_seq, tm.n_in)
    output_seq = wrap_seq(output_seq, tm.n_out)

    Li = len(input_seq) if input_seq is not None else 0
    Lo = len(output_seq) if output_seq is not None else 0

    in_emit = input_seq.emission_weights(tm.n_in) if Li > 0 else None
    out_emit = output_seq.emission_weights(tm.n_out) if Lo > 0 else None

    # Row 0: only delete transitions
    init_cell = jnp.full(S, NEG_INF).at[0].set(0.0)
    init_cell = propagate_silent(init_cell, tm, semiring)

    if Lo > 0:
        def scan_row0_del(cell, out_e):
            ew = emission_weights_1d(tm, out_e, False)
            new = emit_step_forward(
                jnp.full(S, NEG_INF), cell, tm, tm.emit_out_mask, ew, semiring)
            new = propagate_silent(new, tm, semiring)
            return new, new
        _, row0_cells = jax.lax.scan(scan_row0_del, init_cell, out_emit)
        row0 = jnp.concatenate([init_cell[None, :], row0_cells], axis=0)
    else:
        row0 = init_cell[None, :]

    if Li == 0:
        return row0[None, :, :]  # (1, Lo+1, S)

    # Outer scan over rows, collecting all rows
    def _process_row(prev_row, in_e):
        # Column 0: insert only
        ew_ins = emission_weights_1d(tm, in_e, True)
        cell_0 = emit_step_forward(
            jnp.full(S, NEG_INF), prev_row[0], tm, tm.emit_in_mask, ew_ins, semiring)
        cell_0 = propagate_silent(cell_0, tm, semiring)

        if Lo == 0:
            new_row = cell_0[None, :]
            return new_row, new_row

        # Inner scan over columns
        def inner_step(cell_left, col_data):
            out_e, prev_match, prev_ins = col_data
            cell = jnp.full(S, NEG_INF)

            # Match: from prev_row[o-1]
            ew_match = emission_weights_2d(tm, in_e, out_e)
            cell = emit_step_forward(
                cell, prev_match, tm, tm.emit_both_mask, ew_match, semiring)

            # Insert: from prev_row[o]
            cell = emit_step_forward(
                cell, prev_ins, tm, tm.emit_in_mask, ew_ins, semiring)

            # Delete: from cell_left
            ew_del = emission_weights_1d(tm, out_e, False)
            cell = emit_step_forward(
                cell, cell_left, tm, tm.emit_out_mask, ew_del, semiring)

            cell = propagate_silent(cell, tm, semiring)
            return cell, cell

        prev_match_cells = prev_row[:Lo]
        prev_ins_cells = prev_row[1:Lo+1]

        _, inner_cells = jax.lax.scan(
            inner_step, cell_0,
            (out_emit, prev_match_cells, prev_ins_cells))

        new_row = jnp.concatenate([cell_0[None, :], inner_cells], axis=0)
        return new_row, new_row

    _, all_rows = jax.lax.scan(_process_row, row0, in_emit)
    # all_rows: (Li, Lo+1, S); prepend row0
    dp = jnp.concatenate([row0[None, :, :], all_rows], axis=0)
    return dp  # (Li+1, Lo+1, S)


def _forward_2d_simple(tm, input_seq, output_seq, semiring):
    """2D Forward scalar using nested jax.lax.scan."""
    dp = _forward_2d_simple_matrix(tm, input_seq, output_seq, semiring)
    return dp[-1, -1, tm.n_states - 1]


def _backward_2d_simple(tm, input_seq, output_seq, semiring):
    """2D Backward using nested jax.lax.scan."""
    S = tm.n_states

    input_seq = wrap_seq(input_seq, tm.n_in)
    output_seq = wrap_seq(output_seq, tm.n_out)

    Li = len(input_seq) if input_seq is not None else 0
    Lo = len(output_seq) if output_seq is not None else 0

    in_emit = input_seq.emission_weights(tm.n_in) if Li > 0 else None
    out_emit = output_seq.emission_weights(tm.n_out) if Lo > 0 else None

    # Terminal
    term = jnp.full(S, NEG_INF).at[S - 1].set(0.0)
    term = propagate_silent_backward(term, tm, semiring)

    # Last row: only delete, right to left
    if Lo > 0:
        def scan_last_del(cell, out_e):
            ew = emission_weights_1d(tm, out_e, False)
            new = emit_step_backward(
                jnp.full(S, NEG_INF), cell, tm, tm.emit_out_mask, ew, semiring)
            new = propagate_silent_backward(new, tm, semiring)
            return new, new
        _, last_rev = jax.lax.scan(scan_last_del, term, out_emit[::-1])
        last_cells = last_rev[::-1]
        last_row = jnp.concatenate([last_cells, term[None, :]], axis=0)
    else:
        last_row = term[None, :]

    if Li == 0:
        return last_row[None, :, :]

    # Backward outer scan
    def _process_row_bwd(next_row, in_e):
        ew_ins = emission_weights_1d(tm, in_e, True)

        # Column Lo: insert only
        cell_Lo = emit_step_backward(
            jnp.full(S, NEG_INF), next_row[Lo], tm, tm.emit_in_mask, ew_ins, semiring)
        cell_Lo = propagate_silent_backward(cell_Lo, tm, semiring)

        if Lo == 0:
            return cell_Lo[None, :], cell_Lo[None, :]

        # Inner scan right-to-left
        def inner_step(cell_right, col_data):
            out_e, next_match, next_ins = col_data
            cell = jnp.full(S, NEG_INF)

            # Match: to (i+1, o+1)
            ew_match = emission_weights_2d(tm, in_e, out_e)
            cell = emit_step_backward(
                cell, next_match, tm, tm.emit_both_mask, ew_match, semiring)

            # Insert: to (i+1, o)
            cell = emit_step_backward(
                cell, next_ins, tm, tm.emit_in_mask, ew_ins, semiring)

            # Delete: to (i, o+1)
            ew_del = emission_weights_1d(tm, out_e, False)
            cell = emit_step_backward(
                cell, cell_right, tm, tm.emit_out_mask, ew_del, semiring)

            cell = propagate_silent_backward(cell, tm, semiring)
            return cell, cell

        next_match_cells = next_row[1:Lo+1][::-1]
        next_ins_cells = next_row[:Lo][::-1]

        _, inner_rev = jax.lax.scan(
            inner_step, cell_Lo,
            (out_emit[::-1], next_match_cells, next_ins_cells))
        inner_cells = inner_rev[::-1]

        new_row = jnp.concatenate([inner_cells, cell_Lo[None, :]], axis=0)
        return new_row, new_row

    _, bp_rows_rev = jax.lax.scan(
        _process_row_bwd, last_row, in_emit[::-1])
    bp_rows = bp_rows_rev[::-1]

    bp = jnp.concatenate([bp_rows, last_row[None, :, :]], axis=0)
    return bp


# ============================================================
# Optimal strategy (anti-diagonal wavefront)
# ============================================================

def _precompute_emit_trans(tm, emit, is_input, semiring):
    """Precompute (L, S, S) transition matrices for 1D emissions."""
    from .kernel import to_matrix
    mask = tm.emit_in_mask if is_input else tm.emit_out_mask

    def make_one(e):
        e = e.at[0].set(NEG_INF)
        ew = emission_weights_1d(tm, e, is_input)
        return to_matrix(tm, mask, ew, semiring)
    return jax.vmap(make_one)(emit)


def _precompute_all_match(tm, in_emit, out_emit, semiring):
    """Precompute (Li, Lo, S, S) match transition matrices."""
    from .kernel import to_matrix

    def match_pair(in_e, out_e):
        in_e = in_e.at[0].set(NEG_INF)
        out_e = out_e.at[0].set(NEG_INF)
        ew = emission_weights_2d(tm, in_e, out_e)
        return to_matrix(tm, tm.emit_both_mask, ew, semiring)

    def match_row(in_e):
        return jax.vmap(lambda out_e: match_pair(in_e, out_e))(out_emit)

    return jax.vmap(match_row)(in_emit)


def _forward_2d_optimal_matrix(tm, input_seq, output_seq, semiring):
    """2D Forward matrix using anti-diagonal wavefront with vmap.

    Returns full (Li+1, Lo+1, S) forward matrix.
    """
    from ..kernel_dense import propagate_silent as propagate_silent_dense
    from ..kernel_dense import emit_step_forward as emit_step_forward_dense

    S = tm.n_states

    input_seq = wrap_seq(input_seq, tm.n_in)
    output_seq = wrap_seq(output_seq, tm.n_out)

    Li = len(input_seq) if input_seq is not None else 0
    Lo = len(output_seq) if output_seq is not None else 0

    in_emit = input_seq.emission_weights(tm.n_in) if Li > 0 else None
    out_emit = output_seq.emission_weights(tm.n_out) if Lo > 0 else None

    # Build silent matrix for dense propagation
    silent_mat = to_matrix(tm, tm.silent_mask, jnp.zeros(tm.n_transitions), semiring)

    dp = jnp.full((Li + 1, Lo + 1, S), NEG_INF)
    dp = dp.at[0, 0, 0].set(0.0)
    dp = dp.at[0, 0].set(propagate_silent_dense(dp[0, 0], silent_mat, semiring))

    if Li + Lo == 0:
        return dp

    # Precompute transition matrices
    all_ins = (_precompute_emit_trans(tm, in_emit, True, semiring)
               if Li > 0 else jnp.full((1, S, S), NEG_INF))
    all_del = (_precompute_emit_trans(tm, out_emit, False, semiring)
               if Lo > 0 else jnp.full((1, S, S), NEG_INF))
    all_match = (_precompute_all_match(tm, in_emit, out_emit, semiring)
                 if Li > 0 and Lo > 0 else jnp.full((1, 1, S, S), NEG_INF))

    D_max = min(Li + 1, Lo + 1)
    max_i_idx = max(Li - 1, 0)
    max_o_idx = max(Lo - 1, 0)

    def scan_fn(dp, d):
        i_min = jnp.maximum(0, d - Lo)
        js = jnp.arange(D_max)
        i_vals = i_min + js
        o_vals = d - i_vals

        valid = (i_vals <= Li) & (o_vals >= 0) & (o_vals <= Lo)
        valid = valid & ~((i_vals == 0) & (o_vals == 0))

        def compute_cell(i, o):
            ip = jnp.clip(i - 1, 0, Li)
            op = jnp.clip(o - 1, 0, Lo)

            cell = jnp.full(S, NEG_INF)

            mt = all_match[jnp.clip(i - 1, 0, max_i_idx),
                           jnp.clip(o - 1, 0, max_o_idx)]
            mc = emit_step_forward_dense(jnp.full(S, NEG_INF), dp[ip, op], mt, semiring)
            cell = jnp.where((i > 0) & (o > 0), semiring.plus(cell, mc), cell)

            it = all_ins[jnp.clip(i - 1, 0, max_i_idx)]
            ic = emit_step_forward_dense(jnp.full(S, NEG_INF),
                                         dp[ip, jnp.clip(o, 0, Lo)], it, semiring)
            cell = jnp.where(i > 0, semiring.plus(cell, ic), cell)

            dt = all_del[jnp.clip(o - 1, 0, max_o_idx)]
            dc = emit_step_forward_dense(jnp.full(S, NEG_INF),
                                         dp[jnp.clip(i, 0, Li), op], dt, semiring)
            cell = jnp.where(o > 0, semiring.plus(cell, dc), cell)

            cell = propagate_silent_dense(cell, silent_mat, semiring)
            return cell

        cells = jax.vmap(compute_cell)(i_vals, o_vals)

        i_write = jnp.where(valid, i_vals, 0)
        o_write = jnp.where(valid, o_vals, 0)
        vals = jnp.where(valid[:, None], cells, dp[i_write, o_write])
        dp = dp.at[i_write, o_write].set(vals)

        return dp, None

    dp, _ = jax.lax.scan(scan_fn, dp, jnp.arange(1, Li + Lo + 1))
    return dp


def _forward_2d_optimal(tm, input_seq, output_seq, semiring):
    """2D Forward scalar using anti-diagonal wavefront."""
    dp = _forward_2d_optimal_matrix(tm, input_seq, output_seq, semiring)
    S = tm.n_states
    return dp[-1, -1, S - 1]


def _backward_2d_optimal(tm, input_seq, output_seq, semiring):
    """2D Backward using anti-diagonal wavefront with vmap."""
    from ..kernel_dense import propagate_silent_backward as prop_silent_bwd_dense
    from ..kernel_dense import emit_step_backward as emit_step_bwd_dense

    S = tm.n_states

    input_seq = wrap_seq(input_seq, tm.n_in)
    output_seq = wrap_seq(output_seq, tm.n_out)

    Li = len(input_seq) if input_seq is not None else 0
    Lo = len(output_seq) if output_seq is not None else 0

    in_emit = input_seq.emission_weights(tm.n_in) if Li > 0 else None
    out_emit = output_seq.emission_weights(tm.n_out) if Lo > 0 else None

    silent_mat = to_matrix(tm, tm.silent_mask, jnp.zeros(tm.n_transitions), semiring)

    bp = jnp.full((Li + 1, Lo + 1, S), NEG_INF)
    bp = bp.at[Li, Lo, S - 1].set(0.0)
    bp = bp.at[Li, Lo].set(prop_silent_bwd_dense(bp[Li, Lo], silent_mat, semiring))

    if Li + Lo == 0:
        return bp

    all_ins = (_precompute_emit_trans(tm, in_emit, True, semiring)
               if Li > 0 else jnp.full((1, S, S), NEG_INF))
    all_del = (_precompute_emit_trans(tm, out_emit, False, semiring)
               if Lo > 0 else jnp.full((1, S, S), NEG_INF))
    all_match = (_precompute_all_match(tm, in_emit, out_emit, semiring)
                 if Li > 0 and Lo > 0 else jnp.full((1, 1, S, S), NEG_INF))

    D_max = min(Li + 1, Lo + 1)
    max_i_idx = max(Li - 1, 0)
    max_o_idx = max(Lo - 1, 0)

    def scan_fn(bp, d):
        i_min = jnp.maximum(0, d - Lo)
        js = jnp.arange(D_max)
        i_vals = i_min + js
        o_vals = d - i_vals

        valid = (i_vals <= Li) & (o_vals >= 0) & (o_vals <= Lo)
        valid = valid & ~((i_vals == Li) & (o_vals == Lo))

        def compute_cell(i, o):
            i_next = jnp.clip(i + 1, 0, Li)
            o_next = jnp.clip(o + 1, 0, Lo)

            cell = jnp.full(S, NEG_INF)

            mt = all_match[jnp.clip(i, 0, max_i_idx),
                           jnp.clip(o, 0, max_o_idx)]
            mc = emit_step_bwd_dense(jnp.full(S, NEG_INF),
                                     bp[i_next, o_next], mt, semiring)
            cell = jnp.where((i < Li) & (o < Lo),
                             semiring.plus(cell, mc), cell)

            it = all_ins[jnp.clip(i, 0, max_i_idx)]
            ic = emit_step_bwd_dense(jnp.full(S, NEG_INF),
                                     bp[i_next, jnp.clip(o, 0, Lo)], it, semiring)
            cell = jnp.where(i < Li, semiring.plus(cell, ic), cell)

            dt = all_del[jnp.clip(o, 0, max_o_idx)]
            dc = emit_step_bwd_dense(jnp.full(S, NEG_INF),
                                     bp[jnp.clip(i, 0, Li), o_next], dt, semiring)
            cell = jnp.where(o < Lo, semiring.plus(cell, dc), cell)

            cell = prop_silent_bwd_dense(cell, silent_mat, semiring)
            return cell

        cells = jax.vmap(compute_cell)(i_vals, o_vals)

        i_write = jnp.where(valid, i_vals, Li)
        o_write = jnp.where(valid, o_vals, Lo)
        vals = jnp.where(valid[:, None], cells, bp[i_write, o_write])
        bp = bp.at[i_write, o_write].set(vals)

        return bp, None

    bp, _ = jax.lax.scan(scan_fn, bp, jnp.arange(Li + Lo)[::-1])
    return bp


# ============================================================
# Public API
# ============================================================

def forward_2d_matrix(tm: TransMachine, input_seq, output_seq,
                      semiring: LogSemiring = LOGSUMEXP, *,
                      strategy: str = 'auto') -> jnp.ndarray:
    """2D Forward algorithm returning full (Li+1, Lo+1, S) matrix.

    Args:
        tm: TransMachine
        input_seq, output_seq: token sequences
        semiring: LOGSUMEXP (default) or MAXPLUS
        strategy: 'simple', 'optimal', or 'auto'
    """
    if strategy == 'optimal':
        return _forward_2d_optimal_matrix(tm, input_seq, output_seq, semiring)
    else:
        return _forward_2d_simple_matrix(tm, input_seq, output_seq, semiring)


def forward_2d(tm: TransMachine, input_seq, output_seq,
               semiring: LogSemiring = LOGSUMEXP, *,
               strategy: str = 'auto') -> float:
    """2D Forward algorithm returning scalar log-likelihood.

    Args:
        tm: TransMachine
        input_seq, output_seq: token sequences
        semiring: LOGSUMEXP (default) or MAXPLUS
        strategy: 'simple', 'optimal', or 'auto'
    """
    dp = forward_2d_matrix(tm, input_seq, output_seq, semiring, strategy=strategy)
    return dp[-1, -1, tm.n_states - 1]


def backward_2d(tm: TransMachine, input_seq, output_seq,
                semiring: LogSemiring = LOGSUMEXP, *,
                strategy: str = 'auto') -> jnp.ndarray:
    """2D Backward algorithm. Returns (Li+1, Lo+1, S) matrix."""
    if strategy == 'simple':
        return _backward_2d_simple(tm, input_seq, output_seq, semiring)
    elif strategy == 'optimal':
        return _backward_2d_optimal(tm, input_seq, output_seq, semiring)
    else:
        return _backward_2d_simple(tm, input_seq, output_seq, semiring)


def viterbi_2d(tm: TransMachine, input_seq, output_seq, *,
               strategy: str = 'auto') -> float:
    """2D Viterbi algorithm."""
    return forward_2d(tm, input_seq, output_seq, MAXPLUS, strategy=strategy)
