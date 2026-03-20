"""Position-dependent parameterized 2D DP for TransMachine.

Uses ParameterizedTransMachine to build log_w vectors at each (i, j) position.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..types import NEG_INF
from ..semiring import LogSemiring, LOGSUMEXP, MAXPLUS
from .machine import TransMachine
from .parameterized import ParameterizedTransMachine
from .kernel import (
    propagate_silent, propagate_silent_backward,
    emit_step_forward, emit_step_backward,
    emission_weights_1d, emission_weights_2d,
)


def _propagate_silent_fixed(cell, tm, semiring, n_iter):
    """Fixed-iteration silent propagation (differentiable)."""
    from .kernel import _scatter_semiring

    S = tm.n_states

    def body(_, prev):
        vals = prev[tm.src] + tm.log_w
        vals = jnp.where(tm.silent_mask, vals, NEG_INF)
        update = _scatter_semiring(vals, tm.dst, S, semiring)
        return semiring.plus(cell, update)

    return jax.lax.fori_loop(0, n_iter, body, cell)


def _propagate_silent_backward_fixed(cell, tm, semiring, n_iter):
    """Backward fixed-iteration silent propagation (differentiable)."""
    from .kernel import _scatter_semiring

    S = tm.n_states

    def body(_, prev):
        vals = tm.log_w + prev[tm.dst]
        vals = jnp.where(tm.silent_mask, vals, NEG_INF)
        update = _scatter_semiring(vals, tm.src, S, semiring)
        return semiring.plus(cell, update)

    return jax.lax.fori_loop(0, n_iter, body, cell)


def _build_tm_at(ptm, params, i, j):
    """Build TransMachine at position (i, j)."""
    return ptm.build_trans_machine(params, i, j)


def neural_forward_2d(ptm: ParameterizedTransMachine,
                      input_pswm: jnp.ndarray,
                      output_pswm: jnp.ndarray,
                      params: dict[str, jnp.ndarray],
                      semiring: LogSemiring) -> float:
    """2D Forward/Viterbi with position-dependent transition weights.

    Args:
        ptm: ParameterizedTransMachine
        input_pswm: (Li, n_in) input emission log-probs
        output_pswm: (Lo, n_out) output emission log-probs
        params: dict mapping param names to broadcastable arrays
        semiring: LOGSUMEXP or MAXPLUS
    """
    S = ptm.n_states
    Li = input_pswm.shape[0]
    Lo = output_pswm.shape[0]

    # Cell (0, 0)
    tm_00 = _build_tm_at(ptm, params, 0, 0)
    cell_00 = jnp.full(S, NEG_INF).at[0].set(0.0)
    cell_00 = _propagate_silent_fixed(cell_00, tm_00, semiring, S)

    # Row 0: deletes
    def scan_row0(cell_left, j_idx):
        j = j_idx + 1
        tm = _build_tm_at(ptm, params, 0, j)
        out_e = output_pswm[j_idx]
        ew = emission_weights_1d(tm, out_e, False)
        cell = emit_step_forward(
            jnp.full(S, NEG_INF), cell_left, tm, tm.emit_out_mask, ew, semiring)
        cell = _propagate_silent_fixed(cell, tm, semiring, S)
        return cell, cell

    if Lo > 0:
        _, row0_cells = jax.lax.scan(scan_row0, cell_00, jnp.arange(Lo))
        row0 = jnp.concatenate([cell_00[None, :], row0_cells], axis=0)
    else:
        row0 = cell_00[None, :]

    if Li == 0:
        return row0[Lo, S - 1]

    # Rows 1..Li
    def process_row(prev_row, i_idx):
        i = i_idx + 1
        in_e = input_pswm[i_idx]

        # Column 0: insert
        tm_i0 = _build_tm_at(ptm, params, i, 0)
        ew_ins = emission_weights_1d(tm_i0, in_e, True)
        cell_i0 = emit_step_forward(
            jnp.full(S, NEG_INF), prev_row[0], tm_i0, tm_i0.emit_in_mask, ew_ins, semiring)
        cell_i0 = _propagate_silent_fixed(cell_i0, tm_i0, semiring, S)

        if Lo == 0:
            return cell_i0[None, :], None

        # Inner scan
        def inner_step(cell_left, j_idx):
            j = j_idx + 1
            out_e = output_pswm[j_idx]
            tm = _build_tm_at(ptm, params, i, j)

            cell = jnp.full(S, NEG_INF)

            # Match
            ew_match = emission_weights_2d(tm, in_e, out_e)
            cell = emit_step_forward(
                cell, prev_row[j_idx], tm, tm.emit_both_mask, ew_match, semiring)

            # Insert
            ew_ins_inner = emission_weights_1d(tm, in_e, True)
            cell = emit_step_forward(
                cell, prev_row[j], tm, tm.emit_in_mask, ew_ins_inner, semiring)

            # Delete
            ew_del = emission_weights_1d(tm, out_e, False)
            cell = emit_step_forward(
                cell, cell_left, tm, tm.emit_out_mask, ew_del, semiring)

            cell = _propagate_silent_fixed(cell, tm, semiring, S)
            return cell, cell

        _, inner_cells = jax.lax.scan(inner_step, cell_i0, jnp.arange(Lo))
        new_row = jnp.concatenate([cell_i0[None, :], inner_cells], axis=0)
        return new_row, None

    final_row, _ = jax.lax.scan(process_row, row0, jnp.arange(Li))
    return final_row[Lo, S - 1]


def neural_viterbi_2d(ptm, input_pswm, output_pswm, params) -> float:
    """Viterbi with position-dependent parameters."""
    return neural_forward_2d(ptm, input_pswm, output_pswm, params, MAXPLUS)


def neural_backward_2d(ptm: ParameterizedTransMachine,
                       input_pswm: jnp.ndarray,
                       output_pswm: jnp.ndarray,
                       params: dict[str, jnp.ndarray],
                       semiring: LogSemiring) -> jnp.ndarray:
    """2D Backward with position-dependent weights. Returns (Li+1, Lo+1, S)."""
    S = ptm.n_states
    Li = input_pswm.shape[0]
    Lo = output_pswm.shape[0]

    # Terminal cell
    tm_term = _build_tm_at(ptm, params, Li, Lo)
    term = jnp.full(S, NEG_INF).at[S - 1].set(0.0)
    term = _propagate_silent_backward_fixed(term, tm_term, semiring, S)

    # Last row: deletes right to left
    def scan_last_del(cell_right, j_idx):
        j = Lo - 1 - j_idx
        tm = _build_tm_at(ptm, params, Li, j)
        out_e = output_pswm[j]
        ew = emission_weights_1d(tm, out_e, False)
        cell = emit_step_backward(
            jnp.full(S, NEG_INF), cell_right, tm, tm.emit_out_mask, ew, semiring)
        cell = _propagate_silent_backward_fixed(cell, tm, semiring, S)
        return cell, cell

    if Lo > 0:
        _, last_rev = jax.lax.scan(scan_last_del, term, jnp.arange(Lo))
        last_cells = last_rev[::-1]
        last_row = jnp.concatenate([last_cells, term[None, :]], axis=0)
    else:
        last_row = term[None, :]

    if Li == 0:
        return last_row[None, :, :]

    # Rows Li-1..0
    def process_row_bwd(next_row, i_idx):
        i = Li - 1 - i_idx
        in_e = input_pswm[i]

        # Column Lo: insert
        tm_lo = _build_tm_at(ptm, params, i, Lo)
        ew_ins = emission_weights_1d(tm_lo, in_e, True)
        cell_Lo = emit_step_backward(
            jnp.full(S, NEG_INF), next_row[Lo], tm_lo, tm_lo.emit_in_mask, ew_ins, semiring)
        cell_Lo = _propagate_silent_backward_fixed(cell_Lo, tm_lo, semiring, S)

        if Lo == 0:
            return cell_Lo[None, :], cell_Lo[None, :]

        # Inner scan right to left
        def inner_step_bwd(cell_right, j_idx):
            j = Lo - 1 - j_idx
            out_e = output_pswm[j]
            tm = _build_tm_at(ptm, params, i, j)

            cell = jnp.full(S, NEG_INF)

            # Match
            ew_match = emission_weights_2d(tm, in_e, out_e)
            cell = emit_step_backward(
                cell, next_row[j + 1], tm, tm.emit_both_mask, ew_match, semiring)

            # Insert
            ew_ins_inner = emission_weights_1d(tm, in_e, True)
            cell = emit_step_backward(
                cell, next_row[j], tm, tm.emit_in_mask, ew_ins_inner, semiring)

            # Delete
            ew_del = emission_weights_1d(tm, out_e, False)
            cell = emit_step_backward(
                cell, cell_right, tm, tm.emit_out_mask, ew_del, semiring)

            cell = _propagate_silent_backward_fixed(cell, tm, semiring, S)
            return cell, cell

        _, inner_rev = jax.lax.scan(inner_step_bwd, cell_Lo, jnp.arange(Lo))
        inner_cells = inner_rev[::-1]

        new_row = jnp.concatenate([inner_cells, cell_Lo[None, :]], axis=0)
        return new_row, new_row

    _, bp_rows_rev = jax.lax.scan(process_row_bwd, last_row, jnp.arange(Li))
    bp_rows = bp_rows_rev[::-1]

    bp = jnp.concatenate([bp_rows, last_row[None, :, :]], axis=0)
    return bp


def _tok_to_pswm(tokens: jnp.ndarray, n_tokens: int) -> jnp.ndarray:
    """Convert token index array to one-hot PSWM (log-space)."""
    L = tokens.shape[0]
    pswm = jnp.full((L, n_tokens), NEG_INF)
    return pswm.at[jnp.arange(L), tokens].set(0.0)


def neural_forward_2d_tok(ptm, input_tokens, output_tokens, params) -> float:
    """Forward with tokenized sequences and position-dependent params."""
    in_pswm = _tok_to_pswm(input_tokens, ptm.n_in)
    out_pswm = _tok_to_pswm(output_tokens, ptm.n_out)
    return neural_forward_2d(ptm, in_pswm, out_pswm, params, LOGSUMEXP)


def neural_viterbi_2d_tok(ptm, input_tokens, output_tokens, params) -> float:
    """Viterbi with tokenized sequences and position-dependent params."""
    in_pswm = _tok_to_pswm(input_tokens, ptm.n_in)
    out_pswm = _tok_to_pswm(output_tokens, ptm.n_out)
    return neural_forward_2d(ptm, in_pswm, out_pswm, params, MAXPLUS)
