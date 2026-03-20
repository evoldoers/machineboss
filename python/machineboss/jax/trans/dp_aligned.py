"""Alignment-constrained 1D DP for TransMachine.

Scans along a prescribed alignment path instead of visiting every cell.
Supports both fixed-weight and parameterized (neural) variants.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..types import NEG_INF
from ..semiring import LogSemiring, LOGSUMEXP, MAXPLUS
from .machine import TransMachine
from .kernel import (
    propagate_silent, emit_step_forward,
    emission_weights_2d,
)

# Alignment operation codes
MAT = 0
INS = 1
DEL = 2


def aligned_forward(tm: TransMachine,
                    input_tokens: jnp.ndarray,
                    output_tokens: jnp.ndarray,
                    alignment: jnp.ndarray,
                    semiring: LogSemiring) -> float:
    """Forward/Viterbi along a prescribed alignment.

    Args:
        tm: TransMachine
        input_tokens: (Li,) int32 input token indices (1-based)
        output_tokens: (Lo,) int32 output token indices (1-based)
        alignment: (A,) int32 array of MAT=0, INS=1, DEL=2
        semiring: LOGSUMEXP or MAXPLUS
    Returns:
        Log-likelihood or Viterbi score (scalar).
    """
    S = tm.n_states

    cell = jnp.full(S, NEG_INF).at[0].set(0.0)
    cell = propagate_silent(cell, tm, semiring)

    def scan_fn(carry, op):
        cell, i_pos, o_pos = carry

        in_tok = jnp.where((op == MAT) | (op == INS),
                           input_tokens[jnp.minimum(i_pos, input_tokens.shape[0] - 1)],
                           0)
        out_tok = jnp.where((op == MAT) | (op == DEL),
                            output_tokens[jnp.minimum(o_pos, output_tokens.shape[0] - 1)],
                            0)

        # Build emission weights for this step
        n_in = tm.n_in
        n_out = tm.n_out
        in_e = jnp.full(n_in, NEG_INF).at[in_tok].set(0.0)
        out_e = jnp.full(n_out, NEG_INF).at[out_tok].set(0.0)

        # Select mask based on op type
        is_match = (op == MAT)
        is_ins = (op == INS)
        is_del = (op == DEL)

        ew = emission_weights_2d(tm, in_e, out_e)

        # Apply all transition types, masked by op
        new_cell = jnp.full(S, NEG_INF)
        # Match
        match_cell = emit_step_forward(
            jnp.full(S, NEG_INF), cell, tm, tm.emit_both_mask, ew, semiring)
        new_cell = jnp.where(is_match, semiring.plus(new_cell, match_cell), new_cell)
        # Insert
        ins_cell = emit_step_forward(
            jnp.full(S, NEG_INF), cell, tm, tm.emit_in_mask, ew, semiring)
        new_cell = jnp.where(is_ins, semiring.plus(new_cell, ins_cell), new_cell)
        # Delete
        del_cell = emit_step_forward(
            jnp.full(S, NEG_INF), cell, tm, tm.emit_out_mask, ew, semiring)
        new_cell = jnp.where(is_del, semiring.plus(new_cell, del_cell), new_cell)

        new_cell = propagate_silent(new_cell, tm, semiring)

        new_i = i_pos + jnp.where((op == MAT) | (op == INS), 1, 0)
        new_o = o_pos + jnp.where((op == MAT) | (op == DEL), 1, 0)

        return (new_cell, new_i, new_o), None

    (final_cell, _, _), _ = jax.lax.scan(
        scan_fn, (cell, jnp.int32(0), jnp.int32(0)), alignment)

    return final_cell[S - 1]


def aligned_viterbi(tm: TransMachine,
                    input_tokens: jnp.ndarray,
                    output_tokens: jnp.ndarray,
                    alignment: jnp.ndarray) -> float:
    """Viterbi along a prescribed alignment."""
    return aligned_forward(tm, input_tokens, output_tokens, alignment, MAXPLUS)


# ---------------------------------------------------------------------------
# Neural (parameterized) alignment-constrained DP
# ---------------------------------------------------------------------------

def neural_aligned_forward(tm: TransMachine,
                           ptm,  # ParameterizedTransMachine
                           input_tokens: jnp.ndarray,
                           output_tokens: jnp.ndarray,
                           alignment: jnp.ndarray,
                           params: dict[str, jnp.ndarray],
                           semiring: LogSemiring) -> float:
    """Forward/Viterbi along alignment with position-dependent params.

    Args:
        tm: TransMachine (template structure, log_w ignored)
        ptm: ParameterizedTransMachine
        input_tokens, output_tokens: token sequences
        alignment: alignment operations
        params: position-dependent parameters
        semiring: LOGSUMEXP or MAXPLUS
    """
    S = tm.n_states

    # Build initial log_w from params at (0, 0)
    log_w_00 = ptm.build_log_w(params, 0, 0)
    tm0 = _with_log_w(tm, log_w_00)

    cell = jnp.full(S, NEG_INF).at[0].set(0.0)
    cell = propagate_silent(cell, tm0, semiring)

    def scan_fn(carry, op):
        cell, i_pos, o_pos = carry

        in_tok = jnp.where((op == MAT) | (op == INS),
                           input_tokens[jnp.minimum(i_pos, input_tokens.shape[0] - 1)],
                           0)
        out_tok = jnp.where((op == MAT) | (op == DEL),
                            output_tokens[jnp.minimum(o_pos, output_tokens.shape[0] - 1)],
                            0)

        new_i = i_pos + jnp.where((op == MAT) | (op == INS), 1, 0)
        new_o = o_pos + jnp.where((op == MAT) | (op == DEL), 1, 0)

        # Build position-dependent log_w
        log_w = ptm.build_log_w(params, new_i, new_o)
        tm_pos = _with_log_w(tm, log_w)

        n_in = tm.n_in
        n_out = tm.n_out
        in_e = jnp.full(n_in, NEG_INF).at[in_tok].set(0.0)
        out_e = jnp.full(n_out, NEG_INF).at[out_tok].set(0.0)
        ew = emission_weights_2d(tm_pos, in_e, out_e)

        is_match = (op == MAT)
        is_ins = (op == INS)
        is_del = (op == DEL)

        new_cell = jnp.full(S, NEG_INF)
        match_cell = emit_step_forward(
            jnp.full(S, NEG_INF), cell, tm_pos, tm_pos.emit_both_mask, ew, semiring)
        new_cell = jnp.where(is_match, semiring.plus(new_cell, match_cell), new_cell)
        ins_cell = emit_step_forward(
            jnp.full(S, NEG_INF), cell, tm_pos, tm_pos.emit_in_mask, ew, semiring)
        new_cell = jnp.where(is_ins, semiring.plus(new_cell, ins_cell), new_cell)
        del_cell = emit_step_forward(
            jnp.full(S, NEG_INF), cell, tm_pos, tm_pos.emit_out_mask, ew, semiring)
        new_cell = jnp.where(is_del, semiring.plus(new_cell, del_cell), new_cell)

        new_cell = propagate_silent(new_cell, tm_pos, semiring)

        return (new_cell, new_i, new_o), None

    (final_cell, _, _), _ = jax.lax.scan(
        scan_fn, (cell, jnp.int32(0), jnp.int32(0)), alignment)

    return final_cell[S - 1]


def neural_aligned_viterbi(tm, ptm, input_tokens, output_tokens,
                           alignment, params) -> float:
    """Viterbi along alignment with position-dependent params."""
    return neural_aligned_forward(tm, ptm, input_tokens, output_tokens,
                                  alignment, params, MAXPLUS)


def _with_log_w(tm: TransMachine, log_w: jnp.ndarray) -> TransMachine:
    """Return a copy of tm with replaced log_w."""
    return TransMachine(
        tm.src, tm.dst, tm.in_tok, tm.out_tok, log_w,
        tm.silent_mask, tm.emit_in_mask, tm.emit_out_mask, tm.emit_both_mask,
        tm.n_states, tm.n_in, tm.n_out, tm.input_tokens, tm.output_tokens,
    )


def validate_alignment(alignment: jnp.ndarray, Li: int, Lo: int) -> None:
    """Assert that alignment is consistent with sequence lengths."""
    a = jnp.asarray(alignment)
    n_mat = int(jnp.sum(a == MAT))
    n_ins = int(jnp.sum(a == INS))
    n_del = int(jnp.sum(a == DEL))
    if n_mat + n_ins != Li:
        raise ValueError(
            f"Alignment has {n_mat} MAT + {n_ins} INS = {n_mat + n_ins}, "
            f"but input length is {Li}")
    if n_mat + n_del != Lo:
        raise ValueError(
            f"Alignment has {n_mat} MAT + {n_del} DEL = {n_mat + n_del}, "
            f"but output length is {Lo}")
