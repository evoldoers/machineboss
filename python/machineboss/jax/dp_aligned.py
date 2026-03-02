"""Alignment-constrained 1D DP along a prescribed pairwise alignment.

Instead of visiting every cell in the (Li+1, Lo+1) 2D DP matrix, the caller
supplies an alignment: a sequence of {MAT, INS, DEL} operations.  The DP
proceeds along this 1D path, visiting only the cells touched by the alignment.

Alignment operations:
- MAT (0): consume one input AND one output token  (i++, j++)
- INS (1): consume one input token only             (i++)
- DEL (2): consume one output token only             (j++)

The number of MAT + INS operations must equal Li (input length).
The number of MAT + DEL operations must equal Lo (output length).

Two flavors:
- Standard: uses a fixed JAXMachine (pre-evaluated transition tensor)
- Neural: uses a ParameterizedMachine with position-dependent parameters

Both use ``jax.lax.scan`` over the alignment steps → fully JIT-compilable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import JAXMachine, NEG_INF
from .semiring import LogSemiring, LOGSUMEXP, MAXPLUS
from .kernel_dense import propagate_silent
from .jax_weight import ParameterizedMachine

# Alignment operation codes
MAT = 0
INS = 1
DEL = 2


def _propagate_silent_fixed(cell, silent, semiring, n_iter):
    """Fixed-iteration silent propagation (differentiable)."""
    def body(_, prev):
        incoming = prev[:, None] + silent
        update = semiring.reduce(incoming, axis=0)
        return semiring.plus(cell, update)

    return jax.lax.fori_loop(0, n_iter, body, cell)


# ---------------------------------------------------------------------------
# Standard (fixed-weight) alignment-constrained DP
# ---------------------------------------------------------------------------

def aligned_forward(machine: JAXMachine,
                    input_tokens: jnp.ndarray,
                    output_tokens: jnp.ndarray,
                    alignment: jnp.ndarray,
                    semiring: LogSemiring) -> float:
    """Forward/Viterbi along a prescribed alignment.

    Args:
        machine: JAXMachine with dense log_trans (n_in, n_out, S, S).
        input_tokens: (Li,) int32 input token indices (1-based).
        output_tokens: (Lo,) int32 output token indices (1-based).
        alignment: (A,) int32 array of MAT=0, INS=1, DEL=2.
        semiring: LOGSUMEXP for Forward, MAXPLUS for Viterbi.

    Returns:
        Log-likelihood or Viterbi score (scalar).
    """
    S = machine.n_states
    log_trans = machine.log_trans
    silent = log_trans[0, 0]  # (S, S) silent transitions

    # Initialize: start state + silent closure
    cell = jnp.full(S, NEG_INF).at[0].set(0.0)
    cell = propagate_silent(cell, silent, semiring)

    # Track current input/output positions
    # i_pos, o_pos start at 0 and increment based on alignment ops
    def scan_fn(carry, op):
        cell, i_pos, o_pos = carry

        # Determine which token indices to use
        # For MAT/INS: consume input token at i_pos
        # For MAT/DEL: consume output token at o_pos
        in_tok = jnp.where((op == MAT) | (op == INS),
                           input_tokens[jnp.minimum(i_pos, input_tokens.shape[0] - 1)],
                           0)
        out_tok = jnp.where((op == MAT) | (op == DEL),
                            output_tokens[jnp.minimum(o_pos, output_tokens.shape[0] - 1)],
                            0)

        # Build transition matrix for this step
        # trans[in_tok, out_tok, :, :] gives the (S, S) transition matrix
        trans_mat = log_trans[in_tok, out_tok]  # (S, S)

        # Apply transition
        new_cell = semiring.reduce(cell[:, None] + trans_mat, axis=0)

        # Silent closure
        new_cell = propagate_silent(new_cell, silent, semiring)

        # Update positions
        new_i = i_pos + jnp.where((op == MAT) | (op == INS), 1, 0)
        new_o = o_pos + jnp.where((op == MAT) | (op == DEL), 1, 0)

        return (new_cell, new_i, new_o), None

    (final_cell, _, _), _ = jax.lax.scan(
        scan_fn, (cell, jnp.int32(0), jnp.int32(0)), alignment)

    return final_cell[S - 1]


def aligned_log_forward(machine: JAXMachine,
                        input_tokens: jnp.ndarray,
                        output_tokens: jnp.ndarray,
                        alignment: jnp.ndarray) -> float:
    """Forward algorithm along a prescribed alignment.

    Args:
        machine: JAXMachine with dense log_trans.
        input_tokens: (Li,) int32 input token indices (1-based).
        output_tokens: (Lo,) int32 output token indices (1-based).
        alignment: (A,) int32 array of MAT=0, INS=1, DEL=2.

    Returns:
        Log-likelihood (scalar).
    """
    return aligned_forward(machine, input_tokens, output_tokens,
                           alignment, LOGSUMEXP)


def aligned_log_viterbi(machine: JAXMachine,
                        input_tokens: jnp.ndarray,
                        output_tokens: jnp.ndarray,
                        alignment: jnp.ndarray) -> float:
    """Viterbi algorithm along a prescribed alignment.

    Returns:
        Viterbi log-score (scalar).
    """
    return aligned_forward(machine, input_tokens, output_tokens,
                           alignment, MAXPLUS)


# ---------------------------------------------------------------------------
# Neural (parameterized) alignment-constrained DP
# ---------------------------------------------------------------------------

def _build_lt_at(pm, params, i, j):
    """Build log_trans at position (i, j), supporting broadcast shapes."""
    pos_params = {
        name: val[jnp.minimum(i, val.shape[0] - 1),
                   jnp.minimum(j, val.shape[1] - 1)]
        for name, val in params.items()
    }
    return pm.build_log_trans(pos_params)


def neural_aligned_forward(pm: ParameterizedMachine,
                           input_tokens: jnp.ndarray,
                           output_tokens: jnp.ndarray,
                           alignment: jnp.ndarray,
                           params: dict[str, jnp.ndarray],
                           semiring: LogSemiring) -> float:
    """Forward/Viterbi along a prescribed alignment with position-dependent params.

    Args:
        pm: ParameterizedMachine with compiled weight expressions.
        input_tokens: (Li,) int32 input token indices (1-based).
        output_tokens: (Lo,) int32 output token indices (1-based).
        alignment: (A,) int32 array of MAT=0, INS=1, DEL=2.
        params: dict mapping parameter names to arrays broadcastable to
            (Li+1, Lo+1).
        semiring: LOGSUMEXP for Forward, MAXPLUS for Viterbi.

    Returns:
        Log-likelihood or Viterbi score (scalar).
    """
    S = pm.n_states

    # Initialize: start state + silent closure at (0, 0)
    lt_00 = _build_lt_at(pm, params, 0, 0)
    cell = jnp.full(S, NEG_INF).at[0].set(0.0)
    cell = _propagate_silent_fixed(cell, lt_00[0, 0], semiring, S)

    def scan_fn(carry, op):
        cell, i_pos, o_pos = carry

        # Determine token indices
        in_tok = jnp.where((op == MAT) | (op == INS),
                           input_tokens[jnp.minimum(i_pos, input_tokens.shape[0] - 1)],
                           0)
        out_tok = jnp.where((op == MAT) | (op == DEL),
                            output_tokens[jnp.minimum(o_pos, output_tokens.shape[0] - 1)],
                            0)

        # New position (after consuming this alignment op)
        new_i = i_pos + jnp.where((op == MAT) | (op == INS), 1, 0)
        new_o = o_pos + jnp.where((op == MAT) | (op == DEL), 1, 0)

        # Build position-dependent log_trans at the destination cell
        lt = _build_lt_at(pm, params, new_i, new_o)

        # Transition
        trans_mat = lt[in_tok, out_tok]  # (S, S)
        new_cell = semiring.reduce(cell[:, None] + trans_mat, axis=0)

        # Silent closure
        new_cell = _propagate_silent_fixed(new_cell, lt[0, 0], semiring, S)

        return (new_cell, new_i, new_o), None

    (final_cell, _, _), _ = jax.lax.scan(
        scan_fn, (cell, jnp.int32(0), jnp.int32(0)), alignment)

    return final_cell[S - 1]


def neural_aligned_log_forward(pm: ParameterizedMachine,
                               input_tokens: jnp.ndarray,
                               output_tokens: jnp.ndarray,
                               alignment: jnp.ndarray,
                               params: dict[str, jnp.ndarray]) -> float:
    """Forward algorithm along a prescribed alignment with position-dependent params.

    Args:
        pm: ParameterizedMachine with compiled weight expressions.
        input_tokens: (Li,) int32 input token indices (1-based).
        output_tokens: (Lo,) int32 output token indices (1-based).
        alignment: (A,) int32 array of MAT=0, INS=1, DEL=2.
        params: dict mapping parameter names to broadcastable arrays.

    Returns:
        Log-likelihood (scalar).
    """
    return neural_aligned_forward(pm, input_tokens, output_tokens,
                                  alignment, params, LOGSUMEXP)


def neural_aligned_log_viterbi(pm: ParameterizedMachine,
                               input_tokens: jnp.ndarray,
                               output_tokens: jnp.ndarray,
                               alignment: jnp.ndarray,
                               params: dict[str, jnp.ndarray]) -> float:
    """Viterbi algorithm along a prescribed alignment with position-dependent params.

    Returns:
        Viterbi log-score (scalar).
    """
    return neural_aligned_forward(pm, input_tokens, output_tokens,
                                  alignment, params, MAXPLUS)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_alignment(alignment: jnp.ndarray, Li: int, Lo: int) -> None:
    """Assert that alignment is consistent with sequence lengths.

    MAT + INS must equal Li.  MAT + DEL must equal Lo.

    Raises:
        ValueError: if alignment is inconsistent.
    """
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
