"""Parameterized 2D DP with position-dependent weight expressions.

A special PSWM-2D-DENSE-SIMPLE variant where transition weights are
computed at each (i, j) position from position-dependent parameters.
Enables neural transducer constructs (e.g. Alex Graves' neural transducer)
where parameters are computed from sequences by a neural network.

The caller provides:
- A ParameterizedMachine (compiled from a Machine with weight expressions)
- Two PSWMs (input and output emission log-probabilities)
- A dict mapping each parameter name to a (Li+1, Lo+1) tensor

At each cell (i, j) of the 2D DP matrix, the transition tensor is built
from the parameters at that position. JIT compiles the weight expression
evaluation into the computation graph.

Uses nested jax.lax.scan (outer over rows, inner over columns).
No associative scan because the silent closure varies by position.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import NEG_INF
from .semiring import LogSemiring, LOGSUMEXP, MAXPLUS
from .jax_weight import ParameterizedMachine


def _propagate_silent_fixed(cell, silent, semiring, n_iter):
    """Propagate silent transitions with a fixed number of iterations.

    Uses jax.lax.fori_loop with static bounds (differentiable).
    For acyclic machines, n_iter = S is exact. For cyclic machines
    this is an approximation.
    """
    def body(_, prev):
        incoming = prev[:, None] + silent
        update = semiring.reduce(incoming, axis=0)
        return semiring.plus(cell, update)

    return jax.lax.fori_loop(0, n_iter, body, cell)


def _propagate_silent_backward_fixed(cell, silent, semiring, n_iter):
    """Backward silent propagation with fixed iterations (differentiable)."""
    def body(_, prev):
        incoming = silent + prev[None, :]
        update = semiring.reduce(incoming, axis=1)
        return semiring.plus(cell, update)

    return jax.lax.fori_loop(0, n_iter, body, cell)


def _emit_trans_matrix(log_trans, in_emission, out_emission, semiring,
                       emit_in=True, emit_out=True):
    """Compute emission-weighted transition matrix.

    Args:
        log_trans: (n_in, n_out, S, S) transition tensor
        in_emission: (n_in,) input emission log-probs (or None)
        out_emission: (n_out,) output emission log-probs (or None)
        semiring: LogSemiring
        emit_in: whether to emit an input token
        emit_out: whether to emit an output token
    Returns:
        (S, S) combined transition matrix for the specified emission type.
    """
    if emit_in and emit_out:
        # Match: reduce over both in and out tokens
        in_e = in_emission.at[0].set(NEG_INF)
        out_e = out_emission.at[0].set(NEG_INF)
        weighted = (in_e[:, None, None, None] +
                    out_e[None, :, None, None] +
                    log_trans)
        return semiring.reduce(semiring.reduce(weighted, axis=0), axis=0)
    elif emit_in:
        # Insert: reduce over input tokens, out=0 (silent)
        in_e = in_emission.at[0].set(NEG_INF)
        weighted = in_e[:, None, None] + log_trans[:, 0, :, :]
        return semiring.reduce(weighted, axis=0)
    elif emit_out:
        # Delete: reduce over output tokens, in=0 (silent)
        out_e = out_emission.at[0].set(NEG_INF)
        weighted = out_e[:, None, None] + log_trans[0, :, :, :]
        return semiring.reduce(weighted, axis=0)
    else:
        # Silent
        return log_trans[0, 0]


def _build_lt_at(pm, params, i, j):
    """Build log_trans tensor at position (i, j)."""
    pos_params = {name: val[i, j] for name, val in params.items()}
    return pm.build_log_trans(pos_params)


def neural_forward_2d(pm: ParameterizedMachine,
                      input_pswm: jnp.ndarray,
                      output_pswm: jnp.ndarray,
                      params: dict[str, jnp.ndarray],
                      semiring: LogSemiring) -> float:
    """2D Forward/Viterbi with position-dependent transition weights.

    Args:
        pm: ParameterizedMachine (compiled from Machine with weight exprs)
        input_pswm: (Li, n_in) input emission log-probabilities
        output_pswm: (Lo, n_out) output emission log-probabilities
        params: dict mapping parameter names to (Li+1, Lo+1) arrays
        semiring: LOGSUMEXP for Forward, MAXPLUS for Viterbi

    Returns:
        Log-likelihood or Viterbi score (scalar).
    """
    S = pm.n_states
    Li = input_pswm.shape[0]
    Lo = output_pswm.shape[0]

    # --- Cell (0, 0): start state + silent ---
    lt_00 = _build_lt_at(pm, params, 0, 0)
    cell_00 = jnp.full(S, NEG_INF).at[0].set(0.0)
    cell_00 = _propagate_silent_fixed(cell_00, lt_00[0, 0], semiring, S)

    # --- Row 0: delete transitions only ---
    def scan_row0(cell_left, j_idx):
        j = j_idx + 1
        lt = _build_lt_at(pm, params, 0, j)
        out_e = output_pswm[j_idx]
        del_trans = _emit_trans_matrix(lt, None, out_e, semiring,
                                       emit_in=False, emit_out=True)
        cell = semiring.reduce(cell_left[:, None] + del_trans, axis=0)
        cell = _propagate_silent_fixed(cell, lt[0, 0], semiring, S)
        return cell, cell

    if Lo > 0:
        _, row0_cells = jax.lax.scan(scan_row0, cell_00, jnp.arange(Lo))
        row0 = jnp.concatenate([cell_00[None, :], row0_cells], axis=0)
    else:
        row0 = cell_00[None, :]

    if Li == 0:
        return row0[Lo, S - 1]

    # --- Rows 1..Li: outer scan ---
    def process_row(prev_row, i_idx):
        i = i_idx + 1
        in_e = input_pswm[i_idx]

        # Column 0: insert only
        lt_i0 = _build_lt_at(pm, params, i, 0)
        ins_trans = _emit_trans_matrix(lt_i0, in_e, None, semiring,
                                       emit_in=True, emit_out=False)
        cell_i0 = semiring.reduce(prev_row[0][:, None] + ins_trans, axis=0)
        cell_i0 = _propagate_silent_fixed(cell_i0, lt_i0[0, 0], semiring, S)

        if Lo == 0:
            return cell_i0[None, :], None

        # Columns 1..Lo: inner scan
        def inner_step(cell_left, j_idx):
            j = j_idx + 1
            out_e = output_pswm[j_idx]
            lt = _build_lt_at(pm, params, i, j)

            cell = jnp.full(S, NEG_INF)

            # Match: from prev_row[j-1]
            match_trans = _emit_trans_matrix(lt, in_e, out_e, semiring,
                                             emit_in=True, emit_out=True)
            cell = semiring.plus(
                cell,
                semiring.reduce(prev_row[j_idx][:, None] + match_trans, axis=0))

            # Insert: from prev_row[j]
            ins_trans = _emit_trans_matrix(lt, in_e, None, semiring,
                                           emit_in=True, emit_out=False)
            cell = semiring.plus(
                cell,
                semiring.reduce(prev_row[j][:, None] + ins_trans, axis=0))

            # Delete: from cell_left
            del_trans = _emit_trans_matrix(lt, None, out_e, semiring,
                                           emit_in=False, emit_out=True)
            cell = semiring.plus(
                cell,
                semiring.reduce(cell_left[:, None] + del_trans, axis=0))

            # Silent propagation
            cell = _propagate_silent_fixed(cell, lt[0, 0], semiring, S)

            return cell, cell

        _, inner_cells = jax.lax.scan(inner_step, cell_i0, jnp.arange(Lo))
        new_row = jnp.concatenate([cell_i0[None, :], inner_cells], axis=0)
        return new_row, None

    final_row, _ = jax.lax.scan(process_row, row0, jnp.arange(Li))
    return final_row[Lo, S - 1]


def neural_backward_2d(pm: ParameterizedMachine,
                       input_pswm: jnp.ndarray,
                       output_pswm: jnp.ndarray,
                       params: dict[str, jnp.ndarray],
                       semiring: LogSemiring) -> jnp.ndarray:
    """2D Backward with position-dependent transition weights.

    Returns full backward matrix (Li+1, Lo+1, S).
    """
    S = pm.n_states
    Li = input_pswm.shape[0]
    Lo = output_pswm.shape[0]

    # --- Terminal cell (Li, Lo): end state ---
    lt_term = _build_lt_at(pm, params, Li, Lo)
    term = jnp.full(S, NEG_INF).at[S - 1].set(0.0)
    term = _propagate_silent_backward_fixed(term, lt_term[0, 0], semiring, S)

    # --- Last row (i=Li): delete only, right to left ---
    def scan_last_del(cell_right, j_idx):
        # j_idx counts from right: j = Lo - 1 - j_idx
        j = Lo - 1 - j_idx
        lt = _build_lt_at(pm, params, Li, j)
        out_e = output_pswm[j]
        del_trans = _emit_trans_matrix(lt, None, out_e, semiring,
                                       emit_in=False, emit_out=True)
        cell = semiring.reduce(del_trans + cell_right[None, :], axis=1)
        cell = _propagate_silent_backward_fixed(cell, lt[0, 0], semiring, S)
        return cell, cell

    if Lo > 0:
        _, last_rev = jax.lax.scan(scan_last_del, term, jnp.arange(Lo))
        last_cells = last_rev[::-1]  # (Lo, S)
        last_row = jnp.concatenate([last_cells, term[None, :]], axis=0)
    else:
        last_row = term[None, :]

    if Li == 0:
        return last_row[None, :, :]

    # --- Rows Li-1..0: outer scan (backward) ---
    def process_row_bwd(next_row, i_idx):
        # i_idx counts from bottom: i = Li - 1 - i_idx
        i = Li - 1 - i_idx
        in_e = input_pswm[i]

        # Column Lo: insert only
        lt_lo = _build_lt_at(pm, params, i, Lo)
        ins_trans = _emit_trans_matrix(lt_lo, in_e, None, semiring,
                                       emit_in=True, emit_out=False)
        cell_Lo = semiring.reduce(ins_trans + next_row[Lo][None, :], axis=1)
        cell_Lo = _propagate_silent_backward_fixed(cell_Lo, lt_lo[0, 0], semiring, S)

        if Lo == 0:
            return cell_Lo[None, :], cell_Lo[None, :]

        # Columns Lo-1..0: inner scan (right to left)
        def inner_step_bwd(cell_right, j_idx):
            j = Lo - 1 - j_idx
            out_e = output_pswm[j]
            lt = _build_lt_at(pm, params, i, j)

            cell = jnp.full(S, NEG_INF)

            # Match: to (i+1, j+1) = next_row[j+1]
            match_trans = _emit_trans_matrix(lt, in_e, out_e, semiring,
                                             emit_in=True, emit_out=True)
            cell = semiring.plus(
                cell,
                semiring.reduce(match_trans + next_row[j + 1][None, :], axis=1))

            # Insert: to (i+1, j) = next_row[j]
            ins_trans = _emit_trans_matrix(lt, in_e, None, semiring,
                                           emit_in=True, emit_out=False)
            cell = semiring.plus(
                cell,
                semiring.reduce(ins_trans + next_row[j][None, :], axis=1))

            # Delete: to (i, j+1) = cell_right
            del_trans = _emit_trans_matrix(lt, None, out_e, semiring,
                                           emit_in=False, emit_out=True)
            cell = semiring.plus(
                cell,
                semiring.reduce(del_trans + cell_right[None, :], axis=1))

            # Silent
            cell = _propagate_silent_backward_fixed(cell, lt[0, 0], semiring, S)

            return cell, cell

        _, inner_rev = jax.lax.scan(inner_step_bwd, cell_Lo, jnp.arange(Lo))
        inner_cells = inner_rev[::-1]  # (Lo, S)

        new_row = jnp.concatenate([inner_cells, cell_Lo[None, :]], axis=0)
        return new_row, new_row

    _, bp_rows_rev = jax.lax.scan(process_row_bwd, last_row, jnp.arange(Li))
    bp_rows = bp_rows_rev[::-1]  # (Li, Lo+1, S)

    bp = jnp.concatenate([bp_rows, last_row[None, :, :]], axis=0)
    return bp


def neural_log_forward(pm: ParameterizedMachine,
                       input_pswm: jnp.ndarray,
                       output_pswm: jnp.ndarray,
                       params: dict[str, jnp.ndarray]) -> float:
    """Forward algorithm with position-dependent parameters.

    Args:
        pm: ParameterizedMachine compiled from a Machine with weight expressions.
        input_pswm: (Li, n_in) input emission log-probs (PSWMSeq.log_probs).
        output_pswm: (Lo, n_out) output emission log-probs (PSWMSeq.log_probs).
        params: dict mapping each parameter name to a (Li+1, Lo+1) JAX array.

    Returns:
        Log-likelihood (scalar).
    """
    return neural_forward_2d(pm, input_pswm, output_pswm, params, LOGSUMEXP)


def neural_log_viterbi(pm: ParameterizedMachine,
                       input_pswm: jnp.ndarray,
                       output_pswm: jnp.ndarray,
                       params: dict[str, jnp.ndarray]) -> float:
    """Viterbi algorithm with position-dependent parameters.

    Args:
        pm: ParameterizedMachine compiled from a Machine with weight expressions.
        input_pswm: (Li, n_in) input emission log-probs.
        output_pswm: (Lo, n_out) output emission log-probs.
        params: dict mapping each parameter name to a (Li+1, Lo+1) JAX array.

    Returns:
        Viterbi log-score (scalar).
    """
    return neural_forward_2d(pm, input_pswm, output_pswm, params, MAXPLUS)


def neural_log_backward_matrix(pm: ParameterizedMachine,
                               input_pswm: jnp.ndarray,
                               output_pswm: jnp.ndarray,
                               params: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Backward algorithm with position-dependent parameters.

    Returns full backward matrix (Li+1, Lo+1, S).
    """
    return neural_backward_2d(pm, input_pswm, output_pswm, params, LOGSUMEXP)
