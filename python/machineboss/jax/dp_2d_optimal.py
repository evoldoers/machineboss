"""2D DP engine using anti-diagonal wavefront (OPTIMAL strategy).

Outer jax.lax.scan over diagonals d = i + o.
Inner jax.vmap over all cells on each diagonal (they are independent:
match predecessors are on d-2, insert/delete predecessors on d-1).
JIT-compilable: no Python for-loops in the DP computation.

Carry is just two diagonal slices (d-1 and d-2), each of shape (D_max, S)
with D_max = min(Li, Lo) + 1. This keeps the scan carry O(min(Li,Lo)*S)
instead of O(Li*Lo*S), avoiding GPU OOM on large pairs. Diagonals produced
by the scan are scattered into the full grid after the scan completes
(backward only — forward returns a scalar).

Complexity: O(Li + Lo) sequential steps, O(min(Li,Lo) * S^2) parallel work per step.
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


def _precompute_emit_trans(emit, trans_slice, semiring):
    """Precompute weighted transition matrices for all positions.

    Args:
        emit: (L, n_tokens) emission weights
        trans_slice: (n_tokens, S, S) transition matrices
        semiring: LogSemiring
    Returns:
        (L, S, S) per-position transition matrices.
    """
    def make_one(e):
        e = e.at[0].set(NEG_INF)
        return semiring.reduce(e[:, None, None] + trans_slice, axis=0)
    return jax.vmap(make_one)(emit)


def _precompute_all_match(in_emit, out_emit, log_trans, semiring):
    """Precompute match transition matrices for all (input_pos, output_pos) pairs.

    Factored computation: first reduce over input tokens, then output tokens.

    Args:
        in_emit: (Li, n_in) input emission weights
        out_emit: (Lo, n_out) output emission weights
        log_trans: (n_in, n_out, S, S) full transition tensor
        semiring: LogSemiring
    Returns:
        (Li, Lo, S, S) match transition matrices.
    """
    # First reduce over input tokens: in_marginal[i, out_tok, S, S]
    def marginal_in(in_e):
        in_e = in_e.at[0].set(NEG_INF)
        return semiring.reduce(
            in_e[:, None, None, None] + log_trans, axis=0)  # (n_out, S, S)
    in_marg = jax.vmap(marginal_in)(in_emit)  # (Li, n_out, S, S)

    # Then reduce over output tokens for each (i, o) pair
    def match_pair(marg_i, out_e):
        out_e = out_e.at[0].set(NEG_INF)
        return semiring.reduce(out_e[:, None, None] + marg_i, axis=0)  # (S, S)

    def match_row(marg_i):
        return jax.vmap(lambda out_e: match_pair(marg_i, out_e))(out_emit)

    return jax.vmap(match_row)(in_marg)  # (Li, Lo, S, S)


# ============================================================
# Forward
# ============================================================

def forward_2d_optimal(machine: JAXMachine, input_seq, output_seq,
                       semiring: LogSemiring) -> float:
    """2D Forward/Viterbi using anti-diagonal wavefront with scan + vmap.

    Carry is only two diagonal slices (d-1 and d-2), each (D_max, S) where
    D_max = min(Li, Lo) + 1, rather than the full (Li+1, Lo+1, S) grid.

    Args:
        machine: JAXMachine with dense log_trans
        input_seq: TokenSeq, PSWMSeq, or jnp.ndarray
        output_seq: TokenSeq, PSWMSeq, or jnp.ndarray
        semiring: LOGSUMEXP or MAXPLUS
    Returns:
        Log-likelihood or Viterbi score (scalar).
    """
    assert machine.log_trans is not None
    S = machine.n_states
    log_trans = machine.log_trans

    input_seq = wrap_seq(input_seq, machine.n_input_tokens)
    output_seq = wrap_seq(output_seq, machine.n_output_tokens)

    Li = len(input_seq) if input_seq is not None else 0
    Lo = len(output_seq) if output_seq is not None else 0

    in_emit = input_seq.emission_weights(machine.n_input_tokens) if Li > 0 else None
    out_emit = output_seq.emission_weights(machine.n_output_tokens) if Lo > 0 else None

    silent = log_trans[0, 0]

    # Initial cell at (0, 0): start state = 0, silent-propagated.
    cell_00 = jnp.full(S, NEG_INF).at[0].set(0.0)
    cell_00 = propagate_silent(cell_00, silent, semiring)

    if Li + Lo == 0:
        return cell_00[S - 1]

    # Precompute transition matrices
    all_ins = (_precompute_emit_trans(in_emit, log_trans[:, 0, :, :], semiring)
               if Li > 0 else jnp.full((1, S, S), NEG_INF))
    all_del = (_precompute_emit_trans(out_emit, log_trans[0, :, :, :], semiring)
               if Lo > 0 else jnp.full((1, S, S), NEG_INF))
    all_match = (_precompute_all_match(in_emit, out_emit, log_trans, semiring)
                 if Li > 0 and Lo > 0 else jnp.full((1, 1, S, S), NEG_INF))

    D_max = min(Li, Lo) + 1
    max_i_idx = max(Li - 1, 0)
    max_o_idx = max(Lo - 1, 0)

    def _i_min(d):
        return jnp.maximum(0, d - Lo)

    # Diagonal d=0: single cell (0,0) at local index 0.
    diag0 = jnp.full((D_max, S), NEG_INF).at[0].set(cell_00)
    empty_diag = jnp.full((D_max, S), NEG_INF)

    def compute_cell(prev_diag, prev_prev_diag, d, k):
        """Compute cell k on diagonal d from prev (d-1) and prev_prev (d-2)."""
        i = _i_min(d) + k
        o = d - i

        cell = jnp.full(S, NEG_INF)

        # Match predecessor (i-1, o-1) on d-2.
        m_k = jnp.clip((i - 1) - _i_min(d - 2), 0, D_max - 1)
        mt = all_match[jnp.clip(i - 1, 0, max_i_idx),
                       jnp.clip(o - 1, 0, max_o_idx)]
        mc = emit_step_forward(jnp.full(S, NEG_INF),
                               prev_prev_diag[m_k], mt, semiring)
        cell = jnp.where((i > 0) & (o > 0), semiring.plus(cell, mc), cell)

        # Insert predecessor (i-1, o) on d-1.
        ins_k = jnp.clip((i - 1) - _i_min(d - 1), 0, D_max - 1)
        it = all_ins[jnp.clip(i - 1, 0, max_i_idx)]
        ic = emit_step_forward(jnp.full(S, NEG_INF),
                               prev_diag[ins_k], it, semiring)
        cell = jnp.where(i > 0, semiring.plus(cell, ic), cell)

        # Delete predecessor (i, o-1) on d-1.
        del_k = jnp.clip(i - _i_min(d - 1), 0, D_max - 1)
        dt = all_del[jnp.clip(o - 1, 0, max_o_idx)]
        dc = emit_step_forward(jnp.full(S, NEG_INF),
                               prev_diag[del_k], dt, semiring)
        cell = jnp.where(o > 0, semiring.plus(cell, dc), cell)

        cell = propagate_silent(cell, silent, semiring)
        return cell

    def scan_fn(carry, d):
        prev_diag, prev_prev_diag = carry  # (D_max, S) each
        ks = jnp.arange(D_max)
        i_vals = _i_min(d) + ks
        o_vals = d - i_vals

        # Validity: within grid and not the initial cell (0,0).
        valid = (i_vals <= Li) & (o_vals >= 0) & (o_vals <= Lo)
        valid = valid & ~((i_vals == 0) & (o_vals == 0))

        cells = jax.vmap(
            lambda k: compute_cell(prev_diag, prev_prev_diag, d, k)
        )(ks)  # (D_max, S)
        cells = jnp.where(valid[:, None], cells, NEG_INF)
        return (cells, prev_diag), None

    (final_diag, _), _ = jax.lax.scan(
        scan_fn, (diag0, empty_diag),
        jnp.arange(1, Li + Lo + 1))

    # (Li, Lo) is the only valid cell on diagonal d = Li+Lo: i_min = Li,
    # so it sits at local index 0.
    return final_diag[0, S - 1]


# ============================================================
# Backward
# ============================================================

def backward_2d_optimal(machine: JAXMachine, input_seq, output_seq,
                        semiring: LogSemiring) -> jnp.ndarray:
    """2D Backward using anti-diagonal wavefront with scan + vmap.

    Carry is only two diagonal slices (d+1 and d+2), each (D_max, S).
    Scanned diagonals are scattered into the full (Li+1, Lo+1, S) grid
    via linearized indices with a dummy overflow slot.

    Args:
        machine: JAXMachine with dense log_trans
        input_seq: TokenSeq, PSWMSeq, or jnp.ndarray
        output_seq: TokenSeq, PSWMSeq, or jnp.ndarray
        semiring: LOGSUMEXP or MAXPLUS
    Returns:
        Backward matrix of shape (Li+1, Lo+1, S).
    """
    assert machine.log_trans is not None
    S = machine.n_states
    log_trans = machine.log_trans

    input_seq = wrap_seq(input_seq, machine.n_input_tokens)
    output_seq = wrap_seq(output_seq, machine.n_output_tokens)

    Li = len(input_seq) if input_seq is not None else 0
    Lo = len(output_seq) if output_seq is not None else 0

    in_emit = input_seq.emission_weights(machine.n_input_tokens) if Li > 0 else None
    out_emit = output_seq.emission_weights(machine.n_output_tokens) if Lo > 0 else None

    silent = log_trans[0, 0]

    # Terminal cell at (Li, Lo): final state = S-1, silent-propagated backward.
    cell_LiLo = jnp.full(S, NEG_INF).at[S - 1].set(0.0)
    cell_LiLo = propagate_silent_backward(cell_LiLo, silent, semiring)

    if Li + Lo == 0:
        bp = jnp.full((Li + 1, Lo + 1, S), NEG_INF)
        bp = bp.at[0, 0].set(cell_LiLo)
        return bp

    # Precompute transition matrices (same as forward)
    all_ins = (_precompute_emit_trans(in_emit, log_trans[:, 0, :, :], semiring)
               if Li > 0 else jnp.full((1, S, S), NEG_INF))
    all_del = (_precompute_emit_trans(out_emit, log_trans[0, :, :, :], semiring)
               if Lo > 0 else jnp.full((1, S, S), NEG_INF))
    all_match = (_precompute_all_match(in_emit, out_emit, log_trans, semiring)
                 if Li > 0 and Lo > 0 else jnp.full((1, 1, S, S), NEG_INF))

    D_max = min(Li, Lo) + 1
    n_diags = Li + Lo  # we scan d = Li+Lo-1 down to 0
    max_i_idx = max(Li - 1, 0)
    max_o_idx = max(Lo - 1, 0)

    def _i_min(d):
        return jnp.maximum(0, d - Lo)

    # Terminal diagonal d = Li+Lo: single cell (Li,Lo) at local index 0
    # (since i_min = max(0, Li+Lo - Lo) = Li).
    diag_term = jnp.full((D_max, S), NEG_INF).at[0].set(cell_LiLo)
    empty_diag = jnp.full((D_max, S), NEG_INF)

    def compute_cell(next_diag, next_next_diag, d, k):
        """Compute backward cell k on diagonal d from next (d+1) and next_next (d+2)."""
        i = _i_min(d) + k
        o = d - i

        cell = jnp.full(S, NEG_INF)

        # Match successor (i+1, o+1) on d+2.
        m_k = jnp.clip((i + 1) - _i_min(d + 2), 0, D_max - 1)
        mt = all_match[jnp.clip(i, 0, max_i_idx),
                       jnp.clip(o, 0, max_o_idx)]
        mc = emit_step_backward(jnp.full(S, NEG_INF),
                                next_next_diag[m_k], mt, semiring)
        cell = jnp.where((i < Li) & (o < Lo),
                         semiring.plus(cell, mc), cell)

        # Insert successor (i+1, o) on d+1.
        ins_k = jnp.clip((i + 1) - _i_min(d + 1), 0, D_max - 1)
        it = all_ins[jnp.clip(i, 0, max_i_idx)]
        ic = emit_step_backward(jnp.full(S, NEG_INF),
                                next_diag[ins_k], it, semiring)
        cell = jnp.where(i < Li, semiring.plus(cell, ic), cell)

        # Delete successor (i, o+1) on d+1.
        del_k = jnp.clip(i - _i_min(d + 1), 0, D_max - 1)
        dt = all_del[jnp.clip(o, 0, max_o_idx)]
        dc = emit_step_backward(jnp.full(S, NEG_INF),
                                next_diag[del_k], dt, semiring)
        cell = jnp.where(o < Lo, semiring.plus(cell, dc), cell)

        cell = propagate_silent_backward(cell, silent, semiring)
        return cell

    def scan_fn(carry, d):
        next_diag, next_next_diag = carry  # (D_max, S) each
        ks = jnp.arange(D_max)
        i_vals = _i_min(d) + ks
        o_vals = d - i_vals
        valid = (i_vals <= Li) & (o_vals >= 0) & (o_vals <= Lo)
        cells = jax.vmap(
            lambda k: compute_cell(next_diag, next_next_diag, d, k)
        )(ks)  # (D_max, S)
        cells = jnp.where(valid[:, None], cells, NEG_INF)
        return (cells, next_diag), cells

    # Scan d from Li+Lo-1 down to 0. Scan output is aligned with scan input,
    # so all_diags[0] corresponds to d = Li+Lo-1, all_diags[-1] to d = 0.
    (_, _), all_diags = jax.lax.scan(
        scan_fn, (diag_term, empty_diag),
        jnp.arange(n_diags)[::-1])

    # Scatter scanned diagonals into the full (Li+1, Lo+1, S) grid via
    # linearized indices with a dummy overflow slot. Each valid (i,j)
    # appears on exactly one diagonal at one local index, so valid flat
    # indices are unique; invalid slots all write to the dummy slot.
    n_flat = (Li + 1) * (Lo + 1)
    d_vals = jnp.arange(n_diags)[::-1]             # aligned with all_diags axis 0
    k_vals = jnp.arange(D_max)
    dd, kk = jnp.meshgrid(d_vals, k_vals, indexing='ij')  # (n_diags, D_max)
    ii = jnp.maximum(0, dd - Lo) + kk
    jj = dd - ii
    valid_grid = (ii <= Li) & (jj >= 0) & (jj <= Lo)
    lin = ii * (Lo + 1) + jj
    lin_safe = jnp.where(valid_grid, lin, n_flat).ravel()
    vals_flat = all_diags.reshape(-1, S)

    bp_flat = jnp.full((n_flat + 1, S), NEG_INF)
    bp_flat = bp_flat.at[lin_safe].set(vals_flat)
    bp = bp_flat[:n_flat].reshape(Li + 1, Lo + 1, S)

    # Place terminal cell explicitly (it was the scan initial, not a scan output).
    bp = bp.at[Li, Lo].set(cell_LiLo)
    return bp
