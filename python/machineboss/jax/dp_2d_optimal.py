"""2D DP engine using anti-diagonal wavefront (OPTIMAL strategy).

Outer jax.lax.scan over diagonals d = i + o.
Inner jax.vmap over all cells on each diagonal (they are independent:
match predecessors are on d-2, insert/delete predecessors on d-1).
JIT-compilable: no Python for-loops in the DP computation.

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

    # Initialize dp grid
    dp = jnp.full((Li + 1, Lo + 1, S), NEG_INF)
    dp = dp.at[0, 0, 0].set(0.0)
    dp = dp.at[0, 0].set(propagate_silent(dp[0, 0], silent, semiring))

    if Li + Lo == 0:
        return dp[0, 0, S - 1]

    # Precompute transition matrices
    all_ins = (_precompute_emit_trans(in_emit, log_trans[:, 0, :, :], semiring)
               if Li > 0 else jnp.full((1, S, S), NEG_INF))
    all_del = (_precompute_emit_trans(out_emit, log_trans[0, :, :, :], semiring)
               if Lo > 0 else jnp.full((1, S, S), NEG_INF))
    all_match = (_precompute_all_match(in_emit, out_emit, log_trans, semiring)
                 if Li > 0 and Lo > 0 else jnp.full((1, 1, S, S), NEG_INF))

    D_max = min(Li + 1, Lo + 1)
    max_i_idx = max(Li - 1, 0)
    max_o_idx = max(Lo - 1, 0)

    def scan_fn(dp, d):
        i_min = jnp.maximum(0, d - Lo)
        js = jnp.arange(D_max)
        i_vals = i_min + js
        o_vals = d - i_vals

        # Validity: within grid and not the initial cell (0,0)
        valid = (i_vals <= Li) & (o_vals >= 0) & (o_vals <= Lo)
        valid = valid & ~((i_vals == 0) & (o_vals == 0))

        def compute_cell(i, o):
            ip = jnp.clip(i - 1, 0, Li)
            op = jnp.clip(o - 1, 0, Lo)

            cell = jnp.full(S, NEG_INF)

            # Match from (i-1, o-1)
            mt = all_match[jnp.clip(i - 1, 0, max_i_idx),
                           jnp.clip(o - 1, 0, max_o_idx)]
            mc = emit_step_forward(jnp.full(S, NEG_INF), dp[ip, op], mt, semiring)
            cell = jnp.where((i > 0) & (o > 0), semiring.plus(cell, mc), cell)

            # Insert from (i-1, o)
            it = all_ins[jnp.clip(i - 1, 0, max_i_idx)]
            ic = emit_step_forward(jnp.full(S, NEG_INF),
                                   dp[ip, jnp.clip(o, 0, Lo)], it, semiring)
            cell = jnp.where(i > 0, semiring.plus(cell, ic), cell)

            # Delete from (i, o-1)
            dt = all_del[jnp.clip(o - 1, 0, max_o_idx)]
            dc = emit_step_forward(jnp.full(S, NEG_INF),
                                   dp[jnp.clip(i, 0, Li), op], dt, semiring)
            cell = jnp.where(o > 0, semiring.plus(cell, dc), cell)

            cell = propagate_silent(cell, silent, semiring)
            return cell

        cells = jax.vmap(compute_cell)(i_vals, o_vals)  # (D_max, S)

        # Write back: padded cells write to (0,0) to avoid collisions
        i_write = jnp.where(valid, i_vals, 0)
        o_write = jnp.where(valid, o_vals, 0)
        vals = jnp.where(valid[:, None], cells, dp[i_write, o_write])
        dp = dp.at[i_write, o_write].set(vals)

        return dp, None

    dp, _ = jax.lax.scan(scan_fn, dp, jnp.arange(1, Li + Lo + 1))
    return dp[Li, Lo, S - 1]


# ============================================================
# Backward
# ============================================================

def backward_2d_optimal(machine: JAXMachine, input_seq, output_seq,
                        semiring: LogSemiring) -> jnp.ndarray:
    """2D Backward using anti-diagonal wavefront with scan + vmap.

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

    # Initialize bp grid
    bp = jnp.full((Li + 1, Lo + 1, S), NEG_INF)
    bp = bp.at[Li, Lo, S - 1].set(0.0)
    bp = bp.at[Li, Lo].set(propagate_silent_backward(bp[Li, Lo], silent, semiring))

    if Li + Lo == 0:
        return bp

    # Precompute transition matrices (same as forward)
    all_ins = (_precompute_emit_trans(in_emit, log_trans[:, 0, :, :], semiring)
               if Li > 0 else jnp.full((1, S, S), NEG_INF))
    all_del = (_precompute_emit_trans(out_emit, log_trans[0, :, :, :], semiring)
               if Lo > 0 else jnp.full((1, S, S), NEG_INF))
    all_match = (_precompute_all_match(in_emit, out_emit, log_trans, semiring)
                 if Li > 0 and Lo > 0 else jnp.full((1, 1, S, S), NEG_INF))

    D_max = min(Li + 1, Lo + 1)
    max_i_idx = max(Li - 1, 0)
    max_o_idx = max(Lo - 1, 0)

    def scan_fn(bp, d):
        i_min = jnp.maximum(0, d - Lo)
        js = jnp.arange(D_max)
        i_vals = i_min + js
        o_vals = d - i_vals

        # Validity: within grid and not the terminal cell (Li, Lo)
        valid = (i_vals <= Li) & (o_vals >= 0) & (o_vals <= Lo)
        valid = valid & ~((i_vals == Li) & (o_vals == Lo))

        def compute_cell(i, o):
            i_next = jnp.clip(i + 1, 0, Li)
            o_next = jnp.clip(o + 1, 0, Lo)

            cell = jnp.full(S, NEG_INF)

            # Match to (i+1, o+1)
            mt = all_match[jnp.clip(i, 0, max_i_idx),
                           jnp.clip(o, 0, max_o_idx)]
            mc = emit_step_backward(jnp.full(S, NEG_INF),
                                    bp[i_next, o_next], mt, semiring)
            cell = jnp.where((i < Li) & (o < Lo),
                             semiring.plus(cell, mc), cell)

            # Insert to (i+1, o)
            it = all_ins[jnp.clip(i, 0, max_i_idx)]
            ic = emit_step_backward(jnp.full(S, NEG_INF),
                                    bp[i_next, jnp.clip(o, 0, Lo)], it, semiring)
            cell = jnp.where(i < Li, semiring.plus(cell, ic), cell)

            # Delete to (i, o+1)
            dt = all_del[jnp.clip(o, 0, max_o_idx)]
            dc = emit_step_backward(jnp.full(S, NEG_INF),
                                    bp[jnp.clip(i, 0, Li), o_next], dt, semiring)
            cell = jnp.where(o < Lo, semiring.plus(cell, dc), cell)

            cell = propagate_silent_backward(cell, silent, semiring)
            return cell

        cells = jax.vmap(compute_cell)(i_vals, o_vals)  # (D_max, S)

        # Write back: padded cells write to (Li, Lo) to avoid collisions
        i_write = jnp.where(valid, i_vals, Li)
        o_write = jnp.where(valid, o_vals, Lo)
        vals = jnp.where(valid[:, None], cells, bp[i_write, o_write])
        bp = bp.at[i_write, o_write].set(vals)

        return bp, None

    # Scan over diagonals in reverse: d from Li+Lo-1 down to 0
    bp, _ = jax.lax.scan(scan_fn, bp, jnp.arange(Li + Lo)[::-1])
    return bp
