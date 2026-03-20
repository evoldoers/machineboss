"""Forward-backward with expected transition counts for TransMachine.

Fully vectorized: computes per-transition expected counts using
position-broadcasting and vmap over transitions. No Python for-loops
in the count computation. The forward and backward passes use the
same JIT-compiled engines as forward_2d/backward_2d.

Supports both simple and optimal (wavefront) strategies.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..types import NEG_INF
from ..semiring import LOGSUMEXP
from .machine import TransMachine
from .dp_2d import forward_2d_matrix, backward_2d
from ..seq import wrap_seq


def forward_backward(tm: TransMachine,
                     input_seq: jnp.ndarray,
                     output_seq: jnp.ndarray, *,
                     strategy: str = 'auto') -> tuple[float, jnp.ndarray]:
    """Compute log-likelihood and expected transition counts.

    Fully vectorized: uses vmap over transitions and broadcasting
    over positions to compute counts without Python for-loops.

    Args:
        tm: TransMachine
        input_seq: (Li,) int32 input token indices (1-based)
        output_seq: (Lo,) int32 output token indices (1-based)
        strategy: 'simple', 'optimal', or 'auto'

    Returns:
        (log_likelihood, counts) where counts is (T,) expected count
        per transition.
    """
    S = tm.n_states
    semiring = LOGSUMEXP

    # Forward pass: full (Li+1, Lo+1, S) matrix
    dp = forward_2d_matrix(tm, input_seq, output_seq, semiring, strategy=strategy)
    ll = dp[-1, -1, S - 1]

    # Backward pass: full (Li+1, Lo+1, S) matrix
    bp = backward_2d(tm, input_seq, output_seq, semiring, strategy=strategy)

    # Vectorized counts: vmap over transitions
    counts = _compute_counts_vectorized(tm, dp, bp, input_seq, output_seq, ll)
    return ll, counts


def _compute_counts_vectorized(tm, dp, bp, input_seq, output_seq, ll):
    """Compute expected counts vectorized over transitions.

    For each transition t = (src, dst, in_tok, out_tok, log_w):
      count[t] = exp(logsumexp over valid (i,o) of
                     dp[pred_i, pred_o, src] + log_w + bp[i, o, dst] - ll)

    where pred depends on transition type:
      match:  (i-1, o-1) -> (i, o), valid when input[i-1]==in_tok, output[o-1]==out_tok
      insert: (i-1, o)   -> (i, o), valid when input[i-1]==in_tok
      delete: (i, o-1)   -> (i, o), valid when output[o-1]==out_tok
      silent: (i, o)     -> (i, o), always valid

    Vectorized via jax.vmap over all T transitions simultaneously.
    Position sums use broadcasting + masked logsumexp.
    """
    Li = input_seq.shape[0]
    Lo = output_seq.shape[0]

    def count_one(src, dst, in_tok, out_tok, log_w):
        """Compute expected count for a single transition."""
        is_match = (in_tok > 0) & (out_tok > 0)
        is_ins = (in_tok > 0) & (out_tok == 0)
        is_del = (in_tok == 0) & (out_tok > 0)

        # --- Match: dp[i-1, o-1, src] + bp[i, o, dst] for i=1..Li, o=1..Lo ---
        # where input_seq[i-1]==in_tok AND output_seq[o-1]==out_tok
        in_match = (input_seq == in_tok)    # (Li,)
        out_match = (output_seq == out_tok)  # (Lo,)
        match_valid = in_match[:, None] & out_match[None, :]  # (Li, Lo)
        match_contrib = dp[:-1, :-1, src] + bp[1:, 1:, dst]   # (Li, Lo)
        match_contrib = jnp.where(match_valid, match_contrib, NEG_INF)
        match_total = _safe_logsumexp(match_contrib)

        # --- Insert: dp[i-1, o, src] + bp[i, o, dst] for i=1..Li, o=0..Lo ---
        # where input_seq[i-1]==in_tok
        ins_valid = in_match[:, None] & jnp.ones((1, Lo + 1), dtype=bool)  # (Li, Lo+1)
        ins_contrib = dp[:-1, :, src] + bp[1:, :, dst]  # (Li, Lo+1)
        ins_contrib = jnp.where(ins_valid, ins_contrib, NEG_INF)
        ins_total = _safe_logsumexp(ins_contrib)

        # --- Delete: dp[i, o-1, src] + bp[i, o, dst] for i=0..Li, o=1..Lo ---
        # where output_seq[o-1]==out_tok
        del_valid = jnp.ones((Li + 1, 1), dtype=bool) & out_match[None, :]  # (Li+1, Lo)
        del_contrib = dp[:, :-1, src] + bp[:, 1:, dst]  # (Li+1, Lo)
        del_contrib = jnp.where(del_valid, del_contrib, NEG_INF)
        del_total = _safe_logsumexp(del_contrib)

        # --- Silent: dp[i, o, src] + bp[i, o, dst] for all i, o ---
        silent_contrib = dp[:, :, src] + bp[:, :, dst]  # (Li+1, Lo+1)
        silent_total = _safe_logsumexp(silent_contrib)

        # Select based on transition type
        total = jnp.where(is_match, match_total,
                jnp.where(is_ins, ins_total,
                jnp.where(is_del, del_total,
                          silent_total)))

        return total + log_w - ll

    log_counts = jax.vmap(count_one)(
        tm.src, tm.dst, tm.in_tok, tm.out_tok, tm.log_w)
    return jnp.exp(log_counts)


def _safe_logsumexp(x):
    """Logsumexp over all elements, safe for all-NEG_INF inputs."""
    flat = x.ravel()
    m = jnp.max(flat)
    # If all NEG_INF, return NEG_INF
    return jnp.where(m > NEG_INF + 1,
                     m + jnp.log(jnp.sum(jnp.exp(flat - m))),
                     NEG_INF)


# ============================================================
# 1D forward-backward
# ============================================================

def forward_backward_1d(tm: TransMachine,
                        input_seq: jnp.ndarray | None = None,
                        output_seq: jnp.ndarray | None = None, *,
                        strategy: str = 'auto') -> tuple[float, jnp.ndarray]:
    """Compute log-likelihood and expected counts for 1D (generator/recognizer).

    Args:
        tm: TransMachine
        input_seq: (Li,) tokens or None (generator)
        output_seq: (Lo,) tokens or None (recognizer)
        strategy: 'simple', 'optimal', or 'auto'

    Returns:
        (log_likelihood, counts) where counts is (T,) expected count
        per transition.
    """
    from .dp_1d import forward_1d, backward_1d
    from ..seq import wrap_seq

    S = tm.n_states
    semiring = LOGSUMEXP

    if input_seq is None:
        seq = wrap_seq(output_seq, tm.n_out)
        is_input = False
    else:
        seq = wrap_seq(input_seq, tm.n_in)
        is_input = True

    L = len(seq)

    # Forward: build full (L+1, S) matrix
    # Use backward_1d which returns (L+1, S), then also get forward values
    bp = backward_1d(tm, input_seq, output_seq, semiring, strategy=strategy)

    # For 1D forward matrix, run backward on reversed problem or build directly
    # Build forward matrix via scan
    fwd = _forward_1d_matrix(tm, input_seq, output_seq, semiring, strategy=strategy)
    ll = fwd[-1, S - 1]

    # Vectorized 1D counts
    n_tokens = tm.n_in if is_input else tm.n_out
    tok_seq = seq.tokens if hasattr(seq, 'tokens') else None

    def count_one(src, dst, in_tok, out_tok, log_w):
        is_emit = (in_tok > 0) if is_input else (out_tok > 0)
        is_silent = (in_tok == 0) & (out_tok == 0)
        tok = in_tok if is_input else out_tok

        # Emit: fwd[p-1, src] + bp[p, dst] for p=1..L where seq[p-1]==tok
        if tok_seq is not None:
            tok_match = (tok_seq == tok)  # (L,)
        else:
            tok_match = jnp.ones(L, dtype=bool)
        emit_contrib = fwd[:-1, src] + bp[1:, dst]  # (L,)
        emit_contrib = jnp.where(tok_match, emit_contrib, NEG_INF)
        emit_total = _safe_logsumexp(emit_contrib)

        # Silent: fwd[p, src] + bp[p, dst] for p=0..L
        silent_contrib = fwd[:, src] + bp[:, dst]  # (L+1,)
        silent_total = _safe_logsumexp(silent_contrib)

        total = jnp.where(is_emit & ~is_silent, emit_total, silent_total)
        return total + log_w - ll

    log_counts = jax.vmap(count_one)(
        tm.src, tm.dst, tm.in_tok, tm.out_tok, tm.log_w)
    return ll, jnp.exp(log_counts)


def _forward_1d_matrix(tm, input_seq, output_seq, semiring, *, strategy='auto'):
    """Build full (L+1, S) forward matrix for 1D DP."""
    from .kernel import propagate_silent, emit_step_forward, emission_weights_1d
    from ..seq import wrap_seq

    S = tm.n_states

    if input_seq is None:
        seq = wrap_seq(output_seq, tm.n_out)
        is_input = False
    else:
        seq = wrap_seq(input_seq, tm.n_in)
        is_input = True

    L = len(seq)
    n_tokens = tm.n_in if is_input else tm.n_out
    mask = tm.emit_in_mask if is_input else tm.emit_out_mask

    cell = jnp.full(S, NEG_INF).at[0].set(0.0)
    cell = propagate_silent(cell, tm, semiring)

    if L == 0:
        return cell[None, :]  # (1, S)

    emit = seq.emission_weights(n_tokens)

    def scan_fn(cell, e):
        ew = emission_weights_1d(tm, e, is_input)
        new_cell = emit_step_forward(
            jnp.full(S, NEG_INF), cell, tm, mask, ew, semiring)
        new_cell = propagate_silent(new_cell, tm, semiring)
        return new_cell, new_cell

    _, all_cells = jax.lax.scan(scan_fn, cell, emit)
    # all_cells: (L, S)
    fwd = jnp.concatenate([cell[None, :], all_cells], axis=0)
    return fwd  # (L+1, S)
