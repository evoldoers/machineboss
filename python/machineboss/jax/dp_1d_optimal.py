"""1D DP engine using jax.lax.associative_scan (OPTIMAL strategy).

Key insight: 1D Forward is a prefix product of S x S "transfer matrices"
in the log-semiring. This gives O(log L) parallel depth.

Steps:
1. Compute silent closure S* = I + S + S^2 + ...
2. Per-position transfer matrix: M_p = S* @ Emit_p @ S*
3. Associative scan: prefix = associative_scan(mat_mul, M)
4. Extract Forward: fwd[p, dst] = reduce_src(init[src] + prefix[p, src, dst])

Padding: when length < len(seq), positions beyond length use identity
transfer matrices, so the prefix product is unchanged beyond that point.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import JAXMachine, NEG_INF
from .semiring import LogSemiring
from .seq import TokenSeq, PSWMSeq, wrap_seq


def _silent_closure(silent: jnp.ndarray, semiring: LogSemiring,
                    max_iter: int = 100) -> jnp.ndarray:
    """Compute Kleene star of silent transition matrix: S* = I + S + S^2 + ...

    For acyclic machines, terminates in at most S-1 steps.

    Args:
        silent: (S, S) log-weight matrix for silent transitions
        semiring: LogSemiring
    Returns:
        (S, S) closure matrix in log-space.
    """
    S = silent.shape[0]
    identity = _log_identity(S)

    result = identity

    def body_fn(carry):
        prev_result, power, _ = carry
        next_power = semiring.mat_mul(power, silent)
        new_result = semiring.plus(prev_result, next_power)
        return new_result, next_power, prev_result

    def cond_fn(carry):
        new_result, _, prev_result = carry
        return jnp.any(jnp.abs(new_result - prev_result) > 1e-10)

    init = (result, identity, jnp.full_like(result, NEG_INF))
    result, _, _ = jax.lax.while_loop(cond_fn, body_fn, init)
    return result


def _log_identity(S: int) -> jnp.ndarray:
    """Identity matrix in log-space: 0 on diagonal, NEG_INF elsewhere."""
    identity = jnp.full((S, S), NEG_INF)
    return identity.at[jnp.arange(S), jnp.arange(S)].set(0.0)


def _build_emit_matrix(log_trans: jnp.ndarray, emission: jnp.ndarray,
                       is_input: bool, semiring: LogSemiring) -> jnp.ndarray:
    """Build emission transition matrix for one position.

    Args:
        log_trans: (n_in, n_out, S, S) full transition tensor
        emission: (n_tokens,) emission log-probabilities for this position
        is_input: True if the active dimension is input, False if output
        semiring: LogSemiring
    Returns:
        (S, S) combined emit matrix.
    """
    if is_input:
        trans_slice = log_trans[:, 0, :, :]  # (n_in, S, S)
    else:
        trans_slice = log_trans[0, :, :, :]  # (n_out, S, S)

    weighted = emission[:, None, None] + trans_slice
    return semiring.reduce(weighted, axis=0)  # (S, S)


def forward_1d_optimal(machine: JAXMachine, input_seq, output_seq,
                       semiring: LogSemiring, *,
                       length: int | None = None) -> float:
    """1D Forward/Viterbi using associative_scan for O(log L) depth.

    Args:
        machine: JAXMachine with dense log_trans
        input_seq: token sequence or None (generator)
        output_seq: token sequence or None (recognizer)
        semiring: LOGSUMEXP or MAXPLUS
        length: real sequence length (for padded sequences). If None, uses len(seq).
    Returns:
        Log-likelihood or Viterbi score (scalar).
    """
    assert machine.log_trans is not None
    S = machine.n_states
    log_trans = machine.log_trans

    if input_seq is None:
        seq = wrap_seq(output_seq, machine.n_output_tokens)
        is_input = False
        n_tokens = machine.n_output_tokens
    else:
        seq = wrap_seq(input_seq, machine.n_input_tokens)
        is_input = True
        n_tokens = machine.n_input_tokens

    L = len(seq)
    real_L = length if length is not None else L
    if L == 0:
        silent = log_trans[0, 0]
        closure = _silent_closure(silent, semiring)
        return closure[0, S - 1]

    emit = seq.emission_weights(n_tokens)  # (L, n_tokens)
    is_pswm = isinstance(seq, PSWMSeq)

    silent = log_trans[0, 0]
    closure = _silent_closure(silent, semiring)
    identity = _log_identity(S)

    def make_transfer(emission_p):
        if is_pswm:
            e = emission_p.at[0].set(NEG_INF)
        else:
            e = emission_p
        emit_mat = _build_emit_matrix(log_trans, e, is_input, semiring)
        return semiring.mat_mul(closure, semiring.mat_mul(emit_mat, closure))

    transfers = jax.vmap(make_transfer)(emit)  # (L, S, S)

    # Mask padded positions: replace with identity so they don't affect the product
    if length is not None and length < L:
        mask = jnp.arange(L) < length  # (L,) bool
        transfers = jnp.where(mask[:, None, None], transfers, identity[None, :, :])

    def combine(a, b):
        return semiring.mat_mul(a, b)

    prefix = jax.lax.associative_scan(combine, transfers, axis=0)  # (L, S, S)

    init = jnp.full(S, NEG_INF)
    init = init.at[0].set(0.0)
    init_closed = semiring.reduce(init[:, None] + closure, axis=0)

    # Read result at real_L - 1 (not L - 1) for padded sequences
    final = semiring.reduce(init_closed[:, None] + prefix[real_L - 1], axis=0)

    return final[S - 1]


def backward_1d_optimal(machine: JAXMachine, input_seq, output_seq,
                        semiring: LogSemiring, *,
                        length: int | None = None) -> jnp.ndarray:
    """1D Backward using associative_scan with reverse=True.

    Args:
        length: real sequence length (for padded sequences). If None, uses len(seq).
    Returns:
        Backward vector array of shape (real_L+1, S).
    """
    assert machine.log_trans is not None
    S = machine.n_states
    log_trans = machine.log_trans

    if input_seq is None:
        seq = wrap_seq(output_seq, machine.n_output_tokens)
        is_input = False
        n_tokens = machine.n_output_tokens
    else:
        seq = wrap_seq(input_seq, machine.n_input_tokens)
        is_input = True
        n_tokens = machine.n_input_tokens

    L = len(seq)
    real_L = length if length is not None else L

    silent = log_trans[0, 0]
    closure = _silent_closure(silent, semiring)
    is_pswm = isinstance(seq, PSWMSeq)

    if L == 0:
        bp = jnp.full((1, S), NEG_INF)
        bp = bp.at[0, S - 1].set(0.0)
        term = jnp.full(S, NEG_INF)
        term = term.at[S - 1].set(0.0)
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
        emit_mat = _build_emit_matrix(log_trans, e, is_input, semiring)
        return semiring.mat_mul(closure, semiring.mat_mul(emit_mat, closure))

    transfers = jax.vmap(make_transfer)(emit)  # (L, S, S)

    # Mask padded positions with identity
    if length is not None and length < L:
        mask = jnp.arange(L) < length
        transfers = jnp.where(mask[:, None, None], transfers, identity[None, :, :])

    # Note: JAX's associative_scan(reverse=True) calls fn(suffix[i+1], xs[i]),
    # so we swap arguments to get suffix[i] = xs[i] @ suffix[i+1].
    suffix = jax.lax.associative_scan(
        lambda a, b: semiring.mat_mul(b, a),
        transfers,
        axis=0,
        reverse=True,
    )  # (L, S, S)

    term = jnp.full(S, NEG_INF)
    term = term.at[S - 1].set(0.0)
    term_closed = semiring.reduce(closure + term[None, :], axis=1)

    # Only compute backward for real positions
    bp_emit = semiring.reduce(suffix[:real_L] + term_closed[None, None, :], axis=2)

    bp = jnp.full((real_L + 1, S), NEG_INF)
    bp = bp.at[:real_L].set(bp_emit)
    bp = bp.at[real_L].set(term_closed)

    return bp
