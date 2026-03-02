"""Forward-Backward with custom VJP for differentiable log-likelihood.

Provides jax.custom_vjp wrapper so that jax.grad(log_likelihood) works
and computes parameter gradients via expected transition counts.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .types import JAXMachine, NEG_INF
from .forward import log_forward_dense, _propagate_silent
from .backward import log_backward_dense


def compute_counts_dense(machine: JAXMachine,
                         input_seq: jnp.ndarray,
                         output_seq: jnp.ndarray,
                         forward_matrix: jnp.ndarray,
                         backward_matrix: jnp.ndarray,
                         log_likelihood: float) -> jnp.ndarray:
    """Compute expected transition counts from forward and backward matrices.

    Args:
        machine: JAXMachine with dense log_trans
        input_seq, output_seq: token sequences
        forward_matrix, backward_matrix: DP matrices
        log_likelihood: total log-likelihood

    Returns:
        Expected counts per transition: (T,) array aligned with machine's sparse transitions.
    """
    S = machine.n_states
    Li = len(input_seq)
    Lo = len(output_seq)

    # For each transition t: (src, dst, in_tok, out_tok, log_w)
    # count = sum over all (i,o) where transition fires:
    #   exp(fwd[i', o', src] + log_w + bwd[i, o, dst] - ll)
    # where (i', o') is the predecessor cell depending on transition type
    T = len(machine.log_weights)
    counts = jnp.zeros(T)

    for t_idx in range(T):
        src = int(machine.src_states[t_idx])
        dst = int(machine.dst_states[t_idx])
        in_tok = int(machine.in_tokens[t_idx])
        out_tok = int(machine.out_tokens[t_idx])
        log_w = machine.log_weights[t_idx]

        count = NEG_INF
        for i in range(Li + 1):
            for o in range(Lo + 1):
                # Determine predecessor cell based on transition type
                if in_tok > 0 and out_tok > 0:
                    # Match: predecessor is (i-1, o-1)
                    if i > 0 and o > 0 and int(input_seq[i-1]) == in_tok and int(output_seq[o-1]) == out_tok:
                        c = forward_matrix[i-1, o-1, src] + log_w + backward_matrix[i, o, dst]
                        count = jnp.logaddexp(count, c)
                elif in_tok > 0:
                    # Insert: predecessor is (i-1, o)
                    if i > 0 and int(input_seq[i-1]) == in_tok:
                        c = forward_matrix[i-1, o, src] + log_w + backward_matrix[i, o, dst]
                        count = jnp.logaddexp(count, c)
                elif out_tok > 0:
                    # Delete: predecessor is (i, o-1)
                    if o > 0 and int(output_seq[o-1]) == out_tok:
                        c = forward_matrix[i, o-1, src] + log_w + backward_matrix[i, o, dst]
                        count = jnp.logaddexp(count, c)
                else:
                    # Silent: predecessor is (i, o)
                    c = forward_matrix[i, o, src] + log_w + backward_matrix[i, o, dst]
                    count = jnp.logaddexp(count, c)

        counts = counts.at[t_idx].set(count - log_likelihood)

    return jnp.exp(counts)


def log_likelihood_with_counts(machine: JAXMachine,
                               input_seq: jnp.ndarray,
                               output_seq: jnp.ndarray):
    """Compute log-likelihood and expected transition counts.

    Returns:
        (log_likelihood, counts) tuple.
    """
    S = machine.n_states
    Li = len(input_seq)
    Lo = len(output_seq)

    # Forward pass
    dp = jnp.full((Li + 1, Lo + 1, S), NEG_INF)
    dp = dp.at[0, 0, 0].set(0.0)
    silent = machine.log_trans[0, 0]
    dp = dp.at[0, 0].set(_propagate_silent(dp[0, 0], silent))

    for i in range(Li + 1):
        for o in range(Lo + 1):
            if i == 0 and o == 0:
                continue
            cell = dp[i, o]
            if i > 0 and o > 0:
                in_tok = input_seq[i-1]
                out_tok = output_seq[o-1]
                trans = machine.log_trans[in_tok, out_tok]
                prev = dp[i-1, o-1]
                incoming = prev[:, None] + trans
                update = jax.nn.logsumexp(incoming, axis=0)
                cell = jnp.logaddexp(cell, update)
            if i > 0:
                in_tok = input_seq[i-1]
                trans = machine.log_trans[in_tok, 0]
                prev = dp[i-1, o]
                incoming = prev[:, None] + trans
                update = jax.nn.logsumexp(incoming, axis=0)
                cell = jnp.logaddexp(cell, update)
            if o > 0:
                out_tok = output_seq[o-1]
                trans = machine.log_trans[0, out_tok]
                prev = dp[i, o-1]
                incoming = prev[:, None] + trans
                update = jax.nn.logsumexp(incoming, axis=0)
                cell = jnp.logaddexp(cell, update)
            cell = _propagate_silent(cell, silent)
            dp = dp.at[i, o].set(cell)

    ll = dp[Li, Lo, S - 1]

    # Backward pass
    bp = log_backward_dense(machine, input_seq, output_seq)

    # Counts
    counts = compute_counts_dense(machine, input_seq, output_seq, dp, bp, ll)

    return ll, counts
