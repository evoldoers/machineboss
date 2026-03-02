"""Log-space Forward algorithm using JAX.

Computes log P(input, output | machine) via the Forward algorithm.
Uses the dense representation for small machines.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import JAXMachine, NEG_INF


def _propagate_silent(cell: jnp.ndarray, silent_trans: jnp.ndarray,
                      max_iter: int = 100) -> jnp.ndarray:
    """Propagate silent transitions within a DP cell until convergence.

    Args:
        cell: (S,) log-probabilities at current cell
        silent_trans: (S, S) log-weight matrix for silent transitions (in=0, out=0)
        max_iter: maximum iterations for fixed-point convergence
    Returns:
        Updated cell with silent transitions propagated.
    """
    def body_fn(carry):
        prev, _ = carry
        # For each destination state, accumulate from all source states
        # new[dst] = logsumexp(prev[src] + silent_trans[src, dst]) over src
        incoming = prev[:, None] + silent_trans  # (S, S): [src, dst]
        update = jax.nn.logsumexp(incoming, axis=0)  # (S,)
        new = jnp.logaddexp(cell, update)
        return new, prev

    def cond_fn(carry):
        new, prev = carry
        return jnp.any(jnp.abs(new - prev) > 1e-10)

    init = (cell, jnp.full_like(cell, NEG_INF))
    result, _ = jax.lax.while_loop(cond_fn, body_fn, init)
    return result


def log_forward_dense(machine: JAXMachine,
                      input_seq: jnp.ndarray,
                      output_seq: jnp.ndarray) -> float:
    """Forward algorithm using dense transition tensor.

    Args:
        machine: JAXMachine with dense log_trans tensor of shape (I, O, S, S)
        input_seq: (Li,) array of input token indices (1-based; 0=empty)
        output_seq: (Lo,) array of output token indices (1-based; 0=empty)

    Returns:
        Log-likelihood (scalar).
    """
    assert machine.log_trans is not None, "Dense representation required"
    S = machine.n_states
    Li = len(input_seq)
    Lo = len(output_seq)

    # DP matrix: dp[inPos, outPos, state] in log-space
    # inPos ranges 0..Li, outPos ranges 0..Lo
    # Initialize with -inf
    dp = jnp.full((Li + 1, Lo + 1, S), NEG_INF)

    # Start state gets probability 1 (log-prob 0) at position (0, 0)
    dp = dp.at[0, 0, 0].set(0.0)

    # Silent transitions matrix: log_trans[0, 0, src, dst]
    silent = machine.log_trans[0, 0]  # (S, S)

    # Propagate silent transitions at (0, 0)
    dp = dp.at[0, 0].set(_propagate_silent(dp[0, 0], silent))

    # Fill DP matrix in row-major order (excluding (0,0) which is already done)
    for i in range(Li + 1):
        for o in range(Lo + 1):
            if i == 0 and o == 0:
                continue
            cell = dp[i, o]

            # 1. Match: consume input[i-1] + output[o-1], from (i-1, o-1)
            if i > 0 and o > 0:
                in_tok = input_seq[i - 1]
                out_tok = output_seq[o - 1]
                trans = machine.log_trans[in_tok, out_tok]
                prev = dp[i - 1, o - 1]
                incoming = prev[:, None] + trans
                update = jax.nn.logsumexp(incoming, axis=0)
                cell = jnp.logaddexp(cell, update)

            # 2. Insert: consume input[i-1], from (i-1, o)
            if i > 0:
                in_tok = input_seq[i - 1]
                trans = machine.log_trans[in_tok, 0]
                prev = dp[i - 1, o]
                incoming = prev[:, None] + trans
                update = jax.nn.logsumexp(incoming, axis=0)
                cell = jnp.logaddexp(cell, update)

            # 3. Delete: consume output[o-1], from (i, o-1)
            if o > 0:
                out_tok = output_seq[o - 1]
                trans = machine.log_trans[0, out_tok]
                prev = dp[i, o - 1]
                incoming = prev[:, None] + trans
                update = jax.nn.logsumexp(incoming, axis=0)
                cell = jnp.logaddexp(cell, update)

            # 4. Silent transitions within this cell
            cell = _propagate_silent(cell, silent)
            dp = dp.at[i, o].set(cell)

    # Result: log-probability at end state, position (Li, Lo)
    return dp[Li, Lo, S - 1]


def log_forward(machine: JAXMachine,
                input_seq: jnp.ndarray,
                output_seq: jnp.ndarray) -> float:
    """Forward algorithm — dispatches to dense implementation.

    Args:
        machine: JAXMachine
        input_seq: (Li,) input token indices
        output_seq: (Lo,) output token indices

    Returns:
        Log-likelihood (scalar).
    """
    if machine.log_trans is not None:
        return log_forward_dense(machine, input_seq, output_seq)
    raise NotImplementedError("Sparse forward not yet implemented; use dense_threshold > n_states")
