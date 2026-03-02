"""Log-space Viterbi algorithm using JAX.

Same structure as Forward but uses max instead of logsumexp.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import JAXMachine, NEG_INF


def _propagate_silent_max(cell: jnp.ndarray, silent_trans: jnp.ndarray,
                          max_iter: int = 100) -> jnp.ndarray:
    """Propagate silent transitions using max (Viterbi) within a DP cell."""
    def body_fn(carry):
        prev, _ = carry
        incoming = prev[:, None] + silent_trans
        update = jnp.max(incoming, axis=0)
        new = jnp.maximum(cell, update)
        return new, prev

    def cond_fn(carry):
        new, prev = carry
        return jnp.any(jnp.abs(new - prev) > 1e-10)

    init = (cell, jnp.full_like(cell, NEG_INF))
    result, _ = jax.lax.while_loop(cond_fn, body_fn, init)
    return result


def log_viterbi_dense(machine: JAXMachine,
                      input_seq: jnp.ndarray,
                      output_seq: jnp.ndarray) -> float:
    """Viterbi algorithm using dense transition tensor.

    Args:
        machine: JAXMachine with dense log_trans tensor
        input_seq: (Li,) input token indices
        output_seq: (Lo,) output token indices

    Returns:
        Log-probability of most likely path (scalar).
    """
    assert machine.log_trans is not None
    S = machine.n_states
    Li = len(input_seq)
    Lo = len(output_seq)

    dp = jnp.full((Li + 1, Lo + 1, S), NEG_INF)
    dp = dp.at[0, 0, 0].set(0.0)

    silent = machine.log_trans[0, 0]

    dp = dp.at[0, 0].set(_propagate_silent_max(dp[0, 0], silent))

    for i in range(Li + 1):
        for o in range(Lo + 1):
            if i == 0 and o == 0:
                continue
            cell = dp[i, o]

            if i > 0 and o > 0:
                in_tok = input_seq[i - 1]
                out_tok = output_seq[o - 1]
                trans = machine.log_trans[in_tok, out_tok]
                prev = dp[i - 1, o - 1]
                incoming = prev[:, None] + trans
                update = jnp.max(incoming, axis=0)
                cell = jnp.maximum(cell, update)

            if i > 0:
                in_tok = input_seq[i - 1]
                trans = machine.log_trans[in_tok, 0]
                prev = dp[i - 1, o]
                incoming = prev[:, None] + trans
                update = jnp.max(incoming, axis=0)
                cell = jnp.maximum(cell, update)

            if o > 0:
                out_tok = output_seq[o - 1]
                trans = machine.log_trans[0, out_tok]
                prev = dp[i, o - 1]
                incoming = prev[:, None] + trans
                update = jnp.max(incoming, axis=0)
                cell = jnp.maximum(cell, update)

            cell = _propagate_silent_max(cell, silent)
            dp = dp.at[i, o].set(cell)

    return dp[Li, Lo, S - 1]


def log_viterbi(machine: JAXMachine,
                input_seq: jnp.ndarray,
                output_seq: jnp.ndarray) -> float:
    """Viterbi algorithm — dispatches to dense implementation."""
    if machine.log_trans is not None:
        return log_viterbi_dense(machine, input_seq, output_seq)
    raise NotImplementedError("Sparse viterbi not yet implemented")
