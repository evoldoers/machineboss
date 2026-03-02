"""Dense transition kernel for DP algorithms.

Operates on the dense log_trans[in_tok, out_tok, src, dst] tensor.
All operations are parameterized by a LogSemiring for Forward vs Viterbi.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import JAXMachine, NEG_INF
from .semiring import LogSemiring


def propagate_silent(cell: jnp.ndarray, silent: jnp.ndarray,
                     semiring: LogSemiring) -> jnp.ndarray:
    """Propagate silent transitions (in=0, out=0) within a DP cell until convergence.

    Forward direction: accumulate into destination states.

    Args:
        cell: (S,) log-probabilities at current cell
        silent: (S, S) log-weight matrix for silent transitions [src, dst]
        semiring: LogSemiring (LOGSUMEXP or MAXPLUS)
    """
    def body_fn(carry):
        prev, _ = carry
        incoming = prev[:, None] + silent  # (S, S): [src, dst]
        update = semiring.reduce(incoming, axis=0)  # (S,)
        new = semiring.plus(cell, update)
        return new, prev

    def cond_fn(carry):
        new, prev = carry
        return jnp.any(jnp.abs(new - prev) > 1e-10)

    init = (cell, jnp.full_like(cell, NEG_INF))
    result, _ = jax.lax.while_loop(cond_fn, body_fn, init)
    return result


def propagate_silent_backward(cell: jnp.ndarray, silent: jnp.ndarray,
                               semiring: LogSemiring) -> jnp.ndarray:
    """Propagate silent transitions backward: accumulate into source states.

    Args:
        cell: (S,) log-probabilities at current cell
        silent: (S, S) log-weight matrix for silent transitions [src, dst]
        semiring: LogSemiring
    """
    def body_fn(carry):
        prev, _ = carry
        incoming = silent + prev[None, :]  # (S, S): [src, dst]
        update = semiring.reduce(incoming, axis=1)  # (S,)
        new = semiring.plus(cell, update)
        return new, prev

    def cond_fn(carry):
        new, prev = carry
        return jnp.any(jnp.abs(new - prev) > 1e-10)

    init = (cell, jnp.full_like(cell, NEG_INF))
    result, _ = jax.lax.while_loop(cond_fn, body_fn, init)
    return result


def emit_step_forward(cell: jnp.ndarray, prev: jnp.ndarray,
                      trans: jnp.ndarray, semiring: LogSemiring) -> jnp.ndarray:
    """Apply emitting transitions from prev cell to current cell (Forward direction).

    Args:
        cell: (S,) current cell being filled
        prev: (S,) predecessor cell
        trans: (S, S) transition matrix [src, dst] for the relevant (in_tok, out_tok)
        semiring: LogSemiring
    Returns:
        Updated cell.
    """
    incoming = prev[:, None] + trans  # (S, S)
    update = semiring.reduce(incoming, axis=0)  # (S,)
    return semiring.plus(cell, update)


def emit_step_backward(cell: jnp.ndarray, future: jnp.ndarray,
                        trans: jnp.ndarray, semiring: LogSemiring) -> jnp.ndarray:
    """Apply emitting transitions from future cell to current cell (Backward direction).

    Args:
        cell: (S,) current cell being filled
        future: (S,) successor cell
        trans: (S, S) transition matrix [src, dst]
        semiring: LogSemiring
    Returns:
        Updated cell.
    """
    incoming = trans + future[None, :]  # (S, S): [src, dst]
    update = semiring.reduce(incoming, axis=1)  # (S,)
    return semiring.plus(cell, update)


def weighted_trans_matrix(log_trans: jnp.ndarray, emission: jnp.ndarray,
                          semiring: LogSemiring) -> jnp.ndarray:
    """Compute emission-weighted transition matrix over all tokens.

    For each (src, dst), reduces over all token values weighted by emission probability.

    Args:
        log_trans: (n_tokens, S, S) transition matrices indexed by token
        emission: (n_tokens,) emission log-probabilities for each token
        semiring: LogSemiring
    Returns:
        (S, S) combined transition matrix.
    """
    # emission[:, None, None] + log_trans -> (n_tokens, S, S)
    weighted = emission[:, None, None] + log_trans
    return semiring.reduce(weighted, axis=0)  # (S, S)
