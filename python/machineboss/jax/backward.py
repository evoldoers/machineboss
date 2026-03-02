"""Log-space Backward algorithm using JAX.

Reverse scan over the DP matrix, same 4 transition types as Forward.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import JAXMachine, NEG_INF


def _propagate_silent_backward(cell: jnp.ndarray, silent_trans: jnp.ndarray,
                                max_iter: int = 100) -> jnp.ndarray:
    """Propagate silent transitions backward: accumulate into source states."""
    def body_fn(carry):
        prev, _ = carry
        # For backward: source state accumulates from destination states
        # new[src] = logsumexp(prev[dst] + silent_trans[src, dst]) over dst
        incoming = silent_trans + prev[None, :]  # (S, S): [src, dst]
        update = jax.nn.logsumexp(incoming, axis=1)  # (S,)
        new = jnp.logaddexp(cell, update)
        return new, prev

    def cond_fn(carry):
        new, prev = carry
        return jnp.any(jnp.abs(new - prev) > 1e-10)

    init = (cell, jnp.full_like(cell, NEG_INF))
    result, _ = jax.lax.while_loop(cond_fn, body_fn, init)
    return result


def log_backward_dense(machine: JAXMachine,
                       input_seq: jnp.ndarray,
                       output_seq: jnp.ndarray) -> jnp.ndarray:
    """Backward algorithm using dense transition tensor.

    Args:
        machine: JAXMachine with dense log_trans tensor
        input_seq: (Li,) input token indices
        output_seq: (Lo,) output token indices

    Returns:
        Backward matrix of shape (Li+1, Lo+1, S).
    """
    assert machine.log_trans is not None
    S = machine.n_states
    Li = len(input_seq)
    Lo = len(output_seq)

    bp = jnp.full((Li + 1, Lo + 1, S), NEG_INF)

    # End state gets probability 1 at (Li, Lo)
    bp = bp.at[Li, Lo, S - 1].set(0.0)

    silent = machine.log_trans[0, 0]  # (S, S)

    bp = bp.at[Li, Lo].set(_propagate_silent_backward(bp[Li, Lo], silent))

    def fill_cell_backward(bp, inPos, outPos):
        cell = bp[inPos, outPos]

        # Match: transition from (inPos, outPos) consuming input[inPos] + output[outPos]
        # contributes to bp[inPos, outPos] from bp[inPos+1, outPos+1]
        def add_match(cell, bp, inPos, outPos):
            in_tok = input_seq[inPos]
            out_tok = output_seq[outPos]
            trans = machine.log_trans[in_tok, out_tok]  # (S, S): [src, dst]
            future = bp[inPos + 1, outPos + 1]
            # For each src: logsumexp(trans[src, dst] + future[dst]) over dst
            incoming = trans + future[None, :]  # (S, S)
            update = jax.nn.logsumexp(incoming, axis=1)  # (S,)
            return jnp.logaddexp(cell, update)

        cell = jax.lax.cond(
            (inPos < Li) & (outPos < Lo),
            lambda: add_match(cell, bp, inPos, outPos),
            lambda: cell,
        )

        # Insert: consume input[inPos]
        def add_insert(cell, bp, inPos, outPos):
            in_tok = input_seq[inPos]
            trans = machine.log_trans[in_tok, 0]
            future = bp[inPos + 1, outPos]
            incoming = trans + future[None, :]
            update = jax.nn.logsumexp(incoming, axis=1)
            return jnp.logaddexp(cell, update)

        cell = jax.lax.cond(
            inPos < Li,
            lambda: add_insert(cell, bp, inPos, outPos),
            lambda: cell,
        )

        # Delete: consume output[outPos]
        def add_delete(cell, bp, inPos, outPos):
            out_tok = output_seq[outPos]
            trans = machine.log_trans[0, out_tok]
            future = bp[inPos, outPos + 1]
            incoming = trans + future[None, :]
            update = jax.nn.logsumexp(incoming, axis=1)
            return jnp.logaddexp(cell, update)

        cell = jax.lax.cond(
            outPos < Lo,
            lambda: add_delete(cell, bp, inPos, outPos),
            lambda: cell,
        )

        cell = _propagate_silent_backward(cell, silent)
        return cell

    # Fill in reverse order
    for i in range(Li, -1, -1):
        for o in range(Lo, -1, -1):
            if i == Li and o == Lo:
                continue
            bp = bp.at[i, o].set(fill_cell_backward(bp, i, o))

    return bp
