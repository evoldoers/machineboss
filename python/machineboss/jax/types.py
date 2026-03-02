"""JAX array types and machine representation for DP algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..eval import EvaluatedMachine, EvaluatedTransition

NEG_INF = -1e38  # log(0), used as -infinity in log-space


@dataclass
class JAXMachine:
    """Sparse + optional dense representation of an evaluated machine in JAX arrays."""

    # Sparse (COO) representation
    log_weights: jnp.ndarray    # (T,) transition log-weights
    src_states: jnp.ndarray     # (T,) source state indices
    dst_states: jnp.ndarray     # (T,) destination state indices
    in_tokens: jnp.ndarray      # (T,) input tokens (0=empty)
    out_tokens: jnp.ndarray     # (T,) output tokens (0=empty)
    n_states: int
    n_input_tokens: int         # including empty token at index 0
    n_output_tokens: int        # including empty token at index 0

    # Dense representation: log_trans[in_tok, out_tok, src, dst]
    # Only built for small machines
    log_trans: jnp.ndarray | None = None

    @classmethod
    def from_evaluated(cls, em: EvaluatedMachine, dense_threshold: int = 100) -> JAXMachine:
        """Convert an EvaluatedMachine to JAX arrays.

        Args:
            em: Evaluated machine with log-weights
            dense_threshold: Build dense tensor if n_states <= this value
        """
        T = len(em.transitions)

        log_weights = np.array([t.log_weight for t in em.transitions], dtype=np.float32)
        src_states = np.array([t.src for t in em.transitions], dtype=np.int32)
        dst_states = np.array([t.dst for t in em.transitions], dtype=np.int32)
        in_tokens = np.array([t.in_tok for t in em.transitions], dtype=np.int32)
        out_tokens = np.array([t.out_tok for t in em.transitions], dtype=np.int32)

        n_in = len(em.input_tokens)
        n_out = len(em.output_tokens)
        S = em.n_states

        # Build dense tensor for small machines
        dense = None
        if S <= dense_threshold:
            dense = np.full((n_in, n_out, S, S), NEG_INF, dtype=np.float32)
            for t in em.transitions:
                # Use logsumexp for multiple transitions with same (in, out, src, dst)
                cur = dense[t.in_tok, t.out_tok, t.src, t.dst]
                if cur == NEG_INF:
                    dense[t.in_tok, t.out_tok, t.src, t.dst] = t.log_weight
                else:
                    dense[t.in_tok, t.out_tok, t.src, t.dst] = np.logaddexp(cur, t.log_weight)
            dense = jnp.array(dense)

        return cls(
            log_weights=jnp.array(log_weights),
            src_states=jnp.array(src_states),
            dst_states=jnp.array(dst_states),
            in_tokens=jnp.array(in_tokens),
            out_tokens=jnp.array(out_tokens),
            n_states=S,
            n_input_tokens=n_in,
            n_output_tokens=n_out,
            log_trans=dense,
        )
