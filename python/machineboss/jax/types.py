"""JAX array types and machine representation for DP algorithms."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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

    # Token lists for reverse serialization (index 0 = empty token)
    input_token_list: list[str] | None = None
    output_token_list: list[str] | None = None

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
            input_token_list=list(em.input_tokens),
            output_token_list=list(em.output_tokens),
        )

    def to_machine(self):
        """Reconstruct a Machine from sparse JAX arrays and token lists."""
        from ..machine import Machine, MachineState, MachineTransition

        if self.input_token_list is None or self.output_token_list is None:
            raise ValueError("Token lists not available; cannot reconstruct Machine")

        states = [MachineState() for _ in range(self.n_states)]
        lw = np.asarray(self.log_weights)
        src = np.asarray(self.src_states)
        dst = np.asarray(self.dst_states)
        itk = np.asarray(self.in_tokens)
        otk = np.asarray(self.out_tokens)

        for i in range(len(lw)):
            if lw[i] < -1e30:
                continue
            w = math.exp(float(lw[i]))
            states[int(src[i])].trans.append(MachineTransition(
                dest=int(dst[i]),
                input=self.input_token_list[int(itk[i])] or None,
                output=self.output_token_list[int(otk[i])] or None,
                weight=1 if abs(w - 1) < 1e-15 else w,
            ))
        for s in states:
            s.trans.sort(key=lambda t: (t.input or '', t.output or '', t.dest))
        return Machine(state=states)

    def has_input(self) -> bool:
        """True if the machine has input transitions (n_input_tokens > 1)."""
        return self.n_input_tokens > 1

    def has_output(self) -> bool:
        """True if the machine has output transitions (n_output_tokens > 1)."""
        return self.n_output_tokens > 1

    def is_transducer(self) -> bool:
        """True if the machine has both input and output alphabets."""
        return self.has_input() and self.has_output()

    def is_generator(self) -> bool:
        """True if the machine has output but no input (generator)."""
        return (not self.has_input()) and self.has_output()

    def is_recognizer(self) -> bool:
        """True if the machine has input but no output (recognizer)."""
        return self.has_input() and (not self.has_output())

    def machine_type(self) -> str:
        """Return machine type string: 'transducer', 'generator', 'recognizer', or 'null'."""
        if self.is_transducer():
            return 'transducer'
        elif self.is_generator():
            return 'generator'
        elif self.is_recognizer():
            return 'recognizer'
        else:
            return 'null'
