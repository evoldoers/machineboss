"""TransMachine: transition-centric WFST representation for JAX DP.

The Machine IS the list of Transitions. All fields are parallel arrays
of shape (T,) where T is the number of transitions. Pre-built boolean
masks partition transitions into silent, emit-in, emit-out, emit-both.

Registered as a JAX pytree: arrays are children, metadata is aux.
Fully compatible with jit, grad, vmap.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from ..types import JAXMachine, NEG_INF
from ...eval import EvaluatedMachine
from ...machine import Machine


class TransMachine:
    """Transition-centric WFST representation.

    Core transition arrays (all shape (T,), pytree leaves):
        src, dst: int32 state indices
        in_tok, out_tok: int32 token indices (0 = silent)
        log_w: float32 transition log-weights

    Pre-built boolean masks (shape (T,), pytree leaves):
        silent_mask: in_tok==0 & out_tok==0
        emit_in_mask: in_tok>0 & out_tok==0
        emit_out_mask: in_tok==0 & out_tok>0
        emit_both_mask: in_tok>0 & out_tok>0

    Static metadata (pytree aux, not traced):
        n_states, n_in, n_out: int
        input_tokens, output_tokens: tuple[str, ...]
    """

    __slots__ = (
        'src', 'dst', 'in_tok', 'out_tok', 'log_w',
        'silent_mask', 'emit_in_mask', 'emit_out_mask', 'emit_both_mask',
        'n_states', 'n_in', 'n_out', 'input_tokens', 'output_tokens',
    )

    def __init__(self, src, dst, in_tok, out_tok, log_w,
                 silent_mask, emit_in_mask, emit_out_mask, emit_both_mask,
                 n_states, n_in, n_out, input_tokens, output_tokens):
        self.src = src
        self.dst = dst
        self.in_tok = in_tok
        self.out_tok = out_tok
        self.log_w = log_w
        self.silent_mask = silent_mask
        self.emit_in_mask = emit_in_mask
        self.emit_out_mask = emit_out_mask
        self.emit_both_mask = emit_both_mask
        self.n_states = n_states
        self.n_in = n_in
        self.n_out = n_out
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def tree_flatten(self):
        children = (
            self.src, self.dst, self.in_tok, self.out_tok, self.log_w,
            self.silent_mask, self.emit_in_mask, self.emit_out_mask,
            self.emit_both_mask,
        )
        aux = (self.n_states, self.n_in, self.n_out,
               self.input_tokens, self.output_tokens)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (src, dst, in_tok, out_tok, log_w,
         silent_mask, emit_in_mask, emit_out_mask, emit_both_mask) = children
        n_states, n_in, n_out, input_tokens, output_tokens = aux
        return cls(src, dst, in_tok, out_tok, log_w,
                   silent_mask, emit_in_mask, emit_out_mask, emit_both_mask,
                   n_states, n_in, n_out, input_tokens, output_tokens)

    @classmethod
    def from_evaluated(cls, em: EvaluatedMachine) -> TransMachine:
        """Convert an EvaluatedMachine to TransMachine."""
        T = len(em.transitions)
        src = np.array([t.src for t in em.transitions], dtype=np.int32)
        dst = np.array([t.dst for t in em.transitions], dtype=np.int32)
        in_tok = np.array([t.in_tok for t in em.transitions], dtype=np.int32)
        out_tok = np.array([t.out_tok for t in em.transitions], dtype=np.int32)
        log_w = np.array([t.log_weight for t in em.transitions], dtype=np.float32)

        silent_mask = (in_tok == 0) & (out_tok == 0)
        emit_in_mask = (in_tok > 0) & (out_tok == 0)
        emit_out_mask = (in_tok == 0) & (out_tok > 0)
        emit_both_mask = (in_tok > 0) & (out_tok > 0)

        return cls(
            src=jnp.array(src), dst=jnp.array(dst),
            in_tok=jnp.array(in_tok), out_tok=jnp.array(out_tok),
            log_w=jnp.array(log_w),
            silent_mask=jnp.array(silent_mask),
            emit_in_mask=jnp.array(emit_in_mask),
            emit_out_mask=jnp.array(emit_out_mask),
            emit_both_mask=jnp.array(emit_both_mask),
            n_states=em.n_states,
            n_in=len(em.input_tokens),
            n_out=len(em.output_tokens),
            input_tokens=tuple(em.input_tokens),
            output_tokens=tuple(em.output_tokens),
        )

    @classmethod
    def from_machine(cls, machine: Machine,
                     params: dict[str, float] | None = None) -> TransMachine:
        """Convert a Machine to TransMachine (evaluates weights)."""
        em = EvaluatedMachine.from_machine(machine, params)
        return cls.from_evaluated(em)

    @classmethod
    def from_jax_machine(cls, jm: JAXMachine) -> TransMachine:
        """Convert a JAXMachine to TransMachine."""
        in_tok = jm.in_tokens
        out_tok = jm.out_tokens
        silent_mask = (in_tok == 0) & (out_tok == 0)
        emit_in_mask = (in_tok > 0) & (out_tok == 0)
        emit_out_mask = (in_tok == 0) & (out_tok > 0)
        emit_both_mask = (in_tok > 0) & (out_tok > 0)

        return cls(
            src=jm.src_states, dst=jm.dst_states,
            in_tok=in_tok, out_tok=out_tok,
            log_w=jm.log_weights,
            silent_mask=silent_mask,
            emit_in_mask=emit_in_mask,
            emit_out_mask=emit_out_mask,
            emit_both_mask=emit_both_mask,
            n_states=jm.n_states,
            n_in=jm.n_input_tokens,
            n_out=jm.n_output_tokens,
            input_tokens=tuple(jm.input_token_list) if jm.input_token_list else (),
            output_tokens=tuple(jm.output_token_list) if jm.output_token_list else (),
        )

    def to_jax_machine(self, dense_threshold: int = 100) -> JAXMachine:
        """Convert to JAXMachine."""
        S = self.n_states
        n_in = self.n_in
        n_out = self.n_out

        dense = None
        if S <= dense_threshold:
            dense = np.full((n_in, n_out, S, S), NEG_INF, dtype=np.float32)
            lw = np.asarray(self.log_w)
            src = np.asarray(self.src)
            dst = np.asarray(self.dst)
            itk = np.asarray(self.in_tok)
            otk = np.asarray(self.out_tok)
            for i in range(len(lw)):
                cur = dense[itk[i], otk[i], src[i], dst[i]]
                if cur == NEG_INF:
                    dense[itk[i], otk[i], src[i], dst[i]] = lw[i]
                else:
                    dense[itk[i], otk[i], src[i], dst[i]] = np.logaddexp(cur, lw[i])
            dense = jnp.array(dense)

        return JAXMachine(
            log_weights=self.log_w,
            src_states=self.src,
            dst_states=self.dst,
            in_tokens=self.in_tok,
            out_tokens=self.out_tok,
            n_states=S,
            n_input_tokens=n_in,
            n_output_tokens=n_out,
            log_trans=dense,
            input_token_list=list(self.input_tokens) if self.input_tokens else None,
            output_token_list=list(self.output_tokens) if self.output_tokens else None,
        )

    def to_machine(self) -> Machine:
        """Reconstruct a Machine from transition arrays."""
        from ...machine import MachineState, MachineTransition

        if not self.input_tokens or not self.output_tokens:
            raise ValueError("Token lists not available; cannot reconstruct Machine")

        states = [MachineState() for _ in range(self.n_states)]
        lw = np.asarray(self.log_w)
        src = np.asarray(self.src)
        dst = np.asarray(self.dst)
        itk = np.asarray(self.in_tok)
        otk = np.asarray(self.out_tok)

        for i in range(len(lw)):
            if lw[i] < -1e30:
                continue
            w = math.exp(float(lw[i]))
            states[int(src[i])].trans.append(MachineTransition(
                dest=int(dst[i]),
                input=self.input_tokens[int(itk[i])] or None,
                output=self.output_tokens[int(otk[i])] or None,
                weight=1 if abs(w - 1) < 1e-15 else w,
            ))
        for s in states:
            s.trans.sort(key=lambda t: (t.input or '', t.output or '', t.dest))
        return Machine(state=states)

    def has_input(self) -> bool:
        return self.n_in > 1

    def has_output(self) -> bool:
        return self.n_out > 1

    def is_transducer(self) -> bool:
        return self.has_input() and self.has_output()

    def is_generator(self) -> bool:
        return (not self.has_input()) and self.has_output()

    def is_recognizer(self) -> bool:
        return self.has_input() and (not self.has_output())

    @property
    def n_transitions(self) -> int:
        return self.src.shape[0]


jax.tree_util.register_pytree_node_class(TransMachine)
