"""Compile weight expressions into JAX-traceable functions.

Converts the JSON weight expression mini-language (defined in weight.py)
into JAX operations that are JIT-compilable and differentiable.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from ..machine import Machine
from ..weight import WeightExpr, params as expr_params


def compile_expr(expr: WeightExpr):
    """Compile a weight expression into a JAX-traceable function.

    Args:
        expr: JSON weight expression (number, string, or dict).

    Returns:
        Callable: param_dict -> JAX scalar
    """
    if isinstance(expr, (int, float)):
        val = jnp.float32(float(expr))
        return lambda p, _v=val: _v

    if isinstance(expr, str):
        name = expr
        return lambda p, _n=name: p[_n]

    if isinstance(expr, dict):
        if "*" in expr:
            fa = compile_expr(expr["*"][0])
            fb = compile_expr(expr["*"][1])
            return lambda p, _a=fa, _b=fb: _a(p) * _b(p)

        if "+" in expr:
            fa = compile_expr(expr["+"][0])
            fb = compile_expr(expr["+"][1])
            return lambda p, _a=fa, _b=fb: _a(p) + _b(p)

        if "-" in expr:
            fa = compile_expr(expr["-"][0])
            fb = compile_expr(expr["-"][1])
            return lambda p, _a=fa, _b=fb: _a(p) - _b(p)

        if "/" in expr:
            fa = compile_expr(expr["/"][0])
            fb = compile_expr(expr["/"][1])
            return lambda p, _a=fa, _b=fb: _a(p) / _b(p)

        if "pow" in expr:
            fa = compile_expr(expr["pow"][0])
            fb = compile_expr(expr["pow"][1])
            return lambda p, _a=fa, _b=fb: _a(p) ** _b(p)

        if "log" in expr:
            fa = compile_expr(expr["log"])
            return lambda p, _a=fa: jnp.log(_a(p))

        if "exp" in expr:
            fa = compile_expr(expr["exp"])
            return lambda p, _a=fa: jnp.exp(_a(p))

        if "not" in expr:
            fa = compile_expr(expr["not"])
            return lambda p, _a=fa: 1.0 - _a(p)

        raise ValueError(f"Unknown operator: {list(expr.keys())}")

    raise TypeError(f"Unsupported weight expression: {type(expr)}")


@dataclass
class ParameterizedMachine:
    """A machine compiled for position-dependent parameter evaluation.

    Weight expressions are pre-compiled into JAX operations. At each (i, j)
    position in the 2D DP, transition weights are computed from the
    position-specific parameter values.
    """
    n_states: int
    n_input_tokens: int   # including empty token at index 0
    n_output_tokens: int  # including empty token at index 0
    input_tokens: list[str]   # ['', 'a', 'b', ...]
    output_tokens: list[str]  # ['', '0', '1', ...]
    param_names: set[str]

    # Compiled transition structure: list of (in_tok, out_tok, src, dst, [fns])
    # where fns is a list of compiled weight functions for duplicate indices
    _grouped_transitions: list

    # Precomputed flat indices for vectorized scatter (used by build_log_trans)
    _flat_indices: jnp.ndarray = None  # (T,) int32
    _tensor_size: int = 0

    @classmethod
    def from_machine(cls, machine: Machine) -> ParameterizedMachine:
        """Compile a Machine's weight expressions for JAX evaluation.

        Args:
            machine: Machine with weight expressions (unevaluated).

        Returns:
            ParameterizedMachine ready for use with neural_log_forward etc.
        """
        in_alpha = machine.input_alphabet()
        out_alpha = machine.output_alphabet()
        in_tok_map = {sym: i + 1 for i, sym in enumerate(in_alpha)}
        out_tok_map = {sym: i + 1 for i, sym in enumerate(out_alpha)}
        n_in = len(in_alpha) + 1
        n_out = len(out_alpha) + 1
        S = machine.n_states

        # Collect and compile all transitions
        all_params = set()
        grouped = defaultdict(list)

        for src, state in enumerate(machine.state):
            for t in state.trans:
                it = in_tok_map.get(t.input, 0) if t.input else 0
                ot = out_tok_map.get(t.output, 0) if t.output else 0
                fn = compile_expr(t.weight)
                all_params |= expr_params(t.weight)
                grouped[(it, ot, src, t.dest)].append(fn)

        transitions = [
            (it, ot, src, dst, fns)
            for (it, ot, src, dst), fns in grouped.items()
        ]

        # Precompute flat indices for vectorized scatter
        tensor_size = n_in * n_out * S * S
        flat_indices = np.array([
            it * (n_out * S * S) + ot * (S * S) + src * S + dst
            for it, ot, src, dst, _ in transitions
        ], dtype=np.int32)

        return cls(
            n_states=S,
            n_input_tokens=n_in,
            n_output_tokens=n_out,
            input_tokens=[''] + in_alpha,
            output_tokens=[''] + out_alpha,
            param_names=all_params,
            _grouped_transitions=transitions,
            _flat_indices=jnp.array(flat_indices),
            _tensor_size=tensor_size,
        )

    def build_log_trans(self, param_dict: dict) -> jnp.ndarray:
        """Build log_trans[in_tok, out_tok, src, dst] from parameter values.

        Uses a single vectorized scatter for JAX grad compatibility.

        Args:
            param_dict: maps parameter names to JAX scalars.

        Returns:
            (n_in, n_out, S, S) log-weight tensor.
        """
        from .types import NEG_INF

        # Evaluate all transition weights (Python loop, unrolled by JIT)
        log_weights = []
        for _it, _ot, _src, _dst, fns in self._grouped_transitions:
            lw = jnp.log(jnp.maximum(fns[0](param_dict), 1e-45))
            for fn in fns[1:]:
                lw = jnp.logaddexp(
                    lw, jnp.log(jnp.maximum(fn(param_dict), 1e-45)))
            log_weights.append(lw)

        # Single vectorized scatter (grad-compatible)
        all_lw = jnp.stack(log_weights)
        flat = jnp.full(self._tensor_size, NEG_INF)
        flat = flat.at[self._flat_indices].set(all_lw)
        shape = (self.n_input_tokens, self.n_output_tokens,
                 self.n_states, self.n_states)
        return flat.reshape(shape)

    def tokenize_input(self, seq: list[str]) -> list[int]:
        tok_map = {sym: i for i, sym in enumerate(self.input_tokens)}
        return [tok_map[s] for s in seq]

    def tokenize_output(self, seq: list[str]) -> list[int]:
        tok_map = {sym: i for i, sym in enumerate(self.output_tokens)}
        return [tok_map[s] for s in seq]
