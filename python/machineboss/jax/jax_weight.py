"""Compile weight expressions into JAX-traceable functions.

Converts the JSON weight expression mini-language (defined in weight.py)
into JAX operations that are JIT-compilable and differentiable.

Supports machine-level parameter definitions (``defs``): if a parameter
referenced by a transition weight expression is not supplied by the caller
at runtime, the compiler falls back to the machine's own definition
(numeric value or weight expression). Only truly free parameters — those
with no definition anywhere — cause an error.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import jax.numpy as jnp

from ..machine import Machine
from ..weight import WeightExpr, params as expr_params


def compile_expr(expr: WeightExpr, defs: dict | None = None,
                 _compiling: frozenset[str] | None = None):
    """Compile a weight expression into a JAX-traceable function.

    Args:
        expr: JSON weight expression (number, string, or dict).
        defs: Machine parameter/function definitions (fallback values).
            Maps parameter names to numeric values or weight expressions.
        _compiling: (internal) set of parameter names currently being
            compiled, for cycle detection.

    Returns:
        Callable: param_dict -> JAX scalar.
        The returned function checks the caller's param_dict first;
        if the parameter is absent, it falls back to the compiled
        machine definition. If neither exists, a KeyError is raised.
    """
    if defs is None:
        defs = {}
    if _compiling is None:
        _compiling = frozenset()

    if isinstance(expr, (int, float)):
        val = jnp.float32(float(expr))
        return lambda p, _v=val: _v

    if isinstance(expr, bool):
        val = jnp.float32(1.0 if expr else 0.0)
        return lambda p, _v=val: _v

    if isinstance(expr, str):
        name = expr
        if name in _compiling:
            raise ValueError(f"Circular parameter definition: {name}")

        if name in defs:
            # Compile the machine definition as a fallback
            fallback_fn = compile_expr(
                defs[name], defs, _compiling | {name})
            # At runtime: caller's dict overrides, else use fallback.
            # The ``name in p`` check is a Python-level dict membership
            # test, resolved at JAX trace time (no runtime cost).
            return lambda p, _n=name, _fb=fallback_fn: (
                p[_n] if _n in p else _fb(p))
        else:
            # No fallback — must be in caller's dict
            return lambda p, _n=name: p[_n]

    if isinstance(expr, dict):
        def _compile(e):
            return compile_expr(e, defs, _compiling)

        if "*" in expr:
            fa, fb = _compile(expr["*"][0]), _compile(expr["*"][1])
            return lambda p, _a=fa, _b=fb: _a(p) * _b(p)

        if "+" in expr:
            fa, fb = _compile(expr["+"][0]), _compile(expr["+"][1])
            return lambda p, _a=fa, _b=fb: _a(p) + _b(p)

        if "-" in expr:
            fa, fb = _compile(expr["-"][0]), _compile(expr["-"][1])
            return lambda p, _a=fa, _b=fb: _a(p) - _b(p)

        if "/" in expr:
            fa, fb = _compile(expr["/"][0]), _compile(expr["/"][1])
            return lambda p, _a=fa, _b=fb: _a(p) / _b(p)

        if "pow" in expr:
            fa, fb = _compile(expr["pow"][0]), _compile(expr["pow"][1])
            return lambda p, _a=fa, _b=fb: _a(p) ** _b(p)

        if "log" in expr:
            fa = _compile(expr["log"])
            return lambda p, _a=fa: jnp.log(_a(p))

        if "exp" in expr:
            fa = _compile(expr["exp"])
            return lambda p, _a=fa: jnp.exp(_a(p))

        if "not" in expr:
            fa = _compile(expr["not"])
            return lambda p, _a=fa: 1.0 - _a(p)

        raise ValueError(f"Unknown operator: {list(expr.keys())}")

    raise TypeError(f"Unsupported weight expression: {type(expr)}")


def _collect_free_params(expr: WeightExpr, defs: dict,
                         _visiting: frozenset[str] | None = None) -> set[str]:
    """Collect truly free parameters (not defined by the machine).

    A parameter is "free" if it appears in the expression tree and is
    not defined in ``defs``.  Parameters that are defined in ``defs``
    but whose definitions reference other free parameters propagate
    those free parameters upward.
    """
    if _visiting is None:
        _visiting = frozenset()

    if isinstance(expr, (int, float, bool)):
        return set()

    if isinstance(expr, str):
        name = expr
        if name in _visiting:
            return set()  # cycle — already being resolved
        if name in defs:
            return _collect_free_params(
                defs[name], defs, _visiting | {name})
        return {name}

    if isinstance(expr, dict):
        result: set[str] = set()
        for v in expr.values():
            if isinstance(v, list):
                for item in v:
                    result |= _collect_free_params(item, defs, _visiting)
            else:
                result |= _collect_free_params(v, defs, _visiting)
        return result

    return set()


@dataclass
class ParameterizedMachine:
    """A machine compiled for position-dependent parameter evaluation.

    Weight expressions are pre-compiled into JAX operations.  At each
    ``(i, j)`` position in the 2D DP, transition weights are computed
    from position-specific parameter values.

    Parameters referenced by transition weights are resolved in order:

    1. **Caller's param dict** — position-dependent ``(Li+1, Lo+1)``
       tensors supplied at runtime.
    2. **Machine definitions** (``defs``) — numeric assignments or
       weight expressions from the machine JSON.
    3. **Error** — if a parameter is defined in neither place.
    """
    n_states: int
    n_input_tokens: int   # including empty token at index 0
    n_output_tokens: int  # including empty token at index 0
    input_tokens: list[str]   # ['', 'a', 'b', ...]
    output_tokens: list[str]  # ['', '0', '1', ...]
    param_names: set[str]      # all params referenced (including defined ones)
    free_params: set[str]      # params the caller must supply (not in defs)

    # Compiled transition structure
    _grouped_transitions: list

    # Precomputed flat indices for vectorized scatter
    _flat_indices: jnp.ndarray = None
    _tensor_size: int = 0

    @classmethod
    def from_machine(cls, machine: Machine) -> ParameterizedMachine:
        """Compile a Machine's weight expressions for JAX evaluation.

        Reads the machine's ``defs`` to provide fallback values for
        parameters not supplied by the caller.  Only parameters that
        are truly free (not defined in ``defs`` or reachable through
        ``defs``) must be provided at runtime.

        Args:
            machine: Machine with weight expressions (unevaluated).

        Returns:
            ParameterizedMachine ready for use with ``neural_log_forward``
            etc.
        """
        in_alpha = machine.input_alphabet()
        out_alpha = machine.output_alphabet()
        in_tok_map = {sym: i + 1 for i, sym in enumerate(in_alpha)}
        out_tok_map = {sym: i + 1 for i, sym in enumerate(out_alpha)}
        n_in = len(in_alpha) + 1
        n_out = len(out_alpha) + 1
        S = machine.n_states
        defs = machine.defs

        # Collect and compile all transitions
        all_params: set[str] = set()
        free: set[str] = set()
        grouped: dict[tuple, list] = defaultdict(list)

        for src, state in enumerate(machine.state):
            for t in state.trans:
                it = in_tok_map.get(t.input, 0) if t.input else 0
                ot = out_tok_map.get(t.output, 0) if t.output else 0
                fn = compile_expr(t.weight, defs)
                all_params |= expr_params(t.weight)
                free |= _collect_free_params(t.weight, defs)
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
            free_params=free,
            _grouped_transitions=transitions,
            _flat_indices=jnp.array(flat_indices),
            _tensor_size=tensor_size,
        )

    def build_log_trans(self, param_dict: dict) -> jnp.ndarray:
        """Build log_trans[in_tok, out_tok, src, dst] from parameter values.

        Uses a single vectorized scatter for JAX grad compatibility.
        Parameters not in ``param_dict`` fall back to machine ``defs``.

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
