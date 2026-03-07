"""Compile weight expressions into JAX-traceable functions.

Converts the JSON weight expression mini-language (defined in weight.py)
into JAX operations that are JIT-compilable and differentiable.

Supports machine-level parameter definitions (``defs``): if a parameter
referenced by a transition weight expression is not supplied by the caller
at runtime, the compiler falls back to the machine's own definition
(numeric value or weight expression). Only truly free parameters — those
with no definition anywhere — cause an error.

When parameter families are detected (e.g. ``pi_0``, ``pi_1``, …, ``pi_19``),
transitions sharing the same weight template are compiled into a single
vectorized operation, dramatically reducing JIT trace time for large
alphabets.
"""

from __future__ import annotations

import json as _json
import re as _re
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


# ---------------------------------------------------------------------------
# Vectorized compilation: parameter family detection and template grouping
# ---------------------------------------------------------------------------

def _detect_families(param_names: set[str]) -> dict[str, dict[str, int]]:
    """Detect families of parameters with pattern prefix_N.

    Returns dict mapping family prefix to {full_name: index} pairs.
    Only returns families with >= 2 members.
    """
    candidates: dict[str, dict[str, int]] = defaultdict(dict)
    for name in param_names:
        m = _re.match(r'^(.+)_(\d+)$', name)
        if m:
            prefix, idx = m.group(1), int(m.group(2))
            candidates[prefix][name] = idx
    return {k: v for k, v in candidates.items() if len(v) >= 2}


def _normalize_expr(expr: WeightExpr, families: dict[str, dict[str, int]]):
    """Replace family parameter names with placeholders.

    Returns:
        (normalized_expr, family_uses) where family_uses maps
        family_name -> set of indices used in this expression.
    """
    if isinstance(expr, (int, float, bool)):
        return expr, {}

    if isinstance(expr, str):
        for fam_name, members in families.items():
            if expr in members:
                return f"__{fam_name}__", {fam_name: {members[expr]}}
        return expr, {}

    if isinstance(expr, dict):
        result = {}
        all_uses: dict[str, set[int]] = {}
        for op, operands in expr.items():
            if isinstance(operands, list):
                new_operands = []
                for operand in operands:
                    n, u = _normalize_expr(operand, families)
                    new_operands.append(n)
                    for k, v in u.items():
                        all_uses.setdefault(k, set()).update(v)
                result[op] = new_operands
            else:
                n, u = _normalize_expr(operands, families)
                result[op] = n
                for k, v in u.items():
                    all_uses.setdefault(k, set()).update(v)
        return result, all_uses

    return expr, {}


@dataclass
class _VecGroup:
    """A batch of transitions evaluated together via vectorized template."""
    template_fn: object       # callable: augmented_params -> (family_size,) or scalar
    family_name: str | None   # which family this group varies over (or None)
    family_size: int          # number of family members (0 if no family)
    family_indices: jnp.ndarray  # (n_members,) int: index into template output
    flat_indices: jnp.ndarray    # (n_members,) int: position in flat tensor


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

    When parameter families are detected (e.g. ``pi_0`` … ``pi_19``),
    transitions sharing the same weight template are compiled into
    vectorized groups for faster JIT compilation.
    """
    n_states: int
    n_input_tokens: int   # including empty token at index 0
    n_output_tokens: int  # including empty token at index 0
    input_tokens: list[str]   # ['', 'a', 'b', ...]
    output_tokens: list[str]  # ['', '0', '1', ...]
    param_names: set[str]      # all params referenced (including defined ones)
    free_params: set[str]      # params the caller must supply (not in defs)

    # Compiled transition structure (scalar path)
    _grouped_transitions: list

    # Precomputed flat indices for vectorized scatter (scalar path)
    _flat_indices: jnp.ndarray = None
    _tensor_size: int = 0

    # Vectorized compilation (set when families detected)
    _vec_groups: list | None = None      # list of _VecGroup
    _scalar_entries: list | None = None  # list of (flat_index, [fns])
    _families: dict | None = None        # family_name -> {param_name: index}

    @classmethod
    def from_machine(cls, machine: Machine) -> ParameterizedMachine:
        """Compile a Machine's weight expressions for JAX evaluation.

        Reads the machine's ``defs`` to provide fallback values for
        parameters not supplied by the caller.  Only parameters that
        are truly free (not defined in ``defs`` or reachable through
        ``defs``) must be provided at runtime.

        When parameter families are detected (names matching
        ``prefix_0``, ``prefix_1``, …), transitions sharing the same
        weight template are compiled into vectorized groups.  This
        reduces JIT trace time from O(N_transitions) to O(N_templates).

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

        # Collect all transitions: params, free params, and raw expressions
        all_params: set[str] = set()
        free: set[str] = set()
        # (in_tok, out_tok, src, dst) -> [(weight_expr, compiled_fn)]
        position_data: dict[tuple, list] = defaultdict(list)

        for src, state in enumerate(machine.state):
            for t in state.trans:
                it = in_tok_map.get(t.input, 0) if t.input else 0
                ot = out_tok_map.get(t.output, 0) if t.output else 0
                fn = compile_expr(t.weight, defs)
                all_params |= expr_params(t.weight)
                free |= _collect_free_params(t.weight, defs)
                position_data[(it, ot, src, t.dest)].append((t.weight, fn))

        # Build scalar grouped transitions (always, for fallback)
        grouped_transitions = []
        tensor_size = n_in * n_out * S * S
        flat_indices_list = []
        for (it, ot, src, dst), entries in position_data.items():
            fns = [fn for _, fn in entries]
            grouped_transitions.append((it, ot, src, dst, fns))
            flat_indices_list.append(
                it * (n_out * S * S) + ot * (S * S) + src * S + dst)

        flat_indices = np.array(flat_indices_list, dtype=np.int32)

        pm = cls(
            n_states=S,
            n_input_tokens=n_in,
            n_output_tokens=n_out,
            input_tokens=[''] + in_alpha,
            output_tokens=[''] + out_alpha,
            param_names=all_params,
            free_params=free,
            _grouped_transitions=grouped_transitions,
            _flat_indices=jnp.array(flat_indices),
            _tensor_size=tensor_size,
        )

        # Try vectorized compilation
        pm._try_vectorize(position_data, defs)

        return pm

    def _try_vectorize(self, position_data, defs):
        """Detect parameter families and build vectorized groups."""
        families = _detect_families(self.free_params)
        if not families:
            return

        template_groups: dict[tuple, dict] = defaultdict(
            lambda: {"expr": None, "members": []})
        scalar_entries = []

        n_out = self.n_output_tokens
        S = self.n_states

        for (it, ot, src, dst), entries in position_data.items():
            flat_idx = it * (n_out * S * S) + ot * (S * S) + src * S + dst

            if len(entries) > 1:
                # Multiple transitions to same position — keep scalar
                fns = [fn for _, fn in entries]
                scalar_entries.append((flat_idx, fns))
                continue

            weight_expr, compiled_fn = entries[0]
            norm_expr, family_uses = _normalize_expr(weight_expr, families)

            # Only vectorize if 0 or 1 families used, and each family
            # contributes at most 1 distinct index per expression
            if len(family_uses) > 1:
                scalar_entries.append((flat_idx, [compiled_fn]))
                continue
            if any(len(idxs) > 1 for idxs in family_uses.values()):
                scalar_entries.append((flat_idx, [compiled_fn]))
                continue

            template_key = _json.dumps(norm_expr, sort_keys=True)
            family_name = None
            family_index = 0
            if family_uses:
                family_name = next(iter(family_uses))
                family_index = next(iter(family_uses[family_name]))

            group = template_groups[(template_key, family_name)]
            if group["expr"] is None:
                group["expr"] = norm_expr
            group["members"].append((flat_idx, family_index))

        # Compile template functions and build vectorized groups
        vec_groups = []
        for (template_key, family_name), group_info in template_groups.items():
            norm_expr = group_info["expr"]
            members = group_info["members"]

            template_fn = compile_expr(norm_expr, defs)

            flat_idxs = np.array([m[0] for m in members], dtype=np.int32)
            fam_idxs = np.array([m[1] for m in members], dtype=np.int32)

            family_size = 0
            if family_name:
                family_size = max(families[family_name].values()) + 1

            vec_groups.append(_VecGroup(
                template_fn=template_fn,
                family_name=family_name,
                family_size=family_size,
                family_indices=jnp.array(fam_idxs),
                flat_indices=jnp.array(flat_idxs),
            ))

        self._vec_groups = vec_groups
        self._scalar_entries = scalar_entries
        self._families = families

    def build_log_trans(self, param_dict: dict) -> jnp.ndarray:
        """Build log_trans[in_tok, out_tok, src, dst] from parameter values.

        Uses a single vectorized scatter for JAX grad compatibility.
        Parameters not in ``param_dict`` fall back to machine ``defs``.

        When vectorized groups are available (parameter families detected),
        uses batch evaluation for ~60x fewer traced operations.

        Args:
            param_dict: maps parameter names to JAX scalars.

        Returns:
            (n_in, n_out, S, S) log-weight tensor.
        """
        if self._vec_groups is not None:
            return self._build_log_trans_vectorized(param_dict)
        return self._build_log_trans_scalar(param_dict)

    def _build_log_trans_scalar(self, param_dict: dict) -> jnp.ndarray:
        """Scalar path: evaluate each transition individually."""
        from .types import NEG_INF

        log_weights = []
        for _it, _ot, _src, _dst, fns in self._grouped_transitions:
            lw = jnp.log(jnp.maximum(fns[0](param_dict), 1e-45))
            for fn in fns[1:]:
                lw = jnp.logaddexp(
                    lw, jnp.log(jnp.maximum(fn(param_dict), 1e-45)))
            log_weights.append(lw)

        all_lw = jnp.stack(log_weights)
        flat = jnp.full(self._tensor_size, NEG_INF)
        flat = flat.at[self._flat_indices].set(all_lw)
        shape = (self.n_input_tokens, self.n_output_tokens,
                 self.n_states, self.n_states)
        return flat.reshape(shape)

    def _build_log_trans_vectorized(self, param_dict: dict) -> jnp.ndarray:
        """Vectorized path: batch-evaluate transitions by template group."""
        from .types import NEG_INF

        # Build family arrays: e.g. __pi__ = stack([pi_0, pi_1, ..., pi_19])
        augmented = dict(param_dict)
        for family_name, members in self._families.items():
            sorted_by_idx = sorted(members.items(), key=lambda x: x[1])
            max_idx = sorted_by_idx[-1][1]
            if (sorted_by_idx[0][1] == 0
                    and max_idx == len(sorted_by_idx) - 1):
                # Contiguous 0-based — fast path
                family_array = jnp.stack(
                    [param_dict[name] for name, _ in sorted_by_idx])
            else:
                # Non-contiguous — use scatter
                family_array = jnp.zeros(max_idx + 1)
                for name, idx in sorted_by_idx:
                    family_array = family_array.at[idx].set(param_dict[name])
            augmented[f"__{family_name}__"] = family_array

        flat = jnp.full(self._tensor_size, NEG_INF)

        # Vectorized groups
        for vg in self._vec_groups:
            if vg.family_name:
                # Template fn returns (family_size,) array via broadcasting
                weights = vg.template_fn(augmented)
                batch_weights = weights[vg.family_indices]
            else:
                # Template fn returns scalar; broadcast to all members
                weight = vg.template_fn(augmented)
                batch_weights = jnp.broadcast_to(
                    weight, vg.flat_indices.shape)
            log_w = jnp.log(jnp.maximum(batch_weights, 1e-45))
            flat = flat.at[vg.flat_indices].set(log_w)

        # Scalar entries (fallback for multi-transition positions etc.)
        for flat_idx, fns in self._scalar_entries:
            lw = jnp.log(jnp.maximum(fns[0](param_dict), 1e-45))
            for fn in fns[1:]:
                lw = jnp.logaddexp(
                    lw, jnp.log(jnp.maximum(fn(param_dict), 1e-45)))
            flat = flat.at[flat_idx].set(lw)

        shape = (self.n_input_tokens, self.n_output_tokens,
                 self.n_states, self.n_states)
        return flat.reshape(shape)

    def tokenize_input(self, seq: list[str]) -> list[int]:
        tok_map = {sym: i for i, sym in enumerate(self.input_tokens)}
        return [tok_map[s] for s in seq]

    def tokenize_output(self, seq: list[str]) -> list[int]:
        tok_map = {sym: i for i, sym in enumerate(self.output_tokens)}
        return [tok_map[s] for s in seq]
