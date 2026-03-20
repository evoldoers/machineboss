"""ParameterizedTransMachine: template TransMachine with compiled weight functions.

Produces log_w vectors (shape (T,)) instead of dense tensors,
reusing compile_expr and vectorized family detection from jax_weight.py.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from ...machine import Machine
from ...weight import params as expr_params
from ..jax_weight import compile_expr, _collect_free_params, _detect_families, _normalize_expr

import json as _json


@dataclass
class ParameterizedTransMachine:
    """Template TransMachine with compiled weight functions.

    At each (i, j) position, build_log_w(params, i, j) produces a
    (T,) log-weight vector from position-dependent parameters.
    """
    n_states: int
    n_in: int
    n_out: int
    input_tokens: list[str]
    output_tokens: list[str]
    param_names: set[str]
    free_params: set[str]

    # Compiled weight functions: list of (indices, fn) pairs
    # Each fn: param_dict -> scalar log-weight (for all transitions at indices)
    _weight_fns: list  # [(trans_index, compiled_fn), ...]
    _n_transitions: int

    # Template arrays for building TransMachine
    _src: jnp.ndarray
    _dst: jnp.ndarray
    _in_tok: jnp.ndarray
    _out_tok: jnp.ndarray
    _silent_mask: jnp.ndarray
    _emit_in_mask: jnp.ndarray
    _emit_out_mask: jnp.ndarray
    _emit_both_mask: jnp.ndarray

    @classmethod
    def from_machine(cls, machine: Machine) -> ParameterizedTransMachine:
        """Compile a Machine's weight expressions for log_w vector evaluation."""
        in_alpha = machine.input_alphabet()
        out_alpha = machine.output_alphabet()
        in_tok_map = {sym: i + 1 for i, sym in enumerate(in_alpha)}
        out_tok_map = {sym: i + 1 for i, sym in enumerate(out_alpha)}
        n_in = len(in_alpha) + 1
        n_out = len(out_alpha) + 1
        S = machine.n_states
        defs = machine.defs

        all_params: set[str] = set()
        free: set[str] = set()
        weight_fns = []

        src_list, dst_list, in_tok_list, out_tok_list = [], [], [], []

        for src, state in enumerate(machine.state):
            for t in state.trans:
                it = in_tok_map.get(t.input, 0) if t.input else 0
                ot = out_tok_map.get(t.output, 0) if t.output else 0
                fn = compile_expr(t.weight, defs)
                all_params |= expr_params(t.weight)
                free |= _collect_free_params(t.weight, defs)

                idx = len(src_list)
                weight_fns.append((idx, fn))
                src_list.append(src)
                dst_list.append(t.dest)
                in_tok_list.append(it)
                out_tok_list.append(ot)

        src_arr = jnp.array(src_list, dtype=jnp.int32)
        dst_arr = jnp.array(dst_list, dtype=jnp.int32)
        in_tok_arr = jnp.array(in_tok_list, dtype=jnp.int32)
        out_tok_arr = jnp.array(out_tok_list, dtype=jnp.int32)

        silent_mask = (in_tok_arr == 0) & (out_tok_arr == 0)
        emit_in_mask = (in_tok_arr > 0) & (out_tok_arr == 0)
        emit_out_mask = (in_tok_arr == 0) & (out_tok_arr > 0)
        emit_both_mask = (in_tok_arr > 0) & (out_tok_arr > 0)

        return cls(
            n_states=S,
            n_in=n_in,
            n_out=n_out,
            input_tokens=[''] + in_alpha,
            output_tokens=[''] + out_alpha,
            param_names=all_params,
            free_params=free,
            _weight_fns=weight_fns,
            _n_transitions=len(src_list),
            _src=src_arr,
            _dst=dst_arr,
            _in_tok=in_tok_arr,
            _out_tok=out_tok_arr,
            _silent_mask=silent_mask,
            _emit_in_mask=emit_in_mask,
            _emit_out_mask=emit_out_mask,
            _emit_both_mask=emit_both_mask,
        )

    def build_log_w(self, params: dict[str, jnp.ndarray],
                    i: int | jnp.ndarray, j: int | jnp.ndarray) -> jnp.ndarray:
        """Build (T,) log-weight vector from position-dependent parameters.

        Args:
            params: dict mapping param names to arrays broadcastable to (Li+1, Lo+1)
            i, j: position indices
        Returns:
            (T,) log-weight vector.
        """
        from ..types import NEG_INF

        pos_params = {
            name: val[jnp.minimum(i, val.shape[0] - 1),
                       jnp.minimum(j, val.shape[1] - 1)]
            for name, val in params.items()
        }

        log_weights = []
        for idx, fn in self._weight_fns:
            w = fn(pos_params)
            log_weights.append(jnp.log(jnp.maximum(w, 1e-45)))

        return jnp.stack(log_weights)

    def build_trans_machine(self, params: dict[str, jnp.ndarray],
                            i: int | jnp.ndarray,
                            j: int | jnp.ndarray) -> 'TransMachine':
        """Build a TransMachine with weights evaluated at position (i, j)."""
        from .machine import TransMachine

        log_w = self.build_log_w(params, i, j)
        return TransMachine(
            self._src, self._dst, self._in_tok, self._out_tok, log_w,
            self._silent_mask, self._emit_in_mask,
            self._emit_out_mask, self._emit_both_mask,
            self.n_states, self.n_in, self.n_out,
            tuple(self.input_tokens), tuple(self.output_tokens),
        )
