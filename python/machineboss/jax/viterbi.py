"""Log-space Viterbi algorithm using JAX.

Same structure as Forward but uses max instead of logsumexp.
Dispatches to the appropriate engine based on strategy and kernel params.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import JAXMachine, NEG_INF
from .semiring import MAXPLUS
from .kernel_dense import propagate_silent


def _propagate_silent_max(cell: jnp.ndarray, silent_trans: jnp.ndarray,
                          max_iter: int = 100) -> jnp.ndarray:
    """Legacy interface — delegates to kernel_dense.propagate_silent with MAXPLUS."""
    return propagate_silent(cell, silent_trans, MAXPLUS)


def log_viterbi_dense(machine: JAXMachine,
                      input_seq: jnp.ndarray,
                      output_seq: jnp.ndarray) -> float:
    """Viterbi algorithm using dense transition tensor.

    Legacy interface — delegates to dp_2d_simple.forward_2d_dense with MAXPLUS.
    """
    from .dp_2d_simple import forward_2d_dense
    return forward_2d_dense(machine, input_seq, output_seq, MAXPLUS)


def log_viterbi(machine: JAXMachine,
                input_seq: jnp.ndarray | None = None,
                output_seq: jnp.ndarray | None = None,
                *,
                strategy: str = 'auto',
                kernel: str = 'auto',
                length: int | None = None) -> float:
    """Viterbi algorithm — dispatches to appropriate engine.

    Args:
        machine: JAXMachine
        input_seq: (Li,) input token indices, TokenSeq, PSWMSeq, or None
        output_seq: (Lo,) output token indices, TokenSeq, PSWMSeq, or None
        strategy: 'simple', 'optimal', or 'auto'
        kernel: 'dense', 'sparse', or 'auto'
        length: real sequence length for padded 1D sequences. If None, uses len(seq).

    Returns:
        Log-probability of most likely path (scalar).

    Raises:
        ValueError: if sequences don't match the machine type
    """
    from .forward import _validate_seqs, _auto_pad_1d

    _validate_seqs(machine, input_seq, output_seq)

    is_1d = (input_seq is None) or (output_seq is None)

    if kernel == 'auto':
        kernel = 'dense' if machine.log_trans is not None else 'sparse'

    if strategy == 'auto':
        if is_1d and kernel == 'dense':
            strategy = 'optimal'
        else:
            strategy = 'simple'

    if is_1d:
        # Auto-pad for JIT compilation cache efficiency
        input_seq, output_seq, length = _auto_pad_1d(
            input_seq, output_seq, machine, length)

        if strategy == 'optimal' and kernel == 'dense':
            from .dp_1d_optimal import forward_1d_optimal
            return forward_1d_optimal(machine, input_seq, output_seq, MAXPLUS,
                                      length=length)
        else:
            from .dp_1d_simple import forward_1d_simple
            return forward_1d_simple(machine, input_seq, output_seq, MAXPLUS,
                                     kernel=kernel, length=length)
    else:
        if strategy == 'optimal' and kernel == 'dense':
            from .dp_2d_optimal import forward_2d_optimal
            return forward_2d_optimal(machine, input_seq, output_seq, MAXPLUS)
        elif kernel == 'sparse':
            from .dp_2d_simple import forward_2d_sparse
            return forward_2d_sparse(machine, input_seq, output_seq, MAXPLUS)
        else:
            from .dp_2d_simple import forward_2d_dense
            return forward_2d_dense(machine, input_seq, output_seq, MAXPLUS)
