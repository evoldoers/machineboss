"""Log-space Forward algorithm using JAX.

Computes log P(input, output | machine) via the Forward algorithm.
Dispatches to the appropriate engine based on strategy and kernel params.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import JAXMachine, NEG_INF
from .semiring import LOGSUMEXP
from .kernel_dense import propagate_silent, emit_step_forward
from .seq import wrap_seq, TokenSeq, PSWMSeq, pad_length, pad_token_seq, pad_pswm_seq


def _propagate_silent(cell: jnp.ndarray, silent_trans: jnp.ndarray,
                      max_iter: int = 100) -> jnp.ndarray:
    """Legacy interface — delegates to kernel_dense.propagate_silent with LOGSUMEXP."""
    return propagate_silent(cell, silent_trans, LOGSUMEXP)


def log_forward_dense(machine: JAXMachine,
                      input_seq: jnp.ndarray,
                      output_seq: jnp.ndarray) -> float:
    """Forward algorithm using dense transition tensor.

    Legacy interface — delegates to dp_2d_simple.forward_2d_dense.
    """
    from .dp_2d_simple import forward_2d_dense
    return forward_2d_dense(machine, input_seq, output_seq, LOGSUMEXP)


def _seq_is_nonempty(seq) -> bool:
    """Check if a sequence is provided and non-empty."""
    if seq is None:
        return False
    if hasattr(seq, '__len__') and len(seq) == 0:
        return False
    return True


def _auto_pad_1d(input_seq, output_seq, machine, length):
    """Auto-pad the active 1D sequence for JIT compilation cache efficiency.

    When length is not explicitly provided, pads the active sequence to a
    geometric-series length so JAX reuses compiled kernels.

    Returns (input_seq, output_seq, length).
    """
    if length is not None:
        return input_seq, output_seq, length

    if input_seq is None:
        # Generator or transducer without input: pad output_seq
        seq = wrap_seq(output_seq, machine.n_output_tokens)
        L = len(seq)
        padded_L = pad_length(L)
        if padded_L > L:
            if isinstance(seq, PSWMSeq):
                seq, orig_L = pad_pswm_seq(seq, padded_L)
            else:
                seq, orig_L = pad_token_seq(seq, padded_L)
            return None, seq, orig_L
        return None, seq, None
    else:
        # Recognizer or transducer without output: pad input_seq
        seq = wrap_seq(input_seq, machine.n_input_tokens)
        L = len(seq)
        padded_L = pad_length(L)
        if padded_L > L:
            if isinstance(seq, PSWMSeq):
                seq, orig_L = pad_pswm_seq(seq, padded_L)
            else:
                seq, orig_L = pad_token_seq(seq, padded_L)
            return seq, None, orig_L
        return seq, None, None


def _validate_seqs(machine: JAXMachine, input_seq, output_seq):
    """Validate that supplied sequences match the machine type.

    Empty arrays (length 0) are treated as equivalent to None.
    """
    has_in = _seq_is_nonempty(input_seq)
    has_out = _seq_is_nonempty(output_seq)

    if machine.is_generator():
        if has_in:
            raise ValueError(
                f"Machine is a generator (no input alphabet) but a non-empty input_seq was provided")
        if output_seq is None:
            raise ValueError(
                f"Machine is a generator (output-only) but output_seq is None")
    elif machine.is_recognizer():
        if has_out:
            raise ValueError(
                f"Machine is a recognizer (no output alphabet) but a non-empty output_seq was provided")
        if input_seq is None:
            raise ValueError(
                f"Machine is a recognizer (input-only) but input_seq is None")
    elif machine.is_transducer():
        # Transducers accept both 1D and 2D usage: either seq can be None
        pass
    else:
        # Null machine (no input or output)
        if has_in:
            raise ValueError("Machine has no input alphabet but input_seq was provided")
        if has_out:
            raise ValueError("Machine has no output alphabet but output_seq was provided")


def log_forward(machine: JAXMachine,
                input_seq: jnp.ndarray | None = None,
                output_seq: jnp.ndarray | None = None,
                *,
                strategy: str = 'auto',
                kernel: str = 'auto',
                length: int | None = None) -> float:
    """Forward algorithm — dispatches to appropriate engine.

    Args:
        machine: JAXMachine
        input_seq: (Li,) input token indices, TokenSeq, PSWMSeq, or None
        output_seq: (Lo,) output token indices, TokenSeq, PSWMSeq, or None
        strategy: 'simple', 'optimal', or 'auto'
        kernel: 'dense', 'sparse', or 'auto'
        length: real sequence length for padded 1D sequences. If None, uses len(seq).

    Returns:
        Log-likelihood (scalar).

    Raises:
        ValueError: if sequences don't match the machine type
    """
    _validate_seqs(machine, input_seq, output_seq)

    # Determine dimensionality: 1D only if a sequence is literally None
    is_1d = (input_seq is None) or (output_seq is None)

    # Resolve kernel
    if kernel == 'auto':
        kernel = 'dense' if machine.log_trans is not None else 'sparse'

    # Resolve strategy
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
            return forward_1d_optimal(machine, input_seq, output_seq, LOGSUMEXP,
                                      length=length)
        else:
            from .dp_1d_simple import forward_1d_simple
            return forward_1d_simple(machine, input_seq, output_seq, LOGSUMEXP,
                                     kernel=kernel, length=length)
    else:
        if strategy == 'optimal' and kernel == 'dense':
            from .dp_2d_optimal import forward_2d_optimal
            return forward_2d_optimal(machine, input_seq, output_seq, LOGSUMEXP)
        elif kernel == 'sparse':
            from .dp_2d_simple import forward_2d_sparse
            return forward_2d_sparse(machine, input_seq, output_seq, LOGSUMEXP)
        else:
            from .dp_2d_simple import forward_2d_dense
            return forward_2d_dense(machine, input_seq, output_seq, LOGSUMEXP)
