"""Wavefront beam Viterbi for TransMachine.

Delegates to the existing beam_align implementation via JAXMachine conversion.
"""

from __future__ import annotations

import numpy as np

from .machine import TransMachine


def beam_align(tm: TransMachine,
               input_seq: np.ndarray | None,
               output_seq: np.ndarray | None,
               beam_width: int = 100):
    """Wavefront beam-Viterbi alignment.

    Works on cyclic machines (e.g. Plan7) where standard Viterbi
    requires topological sort.

    Args:
        tm: TransMachine
        input_seq: Array of input tokens (1-indexed, 0=empty). Can be None.
        output_seq: Array of output tokens (1-indexed, 0=empty). Can be None.
        beam_width: Maximum cells to keep per wavefront.

    Returns:
        BeamAlignResult with score and alignment path.
    """
    from ..beam_align import beam_align as _beam_align

    jm = tm.to_jax_machine()
    return _beam_align(jm, input_seq, output_seq, beam_width=beam_width)
