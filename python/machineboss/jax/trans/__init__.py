"""Transition-centric JAX DP algorithms for Machine Boss WFSTs.

The guiding principle: the Machine IS the list of Transitions.

TransMachine stores transitions as parallel COO arrays with pre-built
boolean masks, registered as a JAX pytree for jit/grad/vmap compatibility.

Public API:
    TransMachine           - Core data structure
    propagate_silent       - Silent closure (forward)
    propagate_silent_backward - Silent closure (backward)
    emit_step_forward      - Emitting step (forward)
    emit_step_backward     - Emitting step (backward)
    to_matrix              - Build (S,S) transition matrix from masked transitions
    forward_1d             - 1D forward (simple + optimal)
    backward_1d            - 1D backward (simple + optimal)
    viterbi_1d             - 1D Viterbi (simple + optimal)
    forward_2d             - 2D forward (simple + optimal)
    backward_2d            - 2D backward (simple + optimal)
    viterbi_2d             - 2D Viterbi (simple + optimal)
    forward_backward       - Forward-backward with expected counts
    aligned_forward        - Alignment-constrained forward
    aligned_viterbi        - Alignment-constrained Viterbi
    ParameterizedTransMachine - Parameterized weights
    neural_forward_2d      - Position-dependent 2D forward
    neural_viterbi_2d      - Position-dependent 2D Viterbi
    neural_aligned_forward - Position-dependent aligned forward
    fused_forward          - Fused Plan7+transducer forward
    fused_viterbi          - Fused Plan7+transducer Viterbi
    beam_align             - Wavefront beam Viterbi
"""

from .machine import TransMachine
from .kernel import (
    propagate_silent, propagate_silent_backward,
    emit_step_forward, emit_step_backward,
    to_matrix,
)
from .dp_1d import (
    forward_1d, backward_1d, viterbi_1d,
)
from .dp_2d import (
    forward_2d, forward_2d_matrix, backward_2d, viterbi_2d,
)
from .fwdback import forward_backward, forward_backward_1d
from .dp_aligned import (
    aligned_forward, aligned_viterbi,
    neural_aligned_forward, neural_aligned_viterbi,
    validate_alignment, MAT, INS, DEL,
)
from .parameterized import ParameterizedTransMachine
from .dp_neural import (
    neural_forward_2d, neural_viterbi_2d,
    neural_backward_2d,
    neural_forward_2d_tok, neural_viterbi_2d_tok,
)
from .dp_fused import fused_forward, fused_viterbi
from .dp_beam import beam_align
