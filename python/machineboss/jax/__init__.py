"""JAX-accelerated DP algorithms for Machine Boss WFSTs.

Primary interface: TransMachine (transition-centric JAX pytree)

    from machineboss.jax import TransMachine, log_forward
    tm = TransMachine.from_machine(m, params={"t": 0.5})
    ll = log_forward(tm, input_seq, output_seq)

The dispatchers (log_forward, log_viterbi, log_backward_matrix) accept
both TransMachine and legacy JAXMachine.

Transition-centric API (recommended):
    TransMachine              - JAX pytree WFST representation
    ParameterizedTransMachine - Parameterized weights for neural DP
    forward_1d, backward_1d, viterbi_1d
    forward_2d, backward_2d, viterbi_2d
    forward_backward          - Vectorized expected transition counts

Unified dispatchers (accept both TransMachine and JAXMachine):
    log_forward, log_viterbi, log_backward_matrix

Sequence types:
    TokenSeq(tokens)          - observed tokens (one-hot emission)
    PSWMSeq(log_probs)        - position-specific weight matrix

Padding utilities (avoid JAX recompilation for varying lengths):
    pad_length(length)        - round up to geometric series bucket
    pad_token_seq(seq, L)     - pad TokenSeq to length L
    pad_pswm_seq(seq, L)      - pad PSWMSeq to length L
"""

# Core types
from .types import JAXMachine, NEG_INF
from .trans.machine import TransMachine
from .semiring import LogSemiring, LOGSUMEXP, MAXPLUS
from .seq import TokenSeq, PSWMSeq, wrap_seq, pad_length, pad_token_seq, pad_pswm_seq

# Unified dispatchers (accept TransMachine or JAXMachine)
from .forward import log_forward, log_forward_dense
from .backward import log_backward_matrix, log_backward_dense
from .viterbi import log_viterbi, log_viterbi_dense

# TransMachine-native DP (recommended)
from .trans.dp_1d import forward_1d, backward_1d, viterbi_1d
from .trans.dp_2d import forward_2d, forward_2d_matrix, backward_2d, viterbi_2d
from .trans.fwdback import forward_backward, forward_backward_1d
from .trans.parameterized import ParameterizedTransMachine
from .trans.dp_neural import (
    neural_forward_2d as trans_neural_forward_2d,
    neural_viterbi_2d as trans_neural_viterbi_2d,
    neural_backward_2d as trans_neural_backward_2d,
    neural_forward_2d_tok as trans_neural_forward_2d_tok,
    neural_viterbi_2d_tok as trans_neural_viterbi_2d_tok,
)
from .trans.dp_aligned import (
    aligned_forward as trans_aligned_forward,
    aligned_viterbi as trans_aligned_viterbi,
    validate_alignment, MAT, INS, DEL,
)

# Legacy JAXMachine-based APIs (still functional)
from .fused import FusedMachine, fused_log_forward, fused_log_viterbi
from .jax_weight import ParameterizedMachine
from .dp_neural import (
    neural_log_forward, neural_log_viterbi, neural_log_backward_matrix,
    neural_log_forward_tok, neural_log_viterbi_tok, neural_log_backward_matrix_tok,
)
from .dp_aligned import (
    aligned_log_forward, aligned_log_viterbi,
    neural_aligned_log_forward, neural_aligned_log_viterbi,
)
