"""JAX-accelerated DP algorithms for Machine Boss WFSTs.

Provides 36 algorithm variants across 4 axes:
- Sequence type: TOK (token) or PSWM (position-specific weight matrix)
- Dimensionality: 1D (generator/recognizer) or 2D (transducer)
- Kernel: DENSE (tensor indexing) or SPARSE (COO gather/scatter)
- Strategy: SIMPLE (sequential scan) or OPTIMAL (parallel prefix / wavefront)
  (SPARSE + OPTIMAL excluded = 12 combos; 36 total implemented)

Main API:
    log_forward(machine, input_seq, output_seq, strategy='auto', kernel='auto')
    log_viterbi(machine, input_seq, output_seq, strategy='auto', kernel='auto')
    log_backward_matrix(machine, input_seq, output_seq, strategy='auto', kernel='auto')

Sequence types:
    TokenSeq(tokens)          - observed tokens (one-hot emission)
    PSWMSeq(log_probs)        - position-specific weight matrix

Padding utilities (avoid JAX recompilation for varying lengths):
    pad_length(length)        - round up to geometric series bucket
    pad_token_seq(seq, L)     - pad TokenSeq to length L
    pad_pswm_seq(seq, L)      - pad PSWMSeq to length L
"""

from .types import JAXMachine, NEG_INF
from .semiring import LogSemiring, LOGSUMEXP, MAXPLUS
from .seq import TokenSeq, PSWMSeq, wrap_seq, pad_length, pad_token_seq, pad_pswm_seq
from .forward import log_forward, log_forward_dense
from .backward import log_backward_matrix, log_backward_dense
from .viterbi import log_viterbi, log_viterbi_dense
from .fused import FusedMachine, fused_log_forward, fused_log_viterbi
from .jax_weight import ParameterizedMachine
from .dp_neural import neural_log_forward, neural_log_viterbi, neural_log_backward_matrix
