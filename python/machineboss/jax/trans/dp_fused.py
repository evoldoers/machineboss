"""Fused Plan7+transducer DP for TransMachine.

Delegates to the existing fused_plan7 implementation, converting
TransMachine to JAXMachine as needed for the transducer component.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..semiring import LOGSUMEXP, MAXPLUS
from .machine import TransMachine


def fused_forward(plan7_model, transducer_tm: TransMachine,
                  output_seq: jnp.ndarray) -> float:
    """Fused Plan7+transducer Forward algorithm.

    Args:
        plan7_model: HmmerModel or FusedPlan7Machine
        transducer_tm: TransMachine for the transducer
        output_seq: output token sequence
    Returns:
        Log-likelihood (scalar).
    """
    from ..fused_plan7 import FusedPlan7Machine, _fused_plan7_dp

    if isinstance(plan7_model, FusedPlan7Machine):
        fm = plan7_model
    else:
        jm = transducer_tm.to_jax_machine()
        from ...eval import EvaluatedMachine
        td_em = EvaluatedMachine(
            n_states=transducer_tm.n_states,
            input_tokens=list(transducer_tm.input_tokens),
            output_tokens=list(transducer_tm.output_tokens),
            transitions=[],
        )
        fm = FusedPlan7Machine.build(plan7_model, td_em)

    return _fused_plan7_dp(fm, output_seq, LOGSUMEXP)


def fused_viterbi(plan7_model, transducer_tm: TransMachine,
                  output_seq: jnp.ndarray) -> float:
    """Fused Plan7+transducer Viterbi algorithm."""
    from ..fused_plan7 import FusedPlan7Machine, _fused_plan7_dp

    if isinstance(plan7_model, FusedPlan7Machine):
        fm = plan7_model
    else:
        jm = transducer_tm.to_jax_machine()
        from ...eval import EvaluatedMachine
        td_em = EvaluatedMachine(
            n_states=transducer_tm.n_states,
            input_tokens=list(transducer_tm.input_tokens),
            output_tokens=list(transducer_tm.output_tokens),
            transitions=[],
        )
        fm = FusedPlan7Machine.build(plan7_model, td_em)

    return _fused_plan7_dp(fm, output_seq, MAXPLUS)
