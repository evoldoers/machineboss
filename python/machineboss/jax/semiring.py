"""Log-space semiring abstractions for DP algorithms.

Two semirings:
- LOGSUMEXP: log-sum-exp (Forward/Backward)
- MAXPLUS: max-plus (Viterbi)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from .types import NEG_INF


@dataclass(frozen=True)
class LogSemiring:
    """A semiring operating in log-space.

    plus: binary operation (logaddexp for sum, maximum for max)
    zero: identity element for plus (-inf in log-space)
    """
    plus: Callable
    zero: float = NEG_INF

    def reduce(self, x: jnp.ndarray, axis: int | tuple[int, ...]) -> jnp.ndarray:
        """Reduce along axis using the semiring plus operation."""
        if self.plus is jnp.logaddexp:
            return jax.nn.logsumexp(x, axis=axis)
        else:
            return jnp.max(x, axis=axis)

    def mat_mul(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Semiring matrix multiply: C[...,i,k] = plus_j(a[...,i,j] + b[...,j,k]).

        Supports batched inputs with arbitrary leading dimensions
        (required by jax.lax.associative_scan).
        """
        # a: (..., M, N), b: (..., N, K)
        # a[..., :, :, None] shape (..., M, N, 1)
        # b[..., None, :, :] shape (..., 1, N, K)
        # sum: (..., M, N, K), reduce over axis=-2 (N) -> (..., M, K)
        return self.reduce(a[..., :, :, None] + b[..., None, :, :], axis=-2)


LOGSUMEXP = LogSemiring(plus=jnp.logaddexp)
MAXPLUS = LogSemiring(plus=jnp.maximum)
