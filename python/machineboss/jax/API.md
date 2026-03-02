# Machine Boss JAX DP Algorithms

GPU-accelerated Forward, Backward, and Viterbi algorithms for weighted
finite-state transducers, implemented in JAX.

## Algorithm Variants

36 algorithm variants span 4 axes of variation:

| Axis | Options | Description |
|------|---------|-------------|
| **Sequence** | TOK, PSWM | Observed tokens (one-hot) or position-specific weight matrix |
| **Dimension** | 1D, 2D | Generator/recognizer (one sequence) or transducer (two sequences) |
| **Kernel** | DENSE, SPARSE | Dense 4D tensor indexing or COO gather/scatter |
| **Strategy** | SIMPLE, OPTIMAL | Sequential scan or parallel prefix / wavefront |

SPARSE + OPTIMAL is excluded (requires sparse-sparse semiring matmul).
Each combination supports Forward, Backward, and Viterbi = 36 total.

All variants are JIT-compilable (no Python for-loops in any DP kernel).

### Implementation structure

| Strategy | Dimension | Implementation |
|----------|-----------|----------------|
| SIMPLE | 1D | `jax.lax.scan` over sequence positions |
| SIMPLE | 2D | Outer `jax.lax.scan` over rows, inner `jax.lax.associative_scan` over columns |
| OPTIMAL | 1D | `jax.lax.associative_scan` (parallel prefix) over transfer matrices |
| OPTIMAL | 2D | Outer `jax.lax.scan` over anti-diagonals, inner `jax.vmap` over diagonal cells |

## Main API

### Forward algorithm

```python
from machineboss.jax import log_forward

ll = log_forward(machine, input_seq=None, output_seq=None,
                 strategy='auto', kernel='auto', length=None)
```

Computes log P(input, output | machine). Returns a scalar.

### Viterbi algorithm

```python
from machineboss.jax import log_viterbi

ll = log_viterbi(machine, input_seq=None, output_seq=None,
                 strategy='auto', kernel='auto', length=None)
```

Computes log-probability of the most likely path. Returns a scalar.

### Backward algorithm

```python
from machineboss.jax import log_backward_matrix

bp = log_backward_matrix(machine, input_seq=None, output_seq=None,
                         strategy='auto', kernel='auto', length=None)
```

Returns the full backward matrix (for Forward-Backward expected counts).

### Forward-Backward expected counts

```python
from machineboss.jax.fwdback import log_likelihood_with_counts

ll, counts = log_likelihood_with_counts(machine, input_seq, output_seq)
```

Returns (log-likelihood, per-transition expected counts).

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `machine` | `JAXMachine` | Machine with dense and/or sparse transition tensors |
| `input_seq` | `jnp.ndarray`, `TokenSeq`, `PSWMSeq`, or `None` | Input sequence |
| `output_seq` | `jnp.ndarray`, `TokenSeq`, `PSWMSeq`, or `None` | Output sequence |
| `strategy` | `'auto'`, `'simple'`, `'optimal'` | DP strategy selection |
| `kernel` | `'auto'`, `'dense'`, `'sparse'` | Transition kernel selection |
| `length` | `int` or `None` | Real sequence length for padded sequences |

**Auto-dispatch rules:**
- `kernel='auto'`: uses `'dense'` if `machine.log_trans` is not None, else `'sparse'`
- `strategy='auto'`: uses `'optimal'` for 1D dense, `'simple'` otherwise
- 1D vs 2D: 1D if one sequence is `None`, 2D if both provided

## Token alphabet ordering

Token alphabets are **lexicographically sorted** by `Machine.input_alphabet()`
and `Machine.output_alphabet()`. Token index 0 is always the empty (silent)
token. Indices 1..N map to the sorted alphabet symbols.

For example, a machine with input symbols `{C, A, G, T}` produces
`input_tokens = ['', 'A', 'C', 'G', 'T']` with indices `[0, 1, 2, 3, 4]`.

This ordering is used throughout: `JAXMachine.log_trans`, `ParameterizedMachine`,
`TokenSeq`, `PSWMSeq`, and all DP algorithms. When constructing token index
arrays manually, use `pm.tokenize_input()` / `pm.tokenize_output()` or
`em.tokenize_input()` / `em.tokenize_output()` to map symbols to indices.

## Machine types

### Building a JAXMachine

```python
from machineboss.machine import Machine
from machineboss.eval import EvaluatedMachine
from machineboss.jax.types import JAXMachine

m = Machine.from_file("path/to/machine.json")
em = EvaluatedMachine.from_machine(m, params={"p": 0.9})
jm = JAXMachine.from_evaluated(em)
```

`JAXMachine` stores:
- `log_trans`: Dense transition tensor `(n_in, n_out, n_states, n_states)`, or None
- Sparse COO arrays: `src_states`, `dst_states`, `in_tokens`, `out_tokens`, `log_weights`
- `n_states`, `n_input_tokens`, `n_output_tokens`

Machine types are auto-detected:
- **Generator**: no input alphabet → use `output_seq` only (1D)
- **Recognizer**: no output alphabet → use `input_seq` only (1D)
- **Transducer**: both alphabets → use both sequences (2D) or either (1D)

## Sequence types

### TokenSeq — observed tokens

```python
from machineboss.jax import TokenSeq

seq = TokenSeq(jnp.array([1, 2, 1, 3]))  # token indices
# or pass a plain jnp.ndarray — auto-wrapped by dispatch functions
```

Emission weights are one-hot in log-space: 0.0 for the observed token, -inf for others.

### PSWMSeq — position-specific weight matrix

```python
from machineboss.jax import PSWMSeq

log_probs = jnp.array([
    [-0.1, -2.3, -2.3],  # position 0: mostly token 1
    [-2.3, -0.1, -2.3],  # position 1: mostly token 2
])
seq = PSWMSeq(log_probs)  # shape (L, n_tokens_including_empty)
```

Column 0 is the empty (silent) token — should be -inf for observed sequences.
Columns 1..n are log-probabilities for each alphabet symbol.

One-hot PSWM is equivalent to TokenSeq (verified by tests).

## Padding utilities

JAX recompiles kernels for each distinct input shape. Padding sequences to
a geometric-series bucket avoids excessive recompilation.

```python
from machineboss.jax import pad_length, pad_token_seq, pad_pswm_seq

padded_len = pad_length(37)  # → 48 (next bucket)
padded_seq, orig_len = pad_token_seq(tok_seq, padded_len)
padded_pswm, orig_len = pad_pswm_seq(pswm_seq, padded_len)
```

**Auto-padding**: The dispatch functions (`log_forward`, etc.) auto-pad 1D sequences
when `length` is not explicitly provided. This is invisible to the caller.

## Semirings

The same DP code handles Forward (sum-of-paths) and Viterbi (best-path) via semirings:

```python
from machineboss.jax import LOGSUMEXP, MAXPLUS

# LOGSUMEXP: plus = logaddexp, zero = -inf  → Forward/Backward
# MAXPLUS:   plus = maximum,   zero = -inf  → Viterbi
```

Internal engines accept a `semiring` parameter; the dispatch wrappers
(`log_forward`, `log_viterbi`) select the appropriate one.

## Fused Plan7 + Transducer

### Generic fused (any generator + transducer)

```python
from machineboss.jax.fused import FusedMachine, fused_log_forward, fused_log_viterbi

fused = FusedMachine.build(generator_em, transducer_em)
ll = fused_log_forward(fused, output_seq)
vit = fused_log_viterbi(fused, output_seq)
```

Avoids materializing the composite state space (like GeneWise).
Uses `jax.lax.scan` over output positions with vectorized inner operations.

### Plan7-aware fused (HMMER profile + transducer)

```python
from machineboss.hmmer import HmmerModel
from machineboss.jax.fused_plan7 import FusedPlan7Machine, fused_plan7_log_forward

with open("profile.hmm") as f:
    hmmer = HmmerModel.read(f)
fm = FusedPlan7Machine.build(hmmer, transducer_em)
ll = fused_plan7_log_forward(fm, output_seq)
```

Takes `HmmerModel` directly (not a generic Machine) to exploit the Plan7
linear chain structure:
- Outer `jax.lax.scan` over output positions
- Inner `jax.lax.scan` over core profile nodes k=1..K ({M, I, D} blocks)
- Flanking states (N, C, E, J) handled separately

Complexity: O(Lo × K × S_td²) instead of O(Lo × S_p7² × S_td).

## Parameterized (Neural) Transducer

Position-dependent weight expressions for neural transducer constructs.
The caller provides a Machine with weight expressions, PSWMs, and a dict
mapping each parameter name to a `(Li+1, Lo+1)` tensor of scalar values.

```python
from machineboss.machine import Machine
from machineboss.jax.jax_weight import ParameterizedMachine
from machineboss.jax.dp_neural import neural_log_forward, neural_log_viterbi

machine = Machine.from_file("transducer.json")  # with weight expressions
pm = ParameterizedMachine.from_machine(machine)

# Parameter tensors: each (Li+1, Lo+1), scalar at each DP position
params = {
    "p": jnp.full((Li + 1, Lo + 1), 0.9),    # or any position-dependent values
    "q": jnp.full((Li + 1, Lo + 1), 0.1),
}

ll = neural_log_forward(pm, input_pswm, output_pswm, params)
vit = neural_log_viterbi(pm, input_pswm, output_pswm, params)
```

At each cell (i, j), the transition tensor is built from `params[name][i, j]`
by evaluating the machine's weight expressions with JAX-traced operations.
JIT compiles the expression evaluation into the computation graph.

Differentiable via `jax.grad` — enables training with neural network parameters:

```python
def loss(nn_params):
    # Compute position-dependent transducer params from sequences via NN
    p_tensor = neural_net(nn_params, input_seq, output_seq)
    return -neural_log_forward(pm, input_pswm, output_pswm, {"p": p_tensor})

grad = jax.grad(loss)(nn_params)
```

Uses PSWM-2D-DENSE-SIMPLE strategy (nested `jax.lax.scan`, no associative scan
since silent closure varies by position). Fixed-iteration silent propagation
for differentiability.

### Parameter resolution and machine definitions

Parameters referenced by transition weight expressions are resolved in order:

1. **Caller's param dict** — position-dependent `(Li+1, Lo+1)` tensors supplied
   at runtime.
2. **Machine definitions** (`defs`) — numeric assignments or weight expressions
   from the machine JSON `"defs"` section.
3. **Error** — if a parameter is defined in neither place, a `KeyError` is raised.

This means machines with `"defs"` (parameter assignments or function definitions)
can be used directly without supplying values for every parameter. The caller only
needs to provide **free parameters** — those not defined by the machine itself.

```python
# Jukes-Cantor model: defs provide pNoSub, pSub, pSame, pDiff from free param t
machine = Machine.from_file("preset/jukescantor.json")
pm = ParameterizedMachine.from_machine(machine)

print(pm.param_names)   # {'t', 'pNoSub', 'pSub', 'pSame', 'pDiff'}
print(pm.free_params)   # {'t'}  — only t must be supplied

# Only supply the free parameter; defs handle the rest
params = {"t": jnp.full((Li + 1, Lo + 1), 0.5)}
ll = neural_log_forward(pm, input_pswm, output_pswm, params)
```

The caller can also **override** a machine-defined parameter by including it
in the param dict. This takes precedence over the machine's definition:

```python
# Override pSame directly instead of letting defs compute it from t
params = {
    "t": jnp.full((Li + 1, Lo + 1), 0.5),
    "pSame": jnp.full((Li + 1, Lo + 1), 0.95),  # overrides defs
}
```

Definition chains are compiled recursively (e.g. `pSame → pNoSub → t`) and
circular definitions are detected at compile time with a `ValueError`.

### TOK (tokenized sequence) wrappers

For convenience, `_tok` variants accept token index arrays instead of PSWMs:

```python
from machineboss.jax.dp_neural import (
    neural_log_forward_tok, neural_log_viterbi_tok, neural_log_backward_matrix_tok,
)

in_toks = jnp.array(pm.tokenize_input(list("ACGT")), dtype=jnp.int32)
out_toks = jnp.array(pm.tokenize_output(list("ACGA")), dtype=jnp.int32)
params = {"t": jnp.full((Li + 1, Lo + 1), 0.5)}

ll = neural_log_forward_tok(pm, in_toks, out_toks, params)
```

These convert tokens to one-hot PSWMs internally. Results are identical to the
PSWM versions with one-hot inputs.

### Broadcast parameter shapes

Parameter tensors can have any shape broadcastable to `(Li+1, Lo+1)`:
`(Li+1, Lo+1)`, `(Li+1, 1)`, `(1, Lo+1)`, or `(1, 1)`. Size-1 axes are
handled by index clamping — the full broadcast tensor is **never materialized**.
Gradients preserve the input shape (e.g. a `(Li+1, 1)` parameter produces
a `(Li+1, 1)` gradient).

## Alignment-Constrained DP

Alignment-constrained variants visit only the cells of the 2D DP matrix
touched by a prescribed pairwise alignment, using 1D memory.

```python
from machineboss.jax.dp_aligned import (
    aligned_log_forward, aligned_log_viterbi,
    neural_aligned_log_forward, neural_aligned_log_viterbi,
    validate_alignment, MAT, INS, DEL,
)

# Alignment: sequence of MAT=0, INS=1, DEL=2
# MAT: consume input + output (i++, j++)
# INS: consume input only (i++)
# DEL: consume output only (j++)
alignment = jnp.array([MAT, MAT, INS, DEL, MAT], dtype=jnp.int32)
validate_alignment(alignment, Li=4, Lo=4)

# Standard (fixed-weight) aligned DP
ll = aligned_log_forward(jax_machine, input_tokens, output_tokens, alignment)
vit = aligned_log_viterbi(jax_machine, input_tokens, output_tokens, alignment)

# Neural (position-dependent) aligned DP
ll = neural_aligned_log_forward(pm, in_toks, out_toks, alignment, params)
vit = neural_aligned_log_viterbi(pm, in_toks, out_toks, alignment, params)
```

The DP scans along the alignment in a single `jax.lax.scan`, tracking
current (i, j) position. Complexity: O(A × S²) where A is alignment length,
compared to O(Li × Lo × S²) for unconstrained 2D. Differentiable via `jax.grad`.

## Module structure

| Module | Purpose |
|--------|---------|
| `forward.py` | `log_forward()` dispatch wrapper |
| `backward.py` | `log_backward_matrix()` dispatch wrapper |
| `viterbi.py` | `log_viterbi()` dispatch wrapper |
| `fwdback.py` | Forward-Backward expected counts |
| `semiring.py` | `LOGSUMEXP` and `MAXPLUS` semiring definitions |
| `seq.py` | `TokenSeq`, `PSWMSeq`, padding utilities |
| `types.py` | `JAXMachine` data class |
| `kernel_dense.py` | Dense transition operations (propagate_silent, emit_step) |
| `kernel_sparse.py` | Sparse COO transition operations |
| `dp_1d_simple.py` | 1D DP via `jax.lax.scan` |
| `dp_1d_optimal.py` | 1D DP via `jax.lax.associative_scan` |
| `dp_2d_simple.py` | 2D DP: outer scan + inner associative scan |
| `dp_2d_optimal.py` | 2D DP: anti-diagonal wavefront (scan + vmap) |
| `fused.py` | Generic fused Plan7+transducer DP |
| `fused_plan7.py` | Plan7-aware fused DP with nested scans |
| `jax_weight.py` | Weight expression compiler (`ParameterizedMachine`) |
| `dp_neural.py` | Parameterized 2D DP with position-dependent weights (PSWM + TOK) |
| `dp_aligned.py` | Alignment-constrained 1D DP (standard + neural) |

## Testing

```bash
cd python && ~/jax-env/bin/python -m pytest machineboss/ -v
```

Test categories:
- **Ground truth**: all TOK variants compared against C++ `boss -L` / `boss -V`
- **Cross-variant**: all variants for same (machine, sequences) agree within 0.01 nats
- **PSWM = TOK**: one-hot PSWM matches TOK exactly
- **1D = 2D**: 1D generator matches 2D with empty input
- **Viterbi ≤ Forward**: invariant check
- **Backward[start] = Forward**: log-likelihood consistency
- **Forward-Backward counts vs C++ `boss -C`**: per-parameter expected counts match
- **Fused Plan7 vs generic fused**: matching log-likelihoods for HMMER profiles
- **Neural = standard**: constant params match standard forward/viterbi
- **Neural backward = forward**: backward[start] equals forward log-likelihood
- **Neural grad**: JAX autodiff through position-dependent parameters
- **Neural vs C++ boss**: constant-param neural matches `boss -L`
- **Defs fallback**: machine definitions resolve when caller omits parameters
- **Defs override**: caller-supplied values take precedence over machine defs
- **Defs chain**: chained definitions (e.g. Jukes-Cantor pSame → pNoSub → t)
- **Defs circular**: circular definitions detected at compile time
- **Jukes-Cantor vs C++ boss**: defs-based model matches `boss -L -P`
- **Neural TOK = PSWM**: tokenized wrappers match one-hot PSWM exactly
- **Broadcast params**: (Li+1,1), (1,Lo+1), (1,1) match full (Li+1,Lo+1)
- **Broadcast grad shape**: gradient shape matches input parameter shape
- **Aligned ≤ unconstrained**: alignment-constrained forward ≤ unconstrained
- **Aligned Viterbi ≤ Forward**: invariant within aligned DP
- **Neural aligned = standard aligned**: constant params match fixed-weight
- **Aligned grad**: JAX autodiff through neural aligned DP
