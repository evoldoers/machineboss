---
title: Python/JAX API
nav_order: 9
permalink: /python/
---

# Python/JAX API Reference
{: .no_toc }

The `machineboss` Python package provides pure-Python data classes for weighted finite-state transducers,
a subprocess wrapper for the `boss` CLI, an HMMER3 parser, and a JAX subpackage with
GPU-accelerated, differentiable dynamic programming algorithms.

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Installation

```bash
cd python
pip install .               # core package
pip install ".[jax]"        # with JAX support
```

Requires Python 3.10+. The JAX extras install `jax>=0.4.0` and `jaxlib>=0.4.0`.

## Package Overview

```
machineboss/
  machine.py      Core data classes: Machine, MachineState, MachineTransition
  weight.py       Weight expression algebra and symbolic differentiation
  eval.py         Numeric evaluation of machines (tokenization + log-weights)
  boss.py         Subprocess wrapper for the bin/boss CLI
  hmmer.py        HMMER3 profile HMM parser
  seqpair.py      Sequence pair and envelope types
  io.py           File I/O utilities

  jax/            JAX subpackage (GPU-accelerated DP)
    trans/        Primary JAX interface — transition-centric DP
      machine.py    TransMachine (JAX pytree, COO arrays + masks)
      kernel.py     Core scatter/gather operations
      dp_1d.py      1D forward/backward/viterbi (simple + optimal)
      dp_2d.py      2D forward/backward/viterbi (simple + wavefront)
      fwdback.py    Vectorized forward-backward with expected counts
      dp_aligned.py Alignment-constrained DP
      dp_neural.py  Position-dependent (neural) DP
      parameterized.py  ParameterizedTransMachine
      dp_fused.py   Fused Plan7 + transducer DP
      dp_beam.py    Wavefront beam Viterbi
    types.py      JAXMachine (bridge type for legacy code)
    semiring.py   Log-space semirings (log-sum-exp, max-plus)
    seq.py        Sequence representations (TokenSeq, PSWMSeq)
    jax_weight.py   ParameterizedMachine (weight expression compiler)
    fused_plan7.py  Fused Plan7 + transducer DP (optimized)
```

---

## Core Modules

### `machine.py` --- Machine, MachineState, MachineTransition

These dataclasses mirror the [JSON transducer format](/json-format/).

#### `MachineTransition`

A single WFST transition.

| Attribute | Type | Description |
|---|---|---|
| `dest` | `int` | Destination state index |
| `weight` | `int`, `float`, `str`, or `dict` | JSON weight expression (default 1) |
| `input` | `str` or `None` | Input symbol (`None` for silent) |
| `output` | `str` or `None` | Output symbol (`None` for silent) |

| Method | Description |
|---|---|
| `from_json(j)` | Class method. Parse from JSON dict with keys `"to"`, `"weight"`, `"in"`, `"out"` |
| `to_json()` | Serialize to JSON dict |
| `is_silent` | Property. `True` if no input and no output |

#### `MachineState`

A single WFST state with outgoing transitions.

| Attribute | Type | Description |
|---|---|---|
| `trans` | `list[MachineTransition]` | Outgoing transitions |
| `name` | `Any` | Optional state name (string, int, or list) |

#### `Machine`

A complete weighted finite-state transducer.

| Attribute | Type | Description |
|---|---|---|
| `state` | `list[MachineState]` | States (first = start, last = end) |
| `defs` | `dict[str, Any]` | Parameter/function definitions |

| Method | Description |
|---|---|
| `from_json(j)` | Class method. Parse from JSON dict or string |
| `from_file(path)` | Class method. Load from a JSON file |
| `to_json()` | Serialize to JSON dict |
| `to_json_string(indent=None)` | Serialize to JSON string |
| `n_states` | Property. Number of states |
| `start_state` | Property. Always 0 |
| `end_state` | Property. Always `n_states - 1` |
| `n_transitions` | Property. Total transition count |
| `input_alphabet()` | Sorted list of distinct input symbols |
| `output_alphabet()` | Sorted list of distinct output symbols |

```python
from machineboss.machine import Machine

# Load from a JSON file
m = Machine.from_file("preset/jukescantor.json")

# Parse inline JSON
m = Machine.from_json({
    "state": [
        {"n": "S", "trans": [{"to": "E", "in": "a", "out": "b"}]},
        {"n": "E"}
    ]
})

print(m.n_states)          # 2
print(m.input_alphabet())  # ['a']
```

---

### `weight.py` --- Weight Expression Algebra

Symbolic construction and evaluation of [weight expressions](/expressions/).

#### Constants

- `ZERO = 0`
- `ONE = 1`

#### Builder Functions

All return a simplified `WeightExpr` (int, float, str, or dict) with constant-folding for zeros and ones.

| Function | Result |
|---|---|
| `multiply(a, b)` | `{"*": [a, b]}` |
| `add(a, b)` | `{"+": [a, b]}` |
| `subtract(a, b)` | `{"-": [a, b]}` |
| `divide(a, b)` | `{"/": [a, b]}` |
| `power(base, exp)` | `{"pow": [base, exp]}` |
| `log_of(x)` | `{"log": x}` |
| `exp_of(x)` | `{"exp": x}` |
| `negate(x)` | `1 - x` (probability complement) |
| `reciprocal(x)` | `1 / x` |

#### Evaluation and Differentiation

| Function | Description |
|---|---|
| `evaluate(w, params=None)` | Numerically evaluate a weight expression |
| `differentiate(w, param)` | Symbolic differentiation (product rule, chain rule, etc.) |
| `params(w)` | Return set of all parameter names in an expression |

```python
from machineboss.weight import multiply, evaluate, differentiate, params

expr = multiply("p", "q")        # {"*": ["p", "q"]}
v = evaluate(expr, {"p": 0.9, "q": 0.1})  # 0.09
d = differentiate(expr, "p")     # "q"
p = params(expr)                 # {"p", "q"}
```

---

### `eval.py` --- Evaluated Machine

Bridges symbolic JSON machines and numerical DP by tokenizing alphabets and evaluating
all weights to log-space floats.

#### `EvaluatedTransition`

| Attribute | Type | Description |
|---|---|---|
| `src` | `int` | Source state index |
| `dst` | `int` | Destination state index |
| `in_tok` | `int` | Input token index (0 = silent) |
| `out_tok` | `int` | Output token index (0 = silent) |
| `log_weight` | `float` | Log-weight |

#### `EvaluatedMachine`

| Attribute | Type | Description |
|---|---|---|
| `n_states` | `int` | Number of states |
| `input_tokens` | `list[str]` | Token list (index 0 = empty string) |
| `output_tokens` | `list[str]` | Token list (index 0 = empty string) |
| `transitions` | `list[EvaluatedTransition]` | All transitions |

| Method | Description |
|---|---|
| `from_machine(machine, params=None)` | Class method. Evaluate all weights; drop zero-weight transitions |
| `transitions_from(src)` | Transitions originating from a given state |
| `tokenize_input(seq)` | Convert symbol list to token index list |
| `tokenize_output(seq)` | Convert symbol list to token index list |

```python
from machineboss.machine import Machine
from machineboss.eval import EvaluatedMachine

m = Machine.from_file("preset/jukescantor.json")
em = EvaluatedMachine.from_machine(m, params={"t": 0.5})
print(em.n_states, len(em.transitions))
```

---

### `boss.py` --- CLI Subprocess Wrapper

Wraps the `bin/boss` binary for operations not yet available in pure Python.

#### `Boss`

| Method | Description |
|---|---|
| `Boss(executable=None)` | Find `bin/boss` relative to repo root or on `PATH` |
| `run(*args, input_json=None, timeout=60)` | Run boss; return stdout |
| `run_json(*args, ...)` | Run boss; parse JSON output |
| `load_machine(*args)` | Run boss; return a `Machine` |
| `compose(m1, m2)` | Compose two machines via the CLI |
| `forward(machine, input_seq=None, output_seq=None, params=None)` | Forward algorithm; return log-likelihood |

```python
from machineboss.boss import Boss

b = Boss()
m = b.load_machine("--preset", "jukescantor")
ll = b.forward(m, input_seq=list("ACGT"), output_seq=list("ACGA"),
               params={"t": 0.5})
```

---

### `hmmer.py` --- HMMER3 Parser

Pure-Python parser for HMMER3 profile HMM files. Builds Machine objects compatible
with the [Plan7 architecture](/machineboss/#external-formats).

#### `HmmerModel`

| Method | Description |
|---|---|
| `HmmerModel.read(f)` | Class method. Parse an HMMER3 format file |
| `machine(local=True)` | Build core HMM as a `Machine` (local alignment by default) |
| `plan7_machine(multihit=False, L=400)` | Build full Plan7 machine with N/C/J flanking states |
| `calc_match_occupancy()` | Compute match-state occupancy for local entry weighting |

| Attribute | Description |
|---|---|
| `alph` | Alphabet (e.g. 20 amino acids) |
| `nodes` | List of `HmmerNode` (per-position match/insert emissions and transitions) |

```python
from machineboss.hmmer import HmmerModel

with open("model.hmm") as f:
    hmm = HmmerModel.read(f)
m = hmm.plan7_machine(multihit=True, L=400)
print(m.n_states, m.n_transitions)
```

---

### `seqpair.py` --- Sequence Pairs

#### `SeqPair`

| Attribute | Type | Description |
|---|---|---|
| `input` | `list[str]` | Input token sequence |
| `output` | `list[str]` | Output token sequence |

| Method | Description |
|---|---|
| `from_json(j)` | Parse from JSON dict |
| `from_strings(input_str, output_str)` | Split strings into character lists |
| `to_json()` | Serialize to JSON |

#### `Envelope`

Constrains DP computation to a sub-region of the sequence pair grid.

| Attribute | Type |
|---|---|
| `input_start`, `input_end` | `int` |
| `output_start`, `output_end` | `int` |

---

### `io.py` --- File I/O

| Function | Description |
|---|---|
| `load_machine(path)` | Load a `Machine` from a JSON file |
| `save_machine(machine, path, indent=None)` | Save a `Machine` to a JSON file |
| `load_params(path)` | Load parameter dict from JSON |
| `save_params(params, path)` | Save parameter dict to JSON |
| `load_seqpair(path)` | Load a `SeqPair` from JSON |
| `load_seqpair_list(path)` | Load a list of `SeqPair` from JSON |

---

## JAX Subpackage

The `machineboss.jax` subpackage provides GPU-accelerated, JIT-compiled, and
differentiable dynamic programming algorithms using [JAX](https://jax.readthedocs.io/).

The primary interface is `TransMachine` — a transition-centric WFST representation
registered as a JAX pytree, fully compatible with `jit`, `grad`, and `vmap`.

### TransMachine

**The Machine IS the list of Transitions.**

`TransMachine` stores a WFST as parallel COO arrays with pre-built boolean masks.
This is the recommended type for all JAX DP operations.

```python
from machineboss.machine import Machine
from machineboss.jax.trans import TransMachine, forward_2d
import jax.numpy as jnp

# Load from JSON and convert
m = Machine.from_file("preset/jukescantor.json")
tm = TransMachine.from_machine(m, params={"t": 0.5})

# Tokenize sequences and compute log-likelihood
input_seq = jnp.array(tm.to_jax_machine().from_evaluated(
    EvaluatedMachine.from_machine(m, {"t": 0.5})).tokenize_input(list("ACGT")))
# or simply use integer token indices directly:
input_seq = jnp.array([1, 2, 3, 4])   # 1-based token indices
output_seq = jnp.array([1, 2, 3, 1])
ll = forward_2d(tm, input_seq, output_seq)
```

| Attribute | Shape | Description |
|---|---|---|
| `src` | `(T,)` int32 | Source state indices |
| `dst` | `(T,)` int32 | Destination state indices |
| `in_tok` | `(T,)` int32 | Input tokens (0 = silent) |
| `out_tok` | `(T,)` int32 | Output tokens (0 = silent) |
| `log_w` | `(T,)` float32 | Transition log-weights |
| `silent_mask` | `(T,)` bool | `in_tok==0 & out_tok==0` |
| `emit_in_mask` | `(T,)` bool | `in_tok>0 & out_tok==0` |
| `emit_out_mask` | `(T,)` bool | `in_tok==0 & out_tok>0` |
| `emit_both_mask` | `(T,)` bool | `in_tok>0 & out_tok>0` |
| `n_states` | int | Number of states |
| `n_in`, `n_out` | int | Token counts (including empty at 0) |
| `input_tokens`, `output_tokens` | tuple[str] | Token lists |

#### Constructors

| Method | Description |
|---|---|
| `TransMachine.from_machine(m, params=None)` | From a `Machine` (evaluates weight expressions) |
| `TransMachine.from_evaluated(em)` | From an `EvaluatedMachine` |
| `TransMachine.from_jax_machine(jm)` | From a `JAXMachine` |

#### Conversions (for C++/JSON interop)

| Method | Description |
|---|---|
| `tm.to_machine()` | Reconstruct a `Machine` (JSON-serializable) |
| `tm.to_jax_machine()` | Convert to `JAXMachine` (legacy bridge) |

#### JAX Pytree

`TransMachine` is a registered JAX pytree. Arrays are children (traced by `jit`/`grad`);
metadata (n_states, token lists) is aux (static). This means you can pass a `TransMachine`
directly to `jax.jit`, `jax.grad`, and `jax.vmap`.

```python
@jax.jit
def log_prob(tm, in_seq, out_seq):
    return forward_2d(tm, in_seq, out_seq)

ll = log_prob(tm, input_seq, output_seq)
```

---

### Forward, Backward, Viterbi

All DP functions operate on `TransMachine` and support two strategies:
- **simple**: sequential `jax.lax.scan` (O(L) or O(Li * Lo))
- **optimal**: parallel prefix scan (1D, O(log L)) or anti-diagonal wavefront (2D, O(Li + Lo) with vmap)

```python
from machineboss.jax.trans import (
    forward_1d, backward_1d, viterbi_1d,   # generators/recognizers
    forward_2d, backward_2d, viterbi_2d,   # transducers
    forward_2d_matrix,                      # returns full (Li+1, Lo+1, S) grid
)
```

#### 1D DP (generators and recognizers)

```python
# Generator: output only
ll = forward_1d(tm, output_seq=jnp.array([1, 2, 3]))
bp = backward_1d(tm, output_seq=jnp.array([1, 2, 3]))  # (L+1, S)
vit = viterbi_1d(tm, output_seq=jnp.array([1, 2, 3]))

# Recognizer: input only
ll = forward_1d(tm, input_seq=jnp.array([1, 2, 3]))
```

#### 2D DP (transducers)

```python
ll = forward_2d(tm, input_seq, output_seq)
bp = backward_2d(tm, input_seq, output_seq)  # (Li+1, Lo+1, S)
vit = viterbi_2d(tm, input_seq, output_seq)
```

**Parameters:**

| Parameter | Values | Description |
|---|---|---|
| `tm` | `TransMachine` | The machine to run |
| `input_seq` | `jnp.ndarray` or `None` | 1-based token array |
| `output_seq` | `jnp.ndarray` or `None` | 1-based token array |
| `semiring` | `LOGSUMEXP`, `MAXPLUS` | Sum-of-paths or best-path (default `LOGSUMEXP`) |
| `strategy` | `'auto'`, `'simple'`, `'optimal'` | Algorithm variant |

---

### Semirings

Two log-space semirings control whether DP sums over paths (Forward) or maximizes (Viterbi).

```python
from machineboss.jax.trans import forward_2d
from machineboss.jax.semiring import LOGSUMEXP, MAXPLUS

ll = forward_2d(tm, in_seq, out_seq, LOGSUMEXP)   # Forward (default)
vit = forward_2d(tm, in_seq, out_seq, MAXPLUS)     # Viterbi
```

---

### Sequence Representations

#### `TokenSeq`

A sequence of discrete tokens for standard DP.

```python
from machineboss.jax.seq import TokenSeq

seq = TokenSeq(tokens=jnp.array([1, 3, 2, 4]))
```

#### `PSWMSeq` (Position-Specific Weight Matrix)

A soft/probabilistic sequence where each position has a log-probability distribution
over tokens. Used for neural network outputs and profile HMMs.

```python
from machineboss.jax.seq import PSWMSeq

# L=3 positions, 5 tokens (including empty at index 0)
log_probs = jnp.array([[-1e38, 0.0, -2.0, -2.0, -2.0],
                        [-1e38, -2.0, 0.0, -2.0, -2.0],
                        [-1e38, -2.0, -2.0, 0.0, -2.0]])
seq = PSWMSeq(log_probs=log_probs)
```

Both `TokenSeq` and `PSWMSeq` are accepted by all DP functions via `wrap_seq`.
Raw `jnp.ndarray` token arrays are automatically wrapped as `TokenSeq`.

---

### Forward-Backward with Expected Counts

Fully vectorized forward-backward that computes per-transition expected counts
using `vmap` over transitions and position-broadcasting. No Python for-loops.

```python
from machineboss.jax.trans import forward_backward, forward_backward_1d

# 2D transducer
ll, counts = forward_backward(tm, input_seq, output_seq)
# ll: log-likelihood (scalar)
# counts: (T,) expected count per transition

# 1D generator
ll, counts = forward_backward_1d(tm, output_seq=output_seq)
```

Supports `strategy='simple'` and `strategy='optimal'` for the underlying
forward/backward passes. The counts computation itself is always vectorized.

---

### Parameterized Machines (Neural DP)

`ParameterizedTransMachine` compiles weight expressions into JAX-traceable
functions that produce `log_w` vectors (shape `(T,)`) instead of dense tensors.
This enables gradient flow through machine parameters for neural transducers.

```python
from machineboss.jax.trans import ParameterizedTransMachine
from machineboss.jax.trans import neural_forward_2d
from machineboss.jax.semiring import LOGSUMEXP

m = Machine.from_file("preset/jukescantor.json")
ptm = ParameterizedTransMachine.from_machine(m)

print(ptm.free_params)   # {'t'} — params the caller must supply

# Position-independent params (scalars broadcast to all positions)
params = {"t": jnp.array([[0.5]])}   # shape (1, 1) for broadcasting
ll = neural_forward_2d(ptm, input_pswm, output_pswm, params, LOGSUMEXP)
```

The `ParameterizedMachine` class in `jax_weight.py` provides the same
functionality but produces dense `(n_in, n_out, S, S)` tensors. Use
`ParameterizedTransMachine` for the transition-centric interface; use
`ParameterizedMachine` when you need dense tensors or are working with
legacy code.

---

### Alignment-Constrained DP

DP along a prescribed alignment path instead of the full `(Li+1, Lo+1)` grid.
Useful for training on known alignments.

```python
from machineboss.jax.trans import (
    aligned_forward, aligned_viterbi,
    neural_aligned_forward, neural_aligned_viterbi,
    validate_alignment, MAT, INS, DEL,
)

# Alignment: MAT=match (consume both), INS=insert (input only), DEL=delete (output only)
alignment = jnp.array([MAT, INS, MAT, DEL, MAT])

ll = aligned_forward(tm, input_tokens, output_tokens, alignment, LOGSUMEXP)
```

---

### Fused Plan7 + Transducer DP

Avoids the cost of explicitly composing a Plan7 HMM with a transducer
by interleaving their DP recurrences (GeneWise-style).

```python
from machineboss.jax.trans import fused_forward, fused_viterbi

ll = fused_forward(plan7_model, transducer_tm, output_seq)
```

#### Plan7-Optimized Fusion (`fused_plan7.py`)

For direct access to the optimized Plan7 kernel (O(S_td) per node):

```python
from machineboss.hmmer import HmmerModel
from machineboss.jax.fused_plan7 import FusedPlan7Machine, fused_plan7_log_forward

with open("model.hmm") as f:
    hmm = HmmerModel.read(f)

td_em = EvaluatedMachine.from_machine(transducer, params)
fm = FusedPlan7Machine.build(hmm, td_em, multihit=True, L=400)
ll = fused_plan7_log_forward(fm, output_seq)
```

---

### Beam-Viterbi for Cyclic Machines

Wavefront beam search for machines with cycles (TKF92, Plan7) where
standard Viterbi requires topological sort.

```python
from machineboss.jax.trans import beam_align

result = beam_align(tm, input_seq, output_seq, beam_width=100)
print(result.score, result.path)
```

---

### Bridge Types (JSON/C++ Interop)

The `JAXMachine` type is a bridge between the JSON machine format and JAX arrays.
It stores transitions in sparse COO format plus an optional dense 4D tensor.
Use it when interfacing with legacy code or the `log_forward`/`log_viterbi` dispatchers.

```python
from machineboss.jax import JAXMachine, log_forward

jm = JAXMachine.from_evaluated(em)
ll = log_forward(jm, input_seq, output_seq)

# Convert between representations
tm = TransMachine.from_jax_machine(jm)
jm2 = tm.to_jax_machine()
```

---

## Typical Workflows

### Log-likelihood of a sequence pair

```python
from machineboss.machine import Machine
from machineboss.eval import EvaluatedMachine
from machineboss.jax.trans import TransMachine, forward_2d
import jax.numpy as jnp

m = Machine.from_file("preset/jukescantor.json")
tm = TransMachine.from_machine(m, params={"t": 0.5})

em = EvaluatedMachine.from_machine(m, {"t": 0.5})
input_seq = jnp.array(em.tokenize_input(list("ACGT")))
output_seq = jnp.array(em.tokenize_output(list("ACGA")))
ll = forward_2d(tm, input_seq, output_seq)
```

### Gradient-based parameter fitting

```python
import jax
import jax.numpy as jnp
from machineboss.machine import Machine
from machineboss.jax.trans import ParameterizedTransMachine, neural_forward_2d_tok
from machineboss.jax.semiring import LOGSUMEXP

m = Machine.from_file("preset/jukescantor.json")
ptm = ParameterizedTransMachine.from_machine(m)

def neg_ll(t):
    params = {"t": jnp.array([[t]])}
    return -neural_forward_2d_tok(ptm, input_tokens, output_tokens, params)

grad_fn = jax.grad(neg_ll)
t = jnp.float32(1.0)
for _ in range(100):
    t = t - 0.01 * grad_fn(t)
```

### Expected transition counts

```python
from machineboss.jax.trans import TransMachine, forward_backward

m = Machine.from_file("preset/jukescantor.json")
tm = TransMachine.from_machine(m, params={"t": 0.5})

ll, counts = forward_backward(tm, input_seq, output_seq)
# counts[i] = expected usage of transition i
```

### HMMER protein search with fused DP

```python
from machineboss.hmmer import HmmerModel
from machineboss.eval import EvaluatedMachine
from machineboss.machine import Machine
from machineboss.jax.fused_plan7 import FusedPlan7Machine, fused_plan7_log_forward
import jax.numpy as jnp

# Load HMMER model and transducer
with open("profile.hmm") as f:
    hmm = HmmerModel.read(f)
td = Machine.from_file("preset/translate.json")
td_em = EvaluatedMachine.from_machine(td)

# Build fused machine
fm = FusedPlan7Machine.build(hmm, td_em, multihit=True)

# Score a DNA sequence
output_seq = jnp.array([...])  # tokenized DNA
ll = fused_plan7_log_forward(fm, output_seq)
```
