---
title: Python/JAX API
nav_order: 8
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
    types.py      JAXMachine (sparse + dense tensor representation)
    semiring.py   Log-space semirings (log-sum-exp, max-plus)
    seq.py        Sequence representations (TokenSeq, PSWMSeq)
    forward.py    Forward algorithm dispatcher
    backward.py   Backward algorithm dispatcher
    viterbi.py    Viterbi algorithm dispatcher
    fwdback.py    Forward-Backward with custom VJP for autodiff
    fused.py      Fused Plan7 + transducer DP (generic)
    fused_plan7.py  Fused Plan7 + transducer DP (optimized)
    jax_weight.py   Parameterized machines with JAX-compiled weights
    dp_neural.py    Position-dependent (neural) DP
    dp_aligned.py   Alignment-constrained DP
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

All public symbols are re-exported from `machineboss.jax`.

### JAXMachine

The `JAXMachine` is the JAX-ready representation of a WFST, built from an `EvaluatedMachine`.
It stores transitions in both sparse COO format (for large machines) and an optional dense
4D tensor `log_trans[in_tok, out_tok, src, dst]` (for machines with fewer than 100 states).

```python
from machineboss.machine import Machine
from machineboss.eval import EvaluatedMachine
from machineboss.jax import JAXMachine

m = Machine.from_file("preset/jukescantor.json")
em = EvaluatedMachine.from_machine(m, {"t": 0.5})
jm = JAXMachine.from_evaluated(em)

print(jm.machine_type())   # 'transducer'
print(jm.n_states)         # 2
print(jm.log_trans.shape)  # (5, 5, 2, 2)
```

| Attribute | Shape | Description |
|---|---|---|
| `log_weights` | `(T,)` | Transition log-weights |
| `src_states` | `(T,)` | Source state indices |
| `dst_states` | `(T,)` | Destination state indices |
| `in_tokens` | `(T,)` | Input tokens (0 = silent) |
| `out_tokens` | `(T,)` | Output tokens (0 = silent) |
| `log_trans` | `(n_in, n_out, S, S)` | Dense tensor (if `n_states <= dense_threshold`) |

---

### Forward, Backward, Viterbi

The three main DP dispatchers automatically select the best algorithm variant
based on machine type (generator, recognizer, or transducer) and sequence dimensions.

```python
from machineboss.jax import log_forward, log_viterbi, log_backward_matrix
```

#### `log_forward(machine, input_seq=None, output_seq=None, *, strategy='auto', kernel='auto', length=None)`

Compute log-likelihood by summing over all paths (Forward algorithm).

#### `log_viterbi(machine, input_seq=None, output_seq=None, *, strategy='auto', kernel='auto', length=None)`

Compute log-weight of the most likely path (Viterbi algorithm).

#### `log_backward_matrix(machine, input_seq=None, output_seq=None, *, strategy='auto', kernel='auto', length=None)`

Compute the full backward matrix.

**Parameters:**

| Parameter | Values | Description |
|---|---|---|
| `machine` | `JAXMachine` | The machine to run |
| `input_seq` | `jnp.ndarray` or `None` | 1-based token array (for recognizers and transducers) |
| `output_seq` | `jnp.ndarray` or `None` | 1-based token array (for generators and transducers) |
| `strategy` | `'auto'`, `'simple'`, `'optimal'` | `'simple'` uses `jax.lax.scan`; `'optimal'` uses associative scan (1D) or wavefront (2D) |
| `kernel` | `'auto'`, `'dense'`, `'sparse'` | `'dense'` uses tensor indexing; `'sparse'` uses COO scatter/gather |
| `length` | `int` or `None` | Real length for padded sequences |

**Automatic strategy selection:**

| Dimensionality | Kernel | Default strategy |
|---|---|---|
| 1D | dense | optimal (associative scan, O(log L) depth) |
| 1D | sparse | simple (sequential scan) |
| 2D | dense | simple |
| 2D | sparse | simple |

```python
import jax.numpy as jnp
from machineboss.jax import JAXMachine, log_forward, log_viterbi

# 2D transducer DP
ll = log_forward(jm, jnp.array([1, 2, 3, 1]), jnp.array([1, 2, 3, 1]))

# 1D recognizer DP
em_rec = EvaluatedMachine.from_machine(recognizer_machine)
jm_rec = JAXMachine.from_evaluated(em_rec)
ll = log_forward(jm_rec, jnp.array([1, 2, 3]))
```

---

### Sequence Representations

#### `TokenSeq`

A sequence of discrete tokens for standard DP.

```python
from machineboss.jax import TokenSeq

seq = TokenSeq(tokens=jnp.array([1, 3, 2, 4]))
```

#### `PSWMSeq` (Position-Specific Weight Matrix)

A soft/probabilistic sequence where each position has a log-probability distribution
over tokens. Used for neural network outputs and profile HMMs.

```python
from machineboss.jax import PSWMSeq

# L=3 positions, 5 tokens (including empty at index 0)
log_probs = jnp.array([[-1e38, 0.0, -2.0, -2.0, -2.0],
                        [-1e38, -2.0, 0.0, -2.0, -2.0],
                        [-1e38, -2.0, -2.0, 0.0, -2.0]])
seq = PSWMSeq(log_probs=log_probs)
```

---

### Semirings

Two log-space semirings control whether DP sums over paths (Forward) or maximizes (Viterbi).

```python
from machineboss.jax import LOGSUMEXP, MAXPLUS

LOGSUMEXP.plus(a, b)   # jnp.logaddexp(a, b) — for Forward
MAXPLUS.plus(a, b)      # jnp.maximum(a, b)   — for Viterbi
```

---

### Forward-Backward with Autodiff

`fwdback.py` provides `jax.custom_vjp`-compatible functions for differentiable log-likelihood,
enabling gradient-based parameter optimization.

```python
from machineboss.jax.fwdback import log_likelihood_with_counts

ll, counts = log_likelihood_with_counts(jm, input_seq, output_seq)
# ll: log-likelihood (scalar)
# counts: expected transition usage counts (shape (T,))
```

---

### Parameterized Machines (Neural DP)

`ParameterizedMachine` compiles weight expressions into JAX-traceable functions,
enabling gradient flow through machine parameters. This supports neural transducers
where transition weights are produced by a neural network.

```python
from machineboss.jax.jax_weight import ParameterizedMachine

m = Machine.from_file("preset/jukescantor.json")
pm = ParameterizedMachine.from_machine(m)

print(pm.free_params)   # {'t'} — params the caller must supply
print(pm.param_names)   # {'t', 'pNoSub', 'pSub', ...}

# Build log-weight tensor from parameters
import jax.numpy as jnp
lt = pm.build_log_trans({"t": jnp.float32(0.5)})
# lt.shape = (n_in, n_out, S, S)
```

#### Neural DP Functions

Position-dependent DP where weight parameters can vary at each `(i, j)` position
in the sequence pair grid. Parameter tensors must be broadcastable to `(Li+1, Lo+1)`.

| Function | Description |
|---|---|
| `neural_log_forward(pm, input_pswm, output_pswm, params)` | Forward with PSWM sequences |
| `neural_log_viterbi(pm, input_pswm, output_pswm, params)` | Viterbi with PSWM sequences |
| `neural_log_backward_matrix(pm, input_pswm, output_pswm, params)` | Backward with PSWM sequences |
| `neural_log_forward_tok(pm, input_tokens, output_tokens, params)` | Forward with token arrays |
| `neural_log_viterbi_tok(pm, input_tokens, output_tokens, params)` | Viterbi with token arrays |
| `neural_log_backward_matrix_tok(pm, input_tokens, output_tokens, params)` | Backward with token arrays |

```python
from machineboss.jax.dp_neural import neural_log_forward_tok

# Position-independent params (scalars broadcast to all positions)
params = {"t": jnp.float32(0.5)}
ll = neural_log_forward_tok(pm, input_tokens, output_tokens, params)

# Position-dependent params (vary per grid cell)
# Shape (Li+1, Lo+1) or broadcastable
params = {"t": jnp.ones((Li+1, Lo+1)) * 0.5}
ll = neural_log_forward_tok(pm, input_tokens, output_tokens, params)
```

---

### Alignment-Constrained DP

DP along a prescribed alignment path instead of the full `(Li+1, Lo+1)` grid.
Useful for training on known alignments.

```python
from machineboss.jax.dp_aligned import (
    aligned_log_forward, aligned_log_viterbi,
    neural_aligned_log_forward, neural_aligned_log_viterbi,
    MAT, INS, DEL
)

# Alignment: MAT=match (consume both), INS=insert (input only), DEL=delete (output only)
alignment = jnp.array([MAT, INS, MAT, DEL, MAT])

ll = aligned_log_forward(jm, input_tokens, output_tokens, alignment)
```

---

### Fused Plan7 + Transducer DP

Avoids the cost of explicitly composing a Plan7 HMM with a transducer
by interleaving their DP recurrences (GeneWise-style).

#### Generic Fusion (`fused.py`)

```python
from machineboss.jax.fused import FusedMachine, fused_log_forward

fm = FusedMachine.build(plan7_em, transducer_em)
ll = fused_log_forward(fm, output_seq)
```

#### Plan7-Optimized Fusion (`fused_plan7.py`)

Exploits Plan7's linear chain topology for O(S_td) per node instead of O(S_p7 * S_td).
Builds directly from an `HmmerModel` without constructing an intermediate `Machine`.

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

## Typical Workflows

### Log-likelihood of a sequence pair

```python
from machineboss.machine import Machine
from machineboss.eval import EvaluatedMachine
from machineboss.jax import JAXMachine, log_forward
import jax.numpy as jnp

m = Machine.from_file("preset/jukescantor.json")
em = EvaluatedMachine.from_machine(m, {"t": 0.5})
jm = JAXMachine.from_evaluated(em)

input_seq = jnp.array(em.tokenize_input(list("ACGT")))
output_seq = jnp.array(em.tokenize_output(list("ACGA")))
ll = log_forward(jm, input_seq, output_seq)
```

### Gradient-based parameter fitting

```python
import jax
from machineboss.jax.jax_weight import ParameterizedMachine
from machineboss.jax.dp_neural import neural_log_forward_tok

m = Machine.from_file("preset/jukescantor.json")
pm = ParameterizedMachine.from_machine(m)

def neg_ll(t):
    params = {"t": t}
    return -neural_log_forward_tok(pm, input_tokens, output_tokens, params)

grad_fn = jax.grad(neg_ll)
t = jnp.float32(1.0)
for _ in range(100):
    t = t - 0.01 * grad_fn(t)
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
