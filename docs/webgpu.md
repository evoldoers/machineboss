---
title: WebGPU API
nav_order: 8
permalink: /webgpu/
---

# MachineBoss WebGPU API

GPU-accelerated dynamic programming for weighted finite-state transducers,
with pure JavaScript CPU fallback.

{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Quick Start

```javascript
import { MachineBoss } from './machineboss-gpu.mjs';

// A machine is a JSON object describing a weighted finite-state transducer.
// This is the binary symmetric channel: it transduces input bits to output
// bits, preserving each bit with probability p and flipping with probability q.
const bitnoise = {
  "state": [{
    "id": "S",
    "trans": [
      { "in": "0", "out": "0", "to": "S", "weight": "p" },
      { "in": "0", "out": "1", "to": "S", "weight": "q" },
      { "in": "1", "out": "1", "to": "S", "weight": "p" },
      { "in": "1", "out": "0", "to": "S", "weight": "q" }
    ]
  }]
};

// Create instance, binding parameter values
const mb = await MachineBoss.create(bitnoise, { p: 0.9, q: 0.1 });
console.log('Backend:', mb.backend); // 'webgpu' or 'cpu'

// Tokenize sequences
const input  = mb.tokenize('001', 'input');   // Uint32Array
const output = mb.tokenize('101', 'output');

// Forward log-likelihood
const ll = await mb.forward(input, output);

// Viterbi best path
const { score, path } = await mb.viterbi(input, output);

// Forward-Backward posteriors
const { logLikelihood, posteriors } = await mb.posteriors(input, output);

// 1D: generator (input=null) or recognizer (output=null)
const ll1d = await mb.forward(null, output);

// Cleanup
mb.destroy();
```

## Machine JSON Format

The `machineJSON` parameter is a
[Machine Boss JSON transducer](/json-format/)---a
JSON object with a `"state"` array describing a weighted finite-state transducer.
The first state is the start state; the last is the end state (a single-state machine
is both start and end). Each state has an `"id"` and a `"trans"` array
of transitions.

Each transition can have:

| Field | Description |
|---|---|
| `"to"` | Destination state id (required) |
| `"in"` | Input symbol (omit for silent/output-only transitions) |
| `"out"` | Output symbol (omit for silent/input-only transitions) |
| `"weight"` | Transition weight: a number, a parameter name (string), or an expression object (default: 1) |

Weight expressions support arithmetic, logarithms, and named parameters:

| Expression | Meaning |
|---|---|
| `0.9` | Constant |
| `"p"` | Named parameter (bound via the `params` argument) |
| `{"*": ["p", "q"]}` | Product: p × q |
| `{"+": [1, "x"]}` | Sum: 1 + x |
| `{"-": [1, "p"]}` | Difference: 1 - p |
| `{"/": ["a", "b"]}` | Quotient: a / b |
| `{"log": "p"}` | Natural log: ln(p) |
| `{"exp": "x"}` | Exponential: e^x |
| `{"not": "p"}` | Complement: 1 - p |

### Example machines

**Binary symmetric channel** (single-state transducer):

```json
{"state": [
  {"id":"S", "trans":[
    {"in":"0", "out":"0", "to":"S", "weight":"p"},
    {"in":"0", "out":"1", "to":"S", "weight":"q"},
    {"in":"1", "out":"1", "to":"S", "weight":"p"},
    {"in":"1", "out":"0", "to":"S", "weight":"q"}
  ]}
]}
```

**Unit indel model** (two states with silent, insert, delete, and match transitions):

```json
{"state": [
  {"id":"S", "trans":[
    {"out":"x", "to":"S", "weight":"ins"},
    {"in":"x", "to":"S", "weight":{"*":["no_ins","del"]}},
    {"in":"x", "out":"x", "to":"S", "weight":{"*":["no_ins","no_del"]}},
    {"to":"E", "weight":"no_ins"}
  ]},
  {"id":"E"}
]}
```

The `boss` CLI can generate machines from many sources (regex, HMMER profiles,
presets, etc.) and save them as JSON. See the
[JSON Format Reference](/json-format/) and
[Expression Language](/expressions/) for full details.

## API Reference

### `MachineBoss.create(machineJSON, params?, options?)`

Factory method. Returns a `Promise<MachineBoss>`.

| Parameter | Type | Description |
|---|---|---|
| `machineJSON` | Object | Machine Boss JSON transducer |
| `params` | Object | Parameter name → value map |
| `options.backend` | string | `'auto'` (default), `'webgpu'`, or `'cpu'` |

### `mb.forward(inputTokens, outputTokens)`

Returns `Promise<number>` --- log P(input, output \| machine).

### `mb.viterbi(inputTokens, outputTokens)`

Returns `Promise<{score, path}>`. `path` is an array of
`{state, inputToken, outputToken}` objects.

### `mb.posteriors(inputTokens, outputTokens)`

Returns `Promise<{logLikelihood, posteriors}>`.
`posteriors` is a `Float32Array` of shape
`(Li+1)*(Lo+1)*S` (2D) or `(L+1)*S` (1D), row-major.

### `mb.tokenize(seq, direction)`

Converts a symbol sequence (string or array) to `Uint32Array` of 1-based token indices.
`direction` is `'input'` or `'output'`.

### `mb.destroy()`

Releases GPU resources.

## PSWM (Position-Specific Weight Matrix) API

In addition to tokenized sequences, the 1D algorithms support PSWM input:
a profile where each position has a probability distribution over the alphabet,
rather than a single token. This is useful for probabilistic inputs such as
CSV profiles, neural network outputs, or uncertainty-weighted observations.

### `mb.nAlpha(direction)`

Returns the number of emitting symbols (excluding epsilon) for the given direction.
This is the number of columns in a PSWM profile row.

### `MachineBoss.logProfile(probs)`

Static utility. Converts a flat array of probabilities to log-space (`Float64Array`).
Zero probabilities become `-Infinity`.

### `mb.forwardProfile(logProfile, direction)`

Forward log-likelihood with a PSWM. `logProfile` is a `Float64Array`
of shape `(L * nAlpha)` containing log-probabilities, where
`logProfile[p * nAlpha + k]` is log P(symbol k at position p).
`direction` is `'input'` or `'output'`.
Returns `Promise<number>`.

### `mb.viterbiProfile(logProfile, direction)`

Viterbi best path with a PSWM. Same profile format as `forwardProfile`.
Returns `Promise<{score, path}>`.

### `mb.posteriorsProfile(logProfile, direction)`

Forward-Backward posteriors with a PSWM. Returns
`Promise<{logLikelihood, posteriors}>` where
`posteriors` is a `Float32Array` of shape `(L+1)*S`.

### Example: PSWM usage

```javascript
const mb = await MachineBoss.create(machineJSON, params);
const nAlpha = mb.nAlpha('output');

// Build a PSWM from probabilities (3 positions, nAlpha symbols each)
const probs = [
  0.8, 0.1, 0.05, 0.05,  // position 0
  0.1, 0.7, 0.1, 0.1,    // position 1
  0.05, 0.05, 0.1, 0.8,  // position 2
];
const logProfile = MachineBoss.logProfile(probs);

const ll = await mb.forwardProfile(logProfile, 'output');
const { score, path } = await mb.viterbiProfile(logProfile, 'output');
const { logLikelihood, posteriors } = await mb.posteriorsProfile(logProfile, 'output');
```

## Architecture

### 1D: Transfer-Matrix Parallel Prefix Scan

For generators (input=null) or recognizers (output=null), the forward DP is
a prefix product of S×S transfer matrices under the log-semiring.
This gives O(log L) parallel depth on GPU via a Hillis-Steele scan.

1. Compute silent closure S* on CPU
2. Build per-position transfer matrix M[p] = S* · Emit[token_p] · S* on GPU
   (for PSWM: Emit[p] = Σ_k profile[p,k] · Trans[k])
3. Parallel prefix scan of M matrices on GPU
4. Extract forward/backward vectors from prefix/suffix products

### 2D: Anti-Diagonal Wavefront

For transducers, the DP has 2D dependencies. We dispatch one compute shader
per anti-diagonal d = i + o. Each thread computes one cell independently,
combining match (i-1,o-1), insert (i-1,o), and delete (i,o-1) predecessors.

### CPU Fallback

Pure JavaScript with `Float64Array` for better precision.
Same API, wrapped in Promises. Sequential O(L×S²) for 1D,
O(Li×Lo×S²) for 2D.

## Numeric Representation

| Backend | Type | Precision |
|---|---|---|
| GPU | f32 | ~7 decimal digits |
| CPU | Float64Array | ~15 decimal digits |

GPU vs CPU agreement tolerance: 1e-4.

## WGSL Code Generation

The `boss` CLI can generate machine-specific WGSL shaders:

```bash
bin/boss machine.json --codegen output_dir --wgsl
```

This produces:

- `fill-log-trans.wgsl` --- WGSL shader to fill the dense log_trans tensor from parameters
- `machine.mjs` --- ES module with machine constants and `buildLogTrans(params)` for CPU
- `machine-meta.json` --- Machine metadata (state count, alphabets, parameter names)

## Testing

```bash
# CPU fallback tests (Node.js)
node js/webgpu/test/test-cpu.mjs

# GPU-CPU agreement (Node.js, CPU-only mode)
node js/webgpu/test/test-gpu-cpu-agreement.mjs

# WebGPU tests (requires Chromium with WebGPU)
npx playwright test js/webgpu/test/test-webgpu.mjs
```

## Browser Compatibility

| Browser | WebGPU | CPU Fallback |
|---|---|---|
| Chrome 113+ | Yes | Yes |
| Edge 113+ | Yes | Yes |
| Firefox 120+ | Behind flag | Yes |
| Safari 17.4+ | Partial | Yes |
| Node.js 18+ | No* | Yes |

\* Node.js WebGPU support via `@aspect-build/webgpu` or Dawn bindings.
