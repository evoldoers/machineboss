---
layout: page
title: C++ Library API
permalink: /library-api/
---

# C++ Library API

Machine Boss provides a first-class C++ library (`libboss.a`) with a single umbrella header and high-level convenience functions.

## Installation

```bash
make lib              # builds libboss.a
make install-lib      # installs to /usr/local (override with PREFIX=...)
```

## Usage

Include the umbrella header and link against the library:

```cpp
#include <machineboss.h>
```

```bash
clang++ -std=c++11 myprogram.cpp -I/usr/local/include/machineboss -L/usr/local/lib \
    -lboss -lgsl -lgslcblas -lboost_regex -lz
```

## Quick Start

### Load a machine and compute log-likelihood

```cpp
#include <machineboss.h>
#include <iostream>

using namespace MachineBoss;

int main() {
    // Load a machine from file
    Machine machine = loadMachine("my_transducer.json");

    // Set up parameters
    ParamAssign params;
    params.defs["p"] = WeightAlgebra::doubleConstant(0.9);
    params.defs["q"] = WeightAlgebra::doubleConstant(0.1);

    // Set up input/output sequences
    SeqPair seqPair;
    seqPair.input.seq = {"1", "0", "1"};
    seqPair.output.seq = {"0", "0", "1"};

    // Compute forward log-likelihood
    double ll = forwardLogLike(machine, params, seqPair);
    std::cout << "Log-likelihood: " << ll << std::endl;

    return 0;
}
```

### Viterbi alignment

```cpp
MachinePath path = viterbiAlign(machine, params, seqPair);
double vll = viterbiLogLike(machine, params, seqPair);
```

### Forward-Backward counts

```cpp
MachineCounts counts = forwardBackwardCounts(machine, params, seqPair);
```

### Baum-Welch training

```cpp
Constraints constraints;
// ... set up constraints ...

SeqPairList trainingData;
// ... load training data ...

Params fittedParams = baumWelchFit(machine, constraints, trainingData);
```

### Beam search decoding

```cpp
vguard<OutputSymbol> observed = {"0", "0", "1", "1"};
vguard<InputSymbol> decoded = beamDecode(machine, params, observed, 100);
```

### Prefix search decoding

```cpp
vguard<InputSymbol> decoded = prefixDecode(machine, params, observed);
```

## Convenience API Reference

All functions are in the `MachineBoss` namespace.

| Function | Description |
|----------|-------------|
| `loadMachine(filename)` | Load a Machine from a JSON file |
| `loadMachineJson(jsonString)` | Parse a Machine from a JSON string |
| `saveMachine(machine, filename)` | Write a Machine to a JSON file |
| `machineToJson(machine)` | Serialize a Machine to a JSON string |
| `mergeEquivalentStates(machine)` | Merge states with identical outgoing transitions |
| `forwardLogLike(machine, params, seqPair [, envelope])` | Forward algorithm log-likelihood |
| `viterbiLogLike(machine, params, seqPair)` | Viterbi log-likelihood |
| `viterbiAlign(machine, params, seqPair)` | Viterbi alignment (returns MachinePath) |
| `forwardBackwardCounts(machine, params, seqPair\|seqPairList)` | Forward-Backward expected counts |
| `baumWelchFit(machine, constraints, seqPairList [, seed, constants])` | Baum-Welch parameter fitting |
| `beamDecode(machine, params, output [, beamWidth])` | Beam search decoding |
| `prefixDecode(machine, params, output [, maxBacktrack])` | CTC prefix search decoding |

## Core Types

| Type | Header | Description |
|------|--------|-------------|
| `Machine` | `machine.h` | Weighted finite-state transducer |
| `MachineState` | `machine.h` | State with name and transitions |
| `MachineTransition` | `machine.h` | Transition with input/output labels and weight |
| `MachinePath` | `machine.h` | Sequence of transitions through a machine |
| `WeightExpr` | `weight.h` | Algebraic weight expression (JSON) |
| `Params` / `ParamAssign` | `params.h` | Named parameter definitions |
| `Constraints` | `constraints.h` | Parameter constraints for training |
| `SeqPair` / `SeqPairList` | `seqpair.h` | Input-output sequence pairs |
| `Envelope` | `seqpair.h` | Banding envelope for DP |
| `EvaluatedMachine` | `eval.h` | Machine with numerically evaluated weights |
| `MachineCounts` | `counts.h` | Expected transition counts from Forward-Backward |

## Algorithm Classes

For advanced usage, you can construct the DP matrices directly:

```cpp
EvaluatedMachine eval(machine, params);

// Forward matrix
ForwardMatrix fwd(eval, seqPair);
double ll = fwd.logLike();
MachinePath sampled = fwd.samplePath(machine, rng);

// Viterbi matrix
ViterbiMatrix vit(eval, seqPair);
MachinePath best = vit.path(machine);

// Backward matrix + counts
BackwardMatrix back(eval, seqPair);
MachineCounts counts(eval);
back.getCounts(fwd, counts);

// Beam search
BeamSearchMatrix beam(eval, outputSymbols, beamWidth);
vguard<InputSymbol> decoded = beam.bestSeq();

// Prefix tree search
PrefixTree tree(eval, outputSymbols, maxBacktrack);
vguard<InputSymbol> decoded = tree.doPrefixSearch();
```

## Code Generation

```cpp
// Generate C++ forward algorithm code
CPlusPlusCompiler compiler(true);  // true = 64-bit
compiler.compileForward(machine, Compiler::String, Compiler::String, "output_dir");

// Generate JavaScript
JavaScriptCompiler jsCompiler;
jsCompiler.compileForward(machine);

// Generate WebGPU shader
WGSLCompiler::compile(machine, "output_dir");
```

## Import Formats

```cpp
// HMMER profile HMMs
HmmerModel hmmer;
hmmer.read("profile.hmm");
Machine m = hmmer.machine();

// CSV profiles
CSVProfile csv;
csv.read("profile.csv");
Machine m = csv.machine();

// Preset machines
Machine m = MachinePresets::makePreset("dnapsw");

// Regular expressions
Machine m = RegexParser::parse("[ACGT]+");
```

## Internal Types

Types in `MachineBoss::detail` are implementation details and should not be relied upon:

- `detail::IndexMapperBase`, `detail::IdentityIndexMapper`, `detail::RollingOutputIndexMapper` (DP matrix indexing)
- `detail::LogSumExpLookupTable` (log-sum-exp acceleration)
- `detail::MachineSchema` (JSON schema validation)
- `detail::Abort`, `detail::Warn`, `detail::Fail` (error handling primitives)

These are available through `using` declarations in the `MachineBoss` namespace for backward compatibility, but new code should not reference them directly.
