---
title: Composition Algorithm
nav_order: 6
permalink: /composition/
---

# Machine Boss Composition Algorithm

Transducer composition is the core operation in Machine Boss. Given two transducers
A and B, the composition A &#x2218; B produces a new transducer that feeds A's outputs
into B's inputs, mapping A's input alphabet to B's output alphabet.

This document describes the implementation in
`Machine::compose` (`src/machine.cpp`).

{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

Given transducers A (with states 0..I-1) and B (with states 0..J-1), composition
builds a product machine whose states are pairs (i, j) where i is a state in A and
j is a state in B. A transition in the composed machine corresponds to A emitting
a symbol on its output tape that B simultaneously reads on its input tape, or to
one machine making a silent (epsilon) transition while the other waits.

The algorithm proceeds in three phases:

1. **Waiting machine conversion** --- convert B into a "waiting machine" form
2. **Accessibility DFS** --- find all reachable composite states from (0, 0)
3. **Transition construction** --- build the transition list for each reachable state

The result is then post-processed to remove inaccessible states, sort transitions,
and handle silent cycles.

## Waiting Machine Conversion

Before composition, the second transducer B is converted to a *waiting machine*
if it is not already one. In a waiting machine, every state either:

- **Waits** --- all outgoing transitions consume an input symbol (no silent transitions), or
- **Does not wait** --- all outgoing transitions are silent (no input-consuming transitions)

This separation simplifies the composition logic by ensuring that at each composite
state, it is unambiguous whether B is ready to accept a symbol from A or is making
internal silent transitions.

The conversion is performed by `Machine::waitingMachine()`, which splits states that
have both silent and input-consuming transitions into separate wait and non-wait states.

## Composite State Space

The composite state space is the Cartesian product of A's and B's states.
A composite state is identified by a single index computed as:

```
composite_index = i * J + j
```

where `i` is the state index in A, `j` is the state index in B,
and `J` is the total number of states in B.
The inverse mapping is:

```
i = composite_index / J
j = composite_index % J
```

The start state of the composed machine is (0, 0) and the end state is (I-1, J-1).
The maximum number of composite states is I × J, but typically only a fraction
are accessible.

## Accessibility DFS

Not all I × J composite states may be reachable from the start state.
The algorithm performs a depth-first search (DFS) from state (0, 0) to identify
the set of accessible states. Only these states are included in the final machine.

The DFS explores transitions using the same three cases described
[below](#transition-cases). If the end state (I-1, J-1) is not accessible,
a warning is emitted and a zero-probability machine is returned.

After the DFS, accessible states are sorted by index and assigned compact
sequential indices in the output machine.

## Transition Cases

At each composite state (i, j), transitions are computed based on whether
B's state j is a waiting state or not. There are three cases:

<table>
<tr><th>Case</th><th>Condition</th><th>Composed transition</th></tr>
<tr>
  <td><strong>A advances silently</strong></td>
  <td>B waits (or terminates) and A has a transition with no output</td>
  <td>Input: A's input; Output: none<br>
      Destination: (A's dest, j)<br>
      Weight: A's weight</td>
</tr>
<tr>
  <td><strong>Both advance</strong></td>
  <td>B waits (or terminates) and A has a transition with output symbol <em>s</em>,
      and B has a transition with input symbol <em>s</em></td>
  <td>Input: A's input; Output: B's output<br>
      Destination: (A's dest, B's dest)<br>
      Weight: A's weight &times; B's weight</td>
</tr>
<tr>
  <td><strong>B advances silently</strong></td>
  <td>B does not wait (has only silent transitions)</td>
  <td>Input: none; Output: B's output<br>
      Destination: (i, B's dest)<br>
      Weight: B's weight</td>
</tr>
</table>

The key insight is that when B is waiting, A "drives" the composition by either
making a silent transition (no output, so B stays put) or emitting a symbol that B
reads. When B is not waiting, B drives by making its own silent transitions while A
stays put.

### Weight multiplication

When both machines advance simultaneously (case 2), the transition weight is the
product of A's and B's transition weights, computed symbolically via
`WeightAlgebra::multiply`. This preserves algebraic expressions through composition.

### Degenerate transition collapsing

When the `collapseDegenerateTransitions` flag is set (the default for CLI composition),
transitions with the same (input, output, destination) triple are merged by summing
their weights. This produces a more compact machine.

## Silent Cycle Strategies

Composition can introduce silent cycles (loops with no input or output) in the
product machine. These cycles are problematic because the Forward algorithm may
not converge. Machine Boss provides three strategies for handling them:

| Strategy | CLI option | JSON key | Description |
|---|---|---|---|
| Break | `--compose-fast` | `"compose"` | Break silent cycles by removing back-edges. Fast but may lose probability mass. |
| Sum | `--compose` / `-m` | `"compose-sum"` | Sum out silent cycles by computing the geometric series. Preserves probability mass but slower. |
| Leave | `--compose-cyclic` | `"compose-unsort"` | Leave silent cycles intact. The caller is responsible for handling them. |

The default CLI option `--compose` (`-m`) uses the Sum strategy.
The `--compose-fast` option uses Break, which is faster for large machines.
Silent cycle processing is performed by `Machine::processCycles()`.

## Post-processing Pipeline

After the raw composition, the result is passed through a four-stage pipeline:

```
compMachine.ergodicMachine().advanceSort().processCycles(strategy).ergodicMachine()
```

| Stage | Method | Purpose |
|---|---|---|
| 1 | `ergodicMachine()` | Remove states not reachable from the start or from which the end is not reachable |
| 2 | `advanceSort()` | Topologically sort states so that silent transitions go forward where possible |
| 3 | `processCycles()` | Handle silent cycles according to the chosen strategy |
| 4 | `ergodicMachine()` | Remove any states made inaccessible by cycle processing |

## Intersection vs. Composition

Intersection (`Machine::intersect`) is a variant of composition for transducers
with empty output alphabets (recognizers). The key differences are:

- Both machines must have empty output alphabets (asserted at entry)
- The shared symbol is the *input* alphabet, not the output-to-input bridge
- When both machines advance, the input symbols must match (instead of A's output matching B's input)
- The composed output is always empty

The same three silent cycle strategies are available for intersection:

| Strategy | CLI option | JSON key |
|---|---|---|
| Break | `--intersect-fast` | `"intersect"` |
| Sum | `--intersect` / `-i` | `"intersect-sum"` |
| Leave | `--intersect-cyclic` | `"intersect-unsort"` |

## CLI Options

Composition and intersection are available as both infix CLI operators and JSON API operations:

### Command line

```bash
# Compose A with B (sum strategy, default):
boss A.json -m B.json

# Compose A with B (break strategy, faster):
boss A.json --compose-fast B.json

# Intersect two recognizers:
boss --recognize-chars 001 -i --recognize-chars 101

# Chain multiple compositions:
boss A.json -m B.json -m C.json
```

### JSON API

```json
{"compose":     [A, B]}    // Break strategy
{"compose-sum": [A, B]}    // Sum strategy
{"compose-unsort": [A, B]} // Leave strategy

{"intersect":     [A, B]}    // Break strategy
{"intersect-sum": [A, B]}    // Sum strategy
{"intersect-unsort": [A, B]} // Leave strategy
```

Where A and B are machine JSON objects (inline or nested operations).
