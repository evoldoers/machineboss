---
title: Tutorial
nav_order: 2
permalink: /tutorial/
---

# Tutorial: The Occasionally Dishonest Casino

This tutorial walks through the classic "occasionally dishonest casino" example
from Durbin, Eddy, Krogh & Mitchison (1998), _Biological Sequence Analysis_.
We'll build progressively more complex machines and demonstrate key Machine Boss operations:
generators, transducers, composition, simulation, likelihood, alignment, and training.

## Table of Contents
{: .no_toc }

1. TOC
{:toc}

## The Setup

A casino has two dice: a **fair die** (each face has probability 1/6)
and a **loaded die** (face 6 has probability 1/2; the other five faces share the remaining 1/2 equally, each with probability 1/10).
The casino occasionally switches between dice, but the gambler can't see which die is being used.
This is a classic Hidden Markov Model (HMM): the die is the hidden state, and the rolls are the observations.

The Durbin et al. parameters are:
- Probability of switching from fair to loaded: **0.05**
- Probability of switching from loaded to fair: **0.10**
- Probability that the loaded die shows 6: **0.50**
- Probability of ending the game after each roll: **0.01**

## 1. The Non-Parametric Casino

Our first machine is a **generator**: it has output-only transitions (no input),
modeling the casino as a source that emits die rolls.
The machine has three states: **Fair**, **Loaded**, and **End**.

Here is `tutorial/casino.json`:

```json
{% raw %}{"state":
 [{"id": "Fair",
   "trans": [
     {"out": "1", "to": "Fair", "weight": 0.15675},
     {"out": "2", "to": "Fair", "weight": 0.15675},
     {"out": "3", "to": "Fair", "weight": 0.15675},
     {"out": "4", "to": "Fair", "weight": 0.15675},
     {"out": "5", "to": "Fair", "weight": 0.15675},
     {"out": "6", "to": "Fair", "weight": 0.15675},
     {"out": "1", "to": "Loaded", "weight": 0.00825},
     {"out": "2", "to": "Loaded", "weight": 0.00825},
     {"out": "3", "to": "Loaded", "weight": 0.00825},
     {"out": "4", "to": "Loaded", "weight": 0.00825},
     {"out": "5", "to": "Loaded", "weight": 0.00825},
     {"out": "6", "to": "Loaded", "weight": 0.00825},
     {"to": "End", "weight": 0.01}
   ]},
  {"id": "Loaded",
   "trans": [
     {"out": "1", "to": "Loaded", "weight": 0.0891},
     {"out": "2", "to": "Loaded", "weight": 0.0891},
     {"out": "3", "to": "Loaded", "weight": 0.0891},
     {"out": "4", "to": "Loaded", "weight": 0.0891},
     {"out": "5", "to": "Loaded", "weight": 0.0891},
     {"out": "6", "to": "Loaded", "weight": 0.4455},
     {"out": "1", "to": "Fair", "weight": 0.0099},
     {"out": "2", "to": "Fair", "weight": 0.0099},
     {"out": "3", "to": "Fair", "weight": 0.0099},
     {"out": "4", "to": "Fair", "weight": 0.0099},
     {"out": "5", "to": "Fair", "weight": 0.0099},
     {"out": "6", "to": "Fair", "weight": 0.0495},
     {"to": "End", "weight": 0.01}
   ]},
  {"id": "End", "trans": []}
 ]
}{% endraw %}
```

Each emitting state has 12 output transitions (6 faces x 2 destinations)
plus one silent transition to End.
The **first state** is always the start; the **last state** is always the end---this
is a convention of the Machine Boss [JSON format](/json-format/).

The weights encode the joint probability of emitting a face, choosing a destination, and continuing the game.
For the Fair state, the weight of each transition is:

- Stay fair and emit face _d_: `(1 - pEnd) * (1 - changeToLoaded) * 1/6 = 0.99 * 0.95 / 6 = 0.15675`
- Switch to loaded and emit face _d_: `(1 - pEnd) * changeToLoaded * 1/6 = 0.99 * 0.05 / 6 = 0.00825`
- End the game (silent): `pEnd = 0.01`

### Generate Random Rolls

To sample a random sequence of rolls from this machine:

```bash
boss tutorial/casino.json --random-encode --seed 42
```

Output:
```
5 2 5 4 3 1 3 3 1 5 1 5 6 1
```

Try different seeds to get different sequences. Seed 123 gives a longer run:

```bash
boss tutorial/casino.json --random-encode --seed 123
```

Output (73 rolls):
```
5 3 5 5 4 5 3 4 1 3 4 3 2 5 4 5 3 1 5 3 2 4 5 3 3 5 3 2
6 6 6 1 2 2 1 6 6 6 5 4 5 3 4 5 6 1 4 2 3 5 4 6 2 3 1 3
5 6 2 4 2 6 2 4 2 3 6 2 4 2 4 2 6
```

Notice the run of three 6's starting at position 29---that's
likely when the casino switched to the loaded die.

### Log-Likelihood

To compute the log-likelihood of an observed sequence of rolls
(summing over all possible hidden state paths):

```bash
boss tutorial/casino.json --output-chars 525431335151 --loglike
```

```
[["","525431335151",-26.6559]]
```

The log-likelihood is about -26.66 nats.

## 2. The Parametric Casino

Hard-coding all the weights makes the machine hard to read and impossible to train.
Let's redesign the casino with **named parameters** and **defs** (definitions
that derive intermediate quantities from the free parameters).

Here is `tutorial/casino-param.json`:

```json
{% raw %}{"state":
 [{"id": "Fair",
   "trans": [
     {"out": "1", "to": "Fair", "weight": {"*": ["fairStay", "fairDie"]}},
     ...
     {"out": "6", "to": "Loaded", "weight": {"*": ["fairSwitch", "fairDie"]}},
     {"to": "End", "weight": "pEnd"}
   ]},
  {"id": "Loaded",
   "trans": [
     {"out": "1", "to": "Loaded", "weight": {"*": ["loadedStay", "loadedPOther"]}},
     ...
     {"out": "6", "to": "Loaded", "weight": {"*": ["loadedStay", "loadedP6"]}},
     {"out": "1", "to": "Fair", "weight": {"*": ["loadedSwitch", "loadedPOther"]}},
     ...
     {"out": "6", "to": "Fair", "weight": {"*": ["loadedSwitch", "loadedP6"]}},
     {"to": "End", "weight": "pEnd"}
   ]},
  {"id": "End", "trans": []}
 ],
 "defs": {
   "pContinue": {"not": "pEnd"},
   "loadedPOther": {"/": [{"not": "loadedP6"}, 5]},
   "fairDie": {"/": [1, 6]},
   "fairStay": {"*": ["pContinue", {"not": "changeToLoadedDie"}]},
   "fairSwitch": {"*": ["pContinue", "changeToLoadedDie"]},
   "loadedStay": {"*": ["pContinue", {"not": "changeToFairDie"}]},
   "loadedSwitch": {"*": ["pContinue", "changeToFairDie"]}
 }
}{% endraw %}
```

The four free parameters are `pEnd`, `changeToLoadedDie`, `changeToFairDie`, and `loadedP6`.
The `defs` section defines derived quantities:
`{"not": "pEnd"}` means `1 - pEnd` (probability complement);
`{"*": [...]}` is multiplication; `{"/": [...]}` is division.
See the [Expression Language](/expressions/) reference for full details.

### Inspecting Parameters

To see which parameters the machine uses:

```bash
boss tutorial/casino-param.json --show-params
```

This prints the machine with a `"params"` field listing the four free parameters:
`changeToFairDie`, `changeToLoadedDie`, `loadedP6`, `pEnd`.

### Evaluating with Specific Parameters

Save the Durbin et al. values to a file:

```bash
echo '{"pEnd":0.01,"changeToLoadedDie":0.05,"changeToFairDie":0.10,"loadedP6":0.5}' > params.json
```

Then evaluate the parametric machine with those values:

```bash
boss tutorial/casino-param.json --evaluate --params params.json
```

This prints the machine with all symbolic weights replaced by numbers---identical
to `tutorial/casino.json`.

### Constraints

The file `tutorial/casino-cons.json` declares that all four parameters are probabilities (between 0 and 1):

```json
{"prob": ["pEnd", "changeToLoadedDie", "changeToFairDie", "loadedP6"]}
```

These constraints are used during parameter estimation to keep values in valid ranges.

## 3. Parameter Estimation

One of the most powerful features of Machine Boss is **Baum-Welch training**:
given observed data, estimate the model parameters that maximize the likelihood.

First, generate some data:

```bash
boss tutorial/casino.json --random-encode --seed 100
```

This produces 225 rolls. Save the output sequence and use it to train:

```bash
boss tutorial/casino-param.json \
  --output-chars 534112236154412611355563424514433331543462161661113511666662513243616666616352616464262241263441336226151416446153113614536554564534432535636644524615135346166663636665164426425433433246521424133422666416253643321256636416416 \
  --train --constraints tutorial/casino-cons.json
```

Output:
```json
{"changeToFairDie":0.528,"changeToLoadedDie":0.455,"loadedP6":0.376,"pEnd":0.004}
```

The estimates are noisy because we're fitting from a single sequence of 225 rolls.
With more data (or longer sequences), the estimates converge closer to the true
values (0.10, 0.05, 0.50, 0.01).

**Exercise:** Try generating several long sequences with different seeds and fitting parameters to each.
How do the estimates vary? What happens with very short sequences?

## 4. The Gambling Metalhead

Now let's build a **transducer**---a machine that reads input and produces output.
The "gambling metalhead" copies die rolls faithfully, but adds an exclamation mark
whenever three or more consecutive 6's are seen.
It's a deterministic transducer (all weights are 1).

The metalhead has three states tracking consecutive 6's, plus an End state:
- **S0** (no recent 6): on input 6, output "6" and go to S1; on other input, echo it and stay in S0
- **S1** (one recent 6): on input 6, output "6" and go to S2; on other input, echo it and return to S0
- **S2** (two or more recent 6's): on input 6, output "6!" and stay in S2; on other input, echo it and return to S0

Here is `tutorial/metalhead.json`:

```json
{% raw %}{"state":
 [{"id": "S0",
   "trans": [
     {"in": "1", "out": "1", "to": "S0"},
     {"in": "2", "out": "2", "to": "S0"},
     {"in": "3", "out": "3", "to": "S0"},
     {"in": "4", "out": "4", "to": "S0"},
     {"in": "5", "out": "5", "to": "S0"},
     {"in": "6", "out": "6", "to": "S1"},
     {"to": "End"}
   ]},
  {"id": "S1",
   "trans": [
     {"in": "1", "out": "1", "to": "S0"},
     {"in": "2", "out": "2", "to": "S0"},
     {"in": "3", "out": "3", "to": "S0"},
     {"in": "4", "out": "4", "to": "S0"},
     {"in": "5", "out": "5", "to": "S0"},
     {"in": "6", "out": "6", "to": "S2"},
     {"to": "End"}
   ]},
  {"id": "S2",
   "trans": [
     {"in": "1", "out": "1", "to": "S0"},
     {"in": "2", "out": "2", "to": "S0"},
     {"in": "3", "out": "3", "to": "S0"},
     {"in": "4", "out": "4", "to": "S0"},
     {"in": "5", "out": "5", "to": "S0"},
     {"in": "6", "out": "6!", "to": "S2"},
     {"to": "End"}
   ]},
  {"id": "End", "trans": []}
 ]
}{% endraw %}
```

### Testing the Metalhead

Feed it a sequence with three consecutive 6's:

```bash
boss tutorial/metalhead.json --input-chars 316625466613 --random-encode --seed 1
```

Output:
```
3 1 6 6 2 5 4 6 6 6! 1 3
```

The first two 6's pass through normally; the third 6 becomes "6!".

### Composing Casino and Metalhead

The real power of Machine Boss is **composition**: piping one machine's output into another's input.
When you list multiple machines on the command line, they are composed left to right.

```bash
boss tutorial/casino.json tutorial/metalhead.json --random-encode --seed 123
```

Output:
```
5 3 5 5 4 5 3 4 1 3 4 3 2 5 4 5 3 1 5 3 2 4 5 3 3 5 3 2
6 6 6! 1 2 2 1 6 6 6! 5 4 5 3 4 5 6 1 4 2 3 5 4 6 2 3 1
3 5 6 2 4 2 6 2 4 2 3 6 2 4 2 4 2 6
```

Compare with the raw casino output (seed 123, shown earlier): the rolls are
identical, but now the third consecutive 6 in each run is annotated with "!".
The composition **automatically** threads the casino's output through the metalhead.

## 5. The Incredibly Unreliable Reporter

Our next transducer is **probabilistic**: the "incredibly unreliable reporter"
observes die rolls and reports them, but sometimes lies.
The reporter is honest until three or more consecutive 6's trigger a crisis of conscience.
After that, the reporter enters a **lying state** where 6's may be censored (deleted from the output)
and the reporter may randomly snap out of it.

The reporter has four states plus End:
- **H0** (honest, no recent 6's): echoes input; on input 6, goes to H1
- **H1** (honest, one recent 6): echoes input; on input 6, goes to H2; on other input, returns to H0
- **H2** (honest, two recent 6's): echoes input; on input 6, stays in H2 (weight `1 - startLying`) or goes to L (weight `startLying`)
- **L** (lying): on input 6, censors it with probability `censorTheNews` or reports it with probability `1 - censorTheNews`; on other input, returns to H0 (weight `stopLying`) or stays in L (weight `1 - stopLying`)

Three free parameters: `startLying`, `stopLying`, `censorTheNews`.

See `tutorial/reporter.json` for the full machine definition and
`tutorial/reporter-cons.json` for its constraints.

### Testing the Reporter

Let's use aggressive parameters to see the effect clearly:

```bash
echo '{"startLying":0.9,"stopLying":0.1,"censorTheNews":0.9}' > reporter-params.json
```

```bash
boss tutorial/reporter.json \
  --input-chars 66612345666162666 \
  --random-encode --seed 3 \
  --params reporter-params.json
```

Output:
```
6 6 6 1 2 3 4 5 1 2 6
```

The original 17-character input lost 6 characters: the reporter
censored several 6's after entering the lying state.

### Viterbi Alignment

Given the original and reported sequences, we can find the most likely path through the reporter
using **Viterbi alignment**:

```bash
boss tutorial/reporter.json \
  --input-chars 66612345666162666 \
  --output-chars 66612345126 \
  --align --params reporter-params.json
```

The alignment shows which 6's were censored (input mapped to empty output):

```
6 -> 6
6 -> 6
6 -> 6
1 -> 1
2 -> 2
3 -> 3
4 -> 4
5 -> 5
6 -> -  (censored)
6 -> -  (censored)
6 -> -  (censored)
1 -> 1
6 -> -  (censored)
2 -> 2
6 -> -  (censored)
6 -> 6
6 -> -  (censored)
```

### Training the Reporter

Given paired input/output sequences, Baum-Welch training recovers the reporter's parameters:

```bash
boss tutorial/reporter.json \
  --input-chars 66612345666162666 \
  --output-chars 66612345126 \
  --train --constraints tutorial/reporter-cons.json
```

```json
{"censorTheNews":0.857,"startLying":1.000,"stopLying":0.000}
```

The estimates are extreme because we only have one short example.
With more data, training would recover values closer to the true parameters.

## 6. Composing Casino and Reporter

We can compose the casino generator with the reporter transducer:

```bash
boss tutorial/casino.json tutorial/reporter.json \
  --random-encode --seed 123 \
  --params reporter-params.json
```

This generates rolls from the casino and passes them through the reporter.
Comparing with the raw casino output (seed 123), some 6's have been removed
by the reporter's censorship.

## 7. Three-Way Composition

Machine Boss can compose any number of machines. Let's chain all three:
casino (generator) -> reporter (transducer) -> metalhead (transducer):

```bash
boss tutorial/casino.json tutorial/reporter.json tutorial/metalhead.json \
  --random-encode --seed 123 \
  --params reporter-params.json
```

Output:
```
5 3 5 5 4 5 3 4 1 3 4 3 2 5 4 5 3 1 5 3 2 4 5 3 3 5 3 2
6 6 6! 1 1 1 1 6 6 6! 3 3 4 2 3 4 6 1 4 2 3 5 3 5 2 3 1
3 5 6 2 3 2 6 2 4 2 3 6 2 4 2 4 2 6
```

The casino generates rolls, the reporter censors some 6's (and may alter the pattern
of consecutive 6's), and the metalhead annotates any remaining runs of three or more 6's
with "!". The composition is computed automatically by Machine Boss.

## 8. Summary of Commands

| Task | Command |
|------|---------|
| Generate random rolls | `boss tutorial/casino.json --random-encode --seed N` |
| Compute log-likelihood | `boss tutorial/casino.json --output-chars SEQ --loglike` |
| Show free parameters | `boss tutorial/casino-param.json --show-params` |
| Evaluate with parameters | `boss tutorial/casino-param.json --evaluate --params FILE` |
| Train parameters | `boss tutorial/casino-param.json --output-chars SEQ --train --constraints tutorial/casino-cons.json` |
| Compose machines | `boss MACHINE1 MACHINE2 ...` |
| Viterbi alignment | `boss MACHINE --input-chars IN --output-chars OUT --align --params FILE` |
| Run transducer | `boss MACHINE --input-chars SEQ --random-encode --params FILE` |

## Further Reading

- **[JSON Format Reference](/json-format/)** --- full specification of the machine JSON format
- **[Expression Language](/expressions/)** --- weight expression syntax (`not`, `*`, `/`, etc.)
- **[Program Reference](/machineboss/)** --- complete command-line reference for `boss`
- **[Composition Algorithm](/composition/)** --- how transducer composition works internally
