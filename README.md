# Boss Machine

## Bioinformatics Open Source Sequence Machine

In contrast to other C++ HMM libraries
which focus on inference tasks (likelihood calculation, parameter-fitting, and alignment),
this small, focused library emphasizes the **manipulation** of state machines.
For example: the combination of several modular state machines into a more complex one.

Manipulations can include concatenating, composing, intersecting, reverse complementing, Kleene-starring, and other such [operations](https://en.wikipedia.org/wiki/Finite-state_transducer).
Any state machine resulting from such operations can be run through the usual inference algorithms too (Forward, Backward, Viterbi, EM).

Boss Machine is fluent in several forms of communication:
it can read HMMER [profiles](http://hmmer.org/),
write GraphViz [dotfiles](https://www.graphviz.org/doc/info/lang.html), 
and run GeneWise-style [models](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC479130/).
Its native format is a JSON description of a [weighted finite-state transducer](https://en.wikipedia.org/wiki/Finite-state_transducer),
along with a few related data structures such as sequences.

Boss Machine has an associated command-line tool that makes most transducer operations available through its arguments,
defining a small expression language for weighted automata.

## Features

- Construct [weighted finite-state transducers](https://en.wikipedia.org/wiki/Finite-state_transducer) using operations like composition, intersection, union, and Kleene closure
- Fit to data using [Viterbi](https://en.wikipedia.org/wiki/Viterbi_algorithm) and [Baum-Welch](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm) algorithms
- Weight functions fit using EM; M-step uses generic optimizers and probabilistic constraints
- Simple but powerful JSON format for automata; JSON schemas and C++ validators included for all formats

## File formats

Boss Machine defines JSON schemas for the following:

- [transducer](https://github.com/ihh/bossmachine/blob/master/t/machine/bitnoise.json). This file describes the [binary symmetric channel](https://en.wikipedia.org/wiki/Binary_symmetric_channel) from coding theory
- [parameters](https://github.com/ihh/bossmachine/blob/master/t/io/params.json)
- [individual sequence](https://github.com/ihh/bossmachine/blob/master/t/io/seqAGC.json) for constructing generators and acceptors
- [list of sequence-pairs](https://github.com/ihh/bossmachine/blob/master/t/io/seqpairlist.json) for model-fitting and alignment
- [constraints](https://github.com/ihh/bossmachine/blob/master/t/io/constraints.json) for model fitting. This file specifies the constraints `a+b=1` and `x+y+z=1`
	- see also [this file](https://github.com/ihh/bossmachine/blob/master/t/io/pqcons.json) whose constraint `p+q=1` can be used to fit the binary symmetric channel, above

## Command-line usage

<pre><code>

General options:
  -h [ --help ]              display this help message
  -v [ --verbose ] arg (=2)  verbosity level
  -d [ --debug ] arg         log specified function
  -b [ --monochrome ]        log in black & white

Transducer construction:
  -l [ --load ] arg          load machine from file
  -p [ --preset ] arg        select preset (base2acgt, compdna, comprna, 
                             dnapsw, flankbase, null, prot2dna, protpsw, 
                             psw2dna, simple_introns, translate)
  -g [ --generate ] arg      sequence generator '&lt;'
  -a [ --accept ] arg        sequence acceptor '&gt;'
  -w [ --weight ] arg        weighted null transition '#'
  -H [ --hmmer ] arg         load machine from HMMER3 model file

Prefix operators:
  -e [ --reverse ]           reverse
  -r [ --revcomp ]           reverse-complement '~'
  -f [ --flip ]              flip input/output
  -n [ --eliminate ]         eliminate silent transitions

Postfix operators:
  -z [ --zero-or-one ]       union with null '?'
  -k [ --kleene-star ]       Kleene star '*'
  -K [ --kleene-plus ]       Kleene plus '+'

Infix operators:
  -m [ --compose ]           compose '=&gt;'
  -c [ --concat ]            concatenate '.'
  -i [ --and ]               intersect '&&'
  -u [ --or ]                union '||'
  -o [ --loop ]              loop: x '?+' y = x(y.x)*

Miscellaneous:
  -B [ --begin ]             left bracket '('
  -E [ --end ]               right bracket ')'

Transducer application:
  -S [ --save ] arg          save machine to file
  -G [ --graphviz ]          write machine in Graphviz DOT format
  -M [ --memoize ]           memoize repeated expressions for compactness
  -W [ --showparams ]        show unbound parameters in final machine
  -P [ --params ] arg        load parameters
  -F [ --functions ] arg     load functions & constants
  -C [ --constraints ] arg   load constraints
  -D [ --data ] arg          load sequence-pairs
  -T [ --train ]             Baum-Welch parameter fit
  -A [ --align ]             Viterbi sequence alignment

</code></pre>
