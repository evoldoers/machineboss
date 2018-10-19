![](img/machineboss.gif)

In contrast to other C++ HMM libraries
which focus on inference tasks (likelihood calculation, parameter-fitting, and alignment)
and often provide extensions such as [generalized HMMs](https://www.ncbi.nlm.nih.gov/pubmed/8877513),
Machine Boss emphasizes the **manipulation** of state machines defined to a tight specification.
(It does provide lots of useful inference and decoding algorithms too, and parser-generation.)

Manipulations can include concatenating, composing, intersecting, reverse complementing, Kleene-starring, and other such [operations](https://en.wikipedia.org/wiki/Finite-state_transducer).
Any state machine resulting from such operations can be run through the usual inference algorithms too (Forward, Backward, Viterbi, EM).

For example, a protein-to-DNA alignment algorithm like [GeneWise](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC479130/)
can be thought of as the combination of four state machines accounting for the following effects:

1. sequencing errors (e.g. substitutions)
2. splicing of introns
3. translation of DNA to protein
4. mutation of the protein (e.g. using the BLOSUM62 substitution matrix with affine gaps)

With MachineBoss, each of these sub-models can be separately designed, parameter-fitted, and (if necessary) refactored.

MachineBoss can read HMMER [profiles](http://hmmer.org/),
write GraphViz [dotfiles](https://www.graphviz.org/doc/info/lang.html), 
and run GeneWise-style [models](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC479130/).
Its native format is a deliberately restricted (simple and validatable)
JSON representation of a [weighted finite-state transducer](https://en.wikipedia.org/wiki/Finite-state_transducer).

## Features

- Construct [weighted finite-state transducers](https://en.wikipedia.org/wiki/Finite-state_transducer) using operations like composition, intersection, union, and Kleene closure
- Fit to data using [Viterbi](https://en.wikipedia.org/wiki/Viterbi_algorithm) and [Baum-Welch](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm) algorithms
- Weight functions fit using EM; M-step uses generic optimizers and probabilistic constraints
- Simple but powerful JSON format for automata; JSON schemas and C++ validators included for all formats

## JSON file formats

MachineBoss defines JSON schemas for several data structures.
Here are some examples:

- [transducer](https://github.com/evoldoers/machineboss/blob/master/t/machine/bitnoise.json). This file describes the [binary symmetric channel](https://en.wikipedia.org/wiki/Binary_symmetric_channel) from coding theory
    - [parameters](https://github.com/evoldoers/machineboss/blob/master/t/io/params.json)
    - [constraints](https://github.com/evoldoers/machineboss/blob/master/t/io/constraints.json) for model fitting. This file specifies the constraints `a+b=1` and `x+y+z=1`
        - see also [this file](https://github.com/evoldoers/machineboss/blob/master/t/io/pqcons.json) whose constraint `p+q=1` can be used to fit the binary symmetric channel, above
- [individual sequence](https://github.com/evoldoers/machineboss/blob/master/t/io/seqAGC.json) for constructing generators and acceptors
    - [list of sequence-pairs](https://github.com/evoldoers/machineboss/blob/master/t/io/seqpairlist.json) for model-fitting and alignment

## Command-line interface

MachineBoss has an associated command-line tool that makes most transducer operations available through its arguments,
defining a small expression language for weighted automata.

## Command-line usage

<pre><code>

General options:
  -h [ --help ]                display this help message
  -v [ --verbose ] arg (=2)    verbosity level
  -d [ --debug ] arg           log specified function
  -b [ --monochrome ]          log in black & white

Transducer construction:
  -l [ --load ] arg            load machine from file
  -p [ --preset ] arg          select preset (compdna, comprna, dnapsw, null, 
                               prot2dna, protpsw, psw2dna, translate)
  -g [ --generate-chars ] arg  generator for explicit character sequence '&lt;&lt;'
  --generate-one arg           generator for any one of specified characters
  --generate-wild arg          generator for Kleene closure over specified 
                               characters
  --generate-iid arg           as --generate-wild, but followed by 
                               --weight-output p
  --generate-uniform arg       as --generate-iid, but weights outputs by 
                               1/(output alphabet size)
  --generate-fasta arg         generator for FASTA-format sequence
  --generate-csv arg           create generator from CSV file
  --generate-json arg          sequence generator for JSON-format sequence
  -a [ --accept-chars ] arg    acceptor for explicit character sequence '&gt;&gt;'
  --accept-one arg             acceptor for any one of specified characters
  --accept-wild arg            acceptor for Kleene closure over specified 
                               characters
  --accept-iid arg             as --accept-wild, but followed by --weight-input
                               p
  --accept-uniform arg         as --accept-iid, but weights outputs by 1/(input
                               alphabet size)
  --accept-fasta arg           acceptor for FASTA-format sequence
  --accept-csv arg             create acceptor from CSV file
  --accept-json arg            sequence acceptor for JSON-format sequence
  --echo-one arg               identity for any one of specified characters
  --echo-wild arg              identity for Kleene closure over specified 
                               characters
  -w [ --weight ] arg          weighted null transition '#'
  -H [ --hmmer ] arg           create machine from HMMER3 model file

Postfix operators:
  -z [ --zero-or-one ]         union with null '?'
  -k [ --kleene-star ]         Kleene star '*'
  -K [ --kleene-plus ]         Kleene plus '+'
  --count-copies arg           Kleene star with dummy counting parameter
  --repeat arg                 repeat N times
  -e [ --reverse ]             reverse
  -r [ --revcomp ]             reverse-complement '~'
  -t [ --transpose ]           transpose: swap input/output
  --joint-norm                 normalize jointly (outgoing transition weights 
                               sum to 1)
  --cond-norm                  normalize conditionally (outgoing transition 
                               weights for each input symbol sum to 1)
  --sort                       topologically sort silent transition graph, if 
                               possible, but preserve silent cycles
  --sort-sum                   topologically sort, eliminating silent cycles
  --sort-break                 topologically sort, breaking silent cycles 
                               (faster than --sort-sum, but less precise)
  --decode-sort                topologically sort non-outputting transition 
                               graph
  -n [ --eliminate ]           eliminate all silent transitions
  --reciprocal                 element-wise reciprocal: invert all weight 
                               expressions
  --weight-input arg           apply weight parameter with given prefix to 
                               inputs
  --weight-output arg          apply weight parameter with given prefix to 
                               outputs

Infix operators:
  -m [ --compose-sum ]         compose, summing out silent cycles '=&gt;'
  --compose                    compose, breaking silent cycles (faster)
  --compose-unsort             compose, leaving silent cycles
  -c [ --concatenate ]         concatenate '.'
  -i [ --intersect-sum ]       intersect, summing out silent cycles '&&'
  --intersect                  intersect, breaking silent cycles (faster)
  --intersect-unsort           intersect, leaving silent cycles
  -u [ --union ]               union '||'
  -o [ --loop ]                loop: x '?+' y = x(y.x)*

Miscellaneous:
  -B [ --begin ]               left bracket '('
  -E [ --end ]                 right bracket ')'

Transducer application:
  -S [ --save ] arg            save machine to file
  -G [ --graphviz ]            write machine in Graphviz DOT format
  -M [ --memoize ]             memoize repeated expressions for compactness
  -W [ --showparams ]          show unbound parameters in final machine
  -P [ --params ] arg          load parameters (JSON)
  -U [ --use-defaults ]        use defaults (uniform distributions, unit rates)
                               for unspecified parameters; this option is 
                               implicit when training
  -F [ --functions ] arg       load functions & constants (JSON)
  -N [ --constraints ] arg     load normalization constraints (JSON)
  -D [ --data ] arg            load sequence-pairs (JSON)
  -I [ --input-fasta ] arg     load input sequence(s) from FASTA file
  --input-chars arg            specify input character sequence explicitly
  -O [ --output-fasta ] arg    load output sequence(s) from FASTA file
  --output-chars arg           specify output character sequence explicitly
  -T [ --train ]               Baum-Welch parameter fit
  -R [ --wiggle-room ] arg     wiggle room (allowed departure from training 
                               alignment)
  -A [ --align ]               Viterbi sequence alignment
  -L [ --loglike ]             Forward log-likelihood calculation
  -C [ --counts ]              Forward-Backward counts (derivatives of 
                               log-likelihood with respect to logs of 
                               parameters)
  -Z [ --beam-decode ]         find most likely input by beam search
  --beam-width arg             number of sequences to track during beam search 
                               (default 100)
  --prefix-decode              find most likely input by CTC prefix search
  --prefix-backtrack arg       specify max backtracking length for CTC prefix 
                               search
  --cool-decode                find most likely input by simulated annealing
  --mcmc-decode                find most likely input by MCMC search
  --decode-steps arg           simulated annealing steps per initial symbol
  -Y [ --beam-encode ]         find most likely output by beam search
  --prefix-encode              find most likely output by CTC prefix search
  --random-encode              sample random output by stochastic prefix search
  --seed arg                   random number seed

Parser-generator:
  --codegen arg                generate parser code, save to specified filename
                               prefix
  --cpp64                      generate C++ dynamic programming code (64-bit)
  --cpp32                      generate C++ dynamic programming code (32-bit)
  --js                         generate JavaScript dynamic programming code
  --showcells                  include debugging output in generated code
  --inseq arg                  input sequence type (String, Intvec, Profile)
  --outseq arg                 output sequence type (String, Intvec, Profile)

</code></pre>
