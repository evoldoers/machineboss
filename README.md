# Boss Machine
*Bioinformatics Open Source Sequence Machine*

- Construct [weighted finite-state transducers](https://en.wikipedia.org/wiki/Finite-state_transducer) using operations like composition, intersection, union, and Kleene closure.
- Fit to data using [Viterbi](https://en.wikipedia.org/wiki/Viterbi_algorithm) and [Baum-Welch](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm) algorithms.

Examples of JSON file formats:

- [transducer](https://github.com/ihh/bossmachine/blob/master/t/machine/bitnoise.json). This file describes the [binary symmetric channel](https://en.wikipedia.org/wiki/Binary_symmetric_channel)
- [parameters](https://github.com/ihh/bossmachine/blob/master/t/io/params.json)
- [individual sequence](https://github.com/ihh/bossmachine/blob/master/t/io/seqAGC.json) for constructing generators and acceptors
- [list of sequence-pairs](https://github.com/ihh/bossmachine/blob/master/t/io/seqpairlist.json) for model-fitting and alignment
- [constraints](https://github.com/ihh/bossmachine/blob/master/t/io/constraints.json) for model fitting. This file specifies the constraints `a+b=1` and `x+y+z=1`
	- see also [this file](https://github.com/ihh/bossmachine/blob/master/t/io/pqcons.json) whose constraint `p+q=1` can be used to fit the binary symmetric channel, above

## Command-line usage

<pre><code>

General options:
  -h [ --help ]               display this help message
  -v [ --verbose ] arg (=2)   verbosity level
  -L [ --log ] arg            log specified function
  -N [ --nocolor ]            log in monochrome

Transducer manipulation:
  -d [ --load ] arg           load machine from file
  -t [ --preset ] arg         preset machine (compdna, comprna)
  -g [ --generate ] arg       sequence generator
  -a [ --accept ] arg         sequence acceptor
  -p [ --pipe ] arg           pipe (compose) machine
  -M [ --compose ]            compose last two machines
  -i [ --and ] arg            intersect machine
  -I [ --intersect ]          intersect last two machines
  -c [ --concat ] arg         concatenate machine
  -N [ --append ]             concatenate last two machines
  -o [ --or ] arg             take union with machine
  -O [ --union ]              union of last two machines
  -W [ --union-weight ] arg   weighted union of last two machines
  -k [ --kleene ]             Kleene closure
  -K [ --kleene-weight ] arg  weighted Kleene closure
  -l [ --kleene-loop ] arg    Kleene closure via loop machine
  -e [ --reverse ]            reverse
  -r [ --revcomp ]            reverse complement
  -f [ --flip ]               flip input/output
  -n [ --null ]               null transducer
  -w [ --weight ] arg         single weighted transition
  -B [ --begin ]              left bracket '('
  -E [ --end ]                right bracket ')'

Transducer application:
  -S [ --save ] arg           save machine
  -P [ --params ] arg         load parameter file
  -C [ --constraints ] arg    load constraints file
  -D [ --data ] arg           load sequence-pairs file
  -F [ --fit ]                Baum-Welch parameter fit
  -A [ --align ]              Viterbi sequence alignment

</code></pre>
