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
  -h [ --help ]              display this help message
  -v [ --verbose ] arg (=2)  verbosity level
  -d [ --debug ] arg         log specified function
  -b [ --monochrome ]        log in black & white

Transducer construction:
  -l [ --load ] arg          load machine from file
  -p [ --preset ] arg        select preset (compdna, comprna, dnapsw, null, 
                             protpsw)
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
  -P [ --params ] arg        load parameters
  -F [ --functions ] arg     load functions & constants
  -C [ --constraints ] arg   load constraints
  -D [ --data ] arg          load sequence-pairs
  -T [ --train ]             Baum-Welch parameter fit
  -A [ --align ]             Viterbi sequence alignment

</code></pre>
