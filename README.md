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
  -L [ --log ] arg           log specified function
  -m [ --monochrome ]        log in monochrome

Transducer construction:
  -d [ --load ] arg          load machine from file
  -t [ --preset ] arg        select preset (compdna, comprna)
  -g [ --generate ] arg      sequence generator '&lt;'
  -a [ --accept ] arg        sequence acceptor '&gt;'
  -n [ --null ]              null transducer
  -w [ --weight ] arg        weighted null transition '#'

Prefix operations:
  -e [ --reverse ]           reverse
  -r [ --revcomp ]           reverse-complement '~'
  -f [ --flip ]              flip input/output

Postfix operations:
  -z [ --zero-or-one ]       union with null '?'
  -k [ --kleene-star ]       Kleene star '*'
  -K [ --kleene-plus ]       Kleene plus '+'

Infix operations:
  -p [ --compose ]           compose '=&gt;'
  -c [ --concat ]            concatenate '.'
  -i [ --and ]               intersect '&&'
  -u [ --or ]                take union '||'
  -l [ --kleene-loop ]       Kleene with loop: x '?+' y = x(y.x)*

Miscellaneous:
  -B [ --begin ]             left bracket '('
  -E [ --end ]               right bracket ')'

Transducer application:
  -S [ --save ] arg          save machine
  -P [ --params ] arg        load parameter file
  -C [ --constraints ] arg   load constraints file
  -D [ --data ] arg          load sequence-pairs file
  -F [ --fit ]               Baum-Welch parameter fit
  -A [ --align ]             Viterbi sequence alignment

</code></pre>
