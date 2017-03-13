# Boss Machine
Bioinformatics Open Source Sequence machine


## Command-line usage

<pre><code>

General options:
  -h [ --help ]              display this help message
  -v [ --verbose ] arg (=2)  verbosity level
  --log arg                  log specified function
  --nocolor                  log in monochrome

Transducer manipulation:
  --preset arg               preset transducer (compdna,comprna)
  -g [ --generate ] arg      sequence generator
  -a [ --accept ] arg        sequence acceptor
  -p [ --pipe ] arg          pipe (compose) machine
  --compose                  compose last two machines
  -c [ --concat ] arg        concatenate machine
  --append                   concatenate last two machines
  --or arg                   take union with machine
  -u [ --union ]             union of last two machines
  -w [ --weight ] arg        weighted union of last two machines
  -k [ --kleene ]            Kleene closure
  -l [ --loop ] arg          weighted Kleene closure
  -R [ --reverse ]           reverse
  -r [ --revcomp ]           reverse complement
  -f [ --flip ]              flip input/output
  -n [ --null ]              null transducer

Transducer application:
  -S [ --save ] arg          save machine
  -P [ --params ] arg        load parameter file
  -C [ --constraints ] arg   load constraints file
  -D [ --data ] arg          load sequence-pairs file
  -F [ --fit ]               Baum-Welch parameter fit
  -A [ --align ]             Viterbi sequence alignment

</code></pre>
