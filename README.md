# Boss Machine
Bioinformatics Open Source Sequence machine


## Command-line usage

<pre><code>

General options:
  -h [ --help ]              display this help message
  -s [ --save ] arg          save machine
  -F [ --fit ]               Baum-Welch parameter fit
  -P [ --params ] arg        parameter file
  -C [ --constraints ] arg   constraints file
  -D [ --data ] arg          sequence-pair file
  -A [ --align ]             Viterbi sequence alignment
  -v [ --verbose ] arg (=2)  verbosity level
  --log arg                  log specified function
  --nocolor                  log in monochrome

Transducer manipulation:
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

</code></pre>
