# Boss Machine
Bioinformatics Open Source Sequence machine


## Command-line usage

<pre><code>
Allowed options:
  -h [ --help ]             display this help message
  -g [ --generate ] arg     create sequence generator
  -p [ --pipe ] arg         pipe (compose) machine(s)
  -c [ --concat ] arg       concatenate machine(s)
  -u [ --union ] arg        take union with machine
  -w [ --weight ] arg       parameterize union
  -k [ --kleene ]           make Kleene closure
  -l [ --loop ] arg         parameterize Kleene closure
  -a [ --accept ] arg       pipe to sequence acceptor
  -r [ --reverse ]          reverse direction
  -f [ --flip ]             flip input/output
  -n [ --null ]             pipe to null transducer
  -s [ --save ] arg         save machine
  -F [ --fit ]              Baum-Welch parameter fit
  -P [ --params ] arg       parameter file
  -C [ --constraints ] arg  constraints file
  -D [ --data ] arg         sequence-pair file
  -A [ --align ]            Viterbi sequence alignment
  -v [ --verbose ] arg (=2) verbosity level
  --log arg                 log specified function
  --nocolor                 log in monochrome

</code></pre>
