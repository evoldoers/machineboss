
NANO = ../../../bin/nanomachine

r9.4_180mv_450bps_6mer/template_median68pA.model:
	curl https://raw.githubusercontent.com/nanoporetech/kmer_models/master/r9.4_180mv_450bps_6mer/template_median68pA.model >$@

%.json: %.model
	../node/kmer2model.js $< >$@

ont-r94.json: r9.4_180mv_450bps_6mer/template_median68pA.json
	cp $< $@

$S-$M-default.raw: $S.fa $M.json
	../node/simulate.js -f $S.fa -m $M.json >$@

$S-$M-rate$R.raw: $S.fa $M.json
	../node/simulate.js -f $S.fa -m $M.json -r $R >$@

$S-$M-scale$C.raw: $S.fa $M.json
	../node/simulate.js -f $S.fa -m $M.json -c $C >$@

$S-$M-shift$H.raw: $S.fa $M.json
	../node/simulate.js -f $S.fa -m $M.json -s $H >$@

$S-$M-$P.decode: $S-$M-$P.raw $M.json
	$(NANO) -R $S-$M-$P.raw -M $M.json -v8 -C -T $S-$M-$P.trace >$@

