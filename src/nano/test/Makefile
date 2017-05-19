
NANO = ../../../bin/nanomachine

r9.4_180mv_450bps_6mer/template_median68pA.model:
	curl https://raw.githubusercontent.com/nanoporetech/kmer_models/master/r9.4_180mv_450bps_6mer/template_median68pA.model >$@

%.json: %.model
	../node/kmer2model.js $< >$@

ont-r94.json: r9.4_180mv_450bps_6mer/template_median68pA.json
	cp $< $@

$S-$M.raw: $S.fa $M.json
	../node/simulate.js -f $S.fa -m $M.json >$@

$S-$M.decode: $S-$M.raw $M.json
	$(NANO) -R $S-$M.raw -M $M.json -v8 -C >$@

