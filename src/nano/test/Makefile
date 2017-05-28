
NANO = ../../../bin/nanomachine

r9.2_180mv_250bps_6mer/template_median68pA.model r9.4_180mv_450bps_6mer/template_median68pA.model:
	curl https://raw.githubusercontent.com/nanoporetech/kmer_models/master/$@ >$@

%.json: %.model
	../node/kmer2model.js $< >$@

ont-r92.json: r9.2_180mv_250bps_6mer/template_median68pA.json
	cp $< $@

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


LOMAN = $(HOME)/loman-r9/pass
$(READ).$(PARAMS).cmp: $(READ).$(PARAMS).fa $(LOMAN)/$(READ).fast5
	../node/compare.js -r $(LOMAN)/$(READ).fast5 -q $(READ).$(PARAMS).fa

# single geometric length distribution, using fast5 event detection
$(READ).$(K)mer.json: $(LOMAN)/$(READ).fast5
	../node/events2model.js -f $< -k $K >$@

$(READ).$(K)mer.fa: $(READ).$(K)mer.json
	$(NANO) -d $(LOMAN)/$(READ).fast5 -m $(READ).$(K)mer.json -v6 -b -x -E >$@

# negative binomial up to $(R), no event detection
$(READ).$(K)mer.$(R)negbin.json: $(LOMAN)/$(READ).fast5
	../node/events2model.js -f $< -k $K -c $R >$@

$(READ).$(K)mer.$(R)negbin.fa: $(READ).$(K)mer.$(R)negbin.json
	$(NANO) -d $(LOMAN)/$(READ).fast5 -m $(READ).$(K)mer.$(R)negbin.json -v6 -b -x -V1 >$@

# negative binomial up to $(R) aggregated across all kmers, no event detection
$(READ).$(K)mer.$(R)negbin-agg.json: $(LOMAN)/$(READ).fast5
	../node/events2model.js -f $< -k $K -c $R -a >$@

$(READ).$(K)mer.$(R)negbin-agg.fa: $(READ).$(K)mer.$(R)negbin-agg.json
	$(NANO) -d $(LOMAN)/$(READ).fast5 -m $(READ).$(K)mer.$(R)negbin-agg.json -v6 -b -x -V1 >$@
