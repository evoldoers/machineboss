
r9.4_180mv_450bps_6mer/template_median68pA.model:
	curl https://raw.githubusercontent.com/nanoporetech/kmer_models/master/r9.4_180mv_450bps_6mer/template_median68pA.model >$@

ont-r94.json: r9.4_180mv_450bps_6mer/template_median68pA.model
	../node/kmer2model.js $< >$@

