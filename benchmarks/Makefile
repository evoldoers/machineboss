PYTHON     := ~/jax-env/bin/python
RSCRIPT    := Rscript
LATEXMK    := latexmk
REPO_ROOT  := $(shell cd .. && pwd)

export PYTHONPATH := $(REPO_ROOT)/python

HOSTNAME := $(shell python3 -c "import platform; print(platform.node())" 2>/dev/null || hostname)
RESULTS  := results/$(HOSTNAME).json

.PHONY: all results tables figures clean

all: report.pdf

$(RESULTS):
	$(PYTHON) run_benchmarks.py

results: $(RESULTS)

tables: $(RESULTS)
	$(PYTHON) run_benchmarks.py --tables-only

figures: $(RESULTS)
	$(RSCRIPT) plot_benchmarks.R

report.pdf: tables figures report.tex
	$(LATEXMK) -pdf report.tex

clean:
	rm -rf figures/ tables/ report.pdf
	$(LATEXMK) -C report.tex 2>/dev/null || true
