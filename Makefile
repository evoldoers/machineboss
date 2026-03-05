.SECONDARY:

MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MAKEFILE_DIR := $(dir $(MAKEFILE_PATH))

# Pseudotargets that control compilation
IS_32BIT = $(findstring 32bit,$(MAKECMDGOALS))
IS_DEBUG = $(findstring debug,$(MAKECMDGOALS))
IS_UNOPTIMIZED = $(findstring unoptimized,$(MAKECMDGOALS))

# C++ compiler: try clang++, fall back to g++
CPP = clang++
ifeq (, $(shell which $(CPP)))
CPP = g++
endif

GSL_LIB =
GSL_DEPS =
GSL_OBJ_FILES =
# Try to figure out where GSL is
# autoconf would be better but we just need a quick hack for now :)
# Thanks to Torsten Seemann for gsl-config and pkg-config formulae
GSL_PREFIX = $(shell gsl-config --prefix)
ifeq (,$(wildcard $(GSL_PREFIX)/include/gsl/gsl_sf.h))
GSL_PREFIX = /usr
ifeq (,$(wildcard $(GSL_PREFIX)/include/gsl/gsl_sf.h))
GSL_PREFIX = /usr/local
endif
endif

GSL_FLAGS = $(shell pkg-config --cflags gsl)
ifeq (, $(GSL_FLAGS))
GSL_FLAGS = -I$(GSL_PREFIX)/include
endif

GSL_LIBS = $(shell pkg-config --libs gsl)
ifeq (, $(GSL_LIBS))
GSL_LIBS = -L$(GSL_PREFIX)/lib -lgsl -lgslcblas
endif

BOOST_PROGRAM_OPTIONS = program_options
BOOST_OBJ_FILES =
BOOST_DEPS =
# Try to figure out where Boost is
# NB pkg-config support for Boost is lacking; see https://svn.boost.org/trac/boost/ticket/1094
BOOST_PREFIX = /usr
ifeq (,$(wildcard $(BOOST_PREFIX)/include/boost/regex.h))
BOOST_PREFIX = /usr/local
ifeq (,$(wildcard $(BOOST_PREFIX)/include/boost/regex.h))
BOOST_PREFIX = $(shell ls -lrt -d -1 /usr/local/homebrew/Cellar/boost/*)
ifeq (,$(wildcard $(BOOST_PREFIX)/include/boost/regex.h))
BOOST_PREFIX = $(shell ls -lrt -d -1 /opt/homebrew/Cellar/boost/*)
ifeq (,$(wildcard $(BOOST_PREFIX)/include/boost/regex.h))
BOOST_PREFIX =
endif
endif
endif
endif

BOOST_FLAGS =
BOOST_LIBS =
ifneq (,$(BOOST_PREFIX))
BOOST_FLAGS := -I$(BOOST_PREFIX)/include
BOOST_LIBS := -L$(BOOST_PREFIX)/lib -lboost_regex -lboost_$(BOOST_PROGRAM_OPTIONS)
endif

# install dir
PREFIX = /usr/local
INSTALL_BIN = $(PREFIX)/bin
INSTALL_LIB = $(PREFIX)/lib
INSTALL_INCLUDE = $(PREFIX)/include/machineboss

# other flags
ifneq (,$(IS_32BIT))
BUILD_FLAGS = -DIS32BIT
else
BUILD_FLAGS =
endif

ALL_FLAGS = $(GSL_FLAGS) $(BOOST_FLAGS) $(BUILD_FLAGS)
ALL_LIBS = $(GSL_LIBS) $(BOOST_LIBS) $(BUILD_LIBS)

ifneq (,$(IS_DEBUG))
CPP_FLAGS = -std=c++11 -g -DUSE_VECTOR_GUARDS -DDEBUG
else
ifneq (,$(IS_UNOPTIMIZED))
CPP_FLAGS = -std=c++11 -g
else
CPP_FLAGS = -std=c++11 -g -O3
endif
endif

LD_FLAGS = -lstdc++ -lm
CPP_FLAGS += $(ALL_FLAGS) -Isrc -Iinclude -Iext -Iext/nlohmann_json
LD_FLAGS += $(ALL_LIBS) -lz

# files
CPP_FILES = $(wildcard src/*.cpp)
OBJ_FILES = $(subst src/,obj/,$(subst .cpp,.o,$(CPP_FILES)))

# pwd
PWD = $(shell pwd)

# /bin/sh
SH = /bin/sh

# Targets

BOSS = boss
LIBTARGET = libboss.a

WRAP =
BOSSTARGET = bin/$(BOSS)
WRAPBOSS = $(BOSSTARGET)
TESTSUFFIX =
LIBTARGETS = $(LIBTARGET)

all: $(BOSS)

install: $(BOSS)
	cp bin/$(BOSS) $(INSTALL_BIN)/$(BOSS)

lib: $(LIBTARGET)

# Public API headers (umbrella + direct includes)
PUBLIC_HEADERS = include/machineboss.h \
    src/api.h src/machine.h src/weight.h src/params.h src/constraints.h \
    src/seqpair.h src/eval.h src/fastseq.h \
    src/forward.h src/backward.h src/viterbi.h \
    src/counts.h src/fitter.h src/beam.h src/ctc.h src/compiler.h \
    src/preset.h src/hmmer.h src/csv.h src/jphmm.h src/parsers.h

# Transitively-required headers (part of ABI)
ABI_HEADERS = src/dpmatrix.h src/dpmatrix.defs.h src/forward.defs.h \
    src/vguard.h src/stacktrace.h src/util.h src/jsonio.h \
    src/logsumexp.h src/logger.h src/schema.h \
    src/softplus.h src/getparams.h src/regexmacros.h

install-lib: $(LIBTARGET)
	@test -e $(INSTALL_INCLUDE) || mkdir -p $(INSTALL_INCLUDE)
	cp $(PUBLIC_HEADERS) $(INSTALL_INCLUDE)/
	cp $(ABI_HEADERS) $(INSTALL_INCLUDE)/
	cp ext/nlohmann_json/json.hpp $(INSTALL_INCLUDE)/
	cp $(LIBTARGET) $(INSTALL_LIB)

# Main build rules
bin/%: $(OBJ_FILES) obj/%.o target/%.cpp $(GSL_DEPS) $(BOOST_DEPS) $(BOOST_OBJ_FILES)
	@test -e $(dir $@) || mkdir -p $(dir $@)
	$(CPP) $(LD_FLAGS) -o $@ obj/$*.o $(OBJ_FILES) $(GSL_OBJ_FILES) $(BOOST_OBJ_FILES)

obj/%.o: src/%.cpp $(GSL_DEPS)
	@test -e $(dir $@) || mkdir -p $(dir $@)
	$(CPP) $(CPP_FLAGS) -c -o $@ $<

obj/%.o: target/%.cpp $(GSL_DEPS)
	@test -e $(dir $@) || mkdir -p $(dir $@)
	$(CPP) $(CPP_FLAGS) -c -o $@ $<

t/bin/%: $(OBJ_FILES) obj/%.o t/src/%.cpp $(GSL_DEPS) $(BOOST_OBJ_FILES)
	@test -e $(dir $@) || mkdir -p $(dir $@)
	$(CPP) $(LD_FLAGS) -o $@ obj/$*.o $(OBJ_FILES) $(GSL_OBJ_FILES) $(BOOST_OBJ_FILES)

t/codegen/%: $(OBJ_FILES) obj/%.o
	$(MAKE) `ls $(dir t/src/$*)computeForward*.cpp | python3 -c "import sys; [print(l.strip().replace('t/src','obj').replace('.cpp','.o')) for l in sys.stdin]"`
	@test -e $(dir $@) || mkdir -p $(dir $@)
	$(CPP) $(LD_FLAGS) -o $@ $^ `ls $(dir t/src/$*)computeForward*.cpp | python3 -c "import sys; [print(l.strip().replace('t/src','obj').replace('.cpp','.o')) for l in sys.stdin]"`

obj/%.o: t/src/%.cpp
	@test -e $(dir $@) || mkdir -p $(dir $@)
	$(CPP) $(CPP_FLAGS) -c -o $@ $<

# Top-level target
$(BOSS): bin/$(BOSS)

# Library target
$(LIBTARGET): $(OBJ_FILES)
	ar rc $@ $(OBJ_FILES)

# test: build boss using library
boss-with-lib: $(LIBTARGET) obj/boss.o
	$(CPP) $(LD_FLAGS) -L. -lboss -o $@ obj/boss.o

# Clean
clean:
	rm -rf bin/$(BOSS) t/bin/* obj/*

# Fake pseudotargets
debug unoptimized 32bit:

# Schemas, presets, grammars, and any other autogenerated source files
# The relevant pseudotargets are generate-schemas, generate-presets, and generate-grammars (biomake required)
# Get biomake here: https://github.com/evoldoers/biomake
generate-grammars: $(patsubst %.abnf,%.h,$(wildcard src/grammars/*.abnf))

src/grammars/%.h: src/grammars/%.abnf
	python3 -c "import sys; [print(chr(34)+line.rstrip().replace(chr(92),chr(92)+chr(92))+'\\\\n'+chr(34)) for line in open(sys.argv[1])]" $< >$@

generate-$(CATEGORY)s: $(patsubst $(CATEGORY)/%.json,src/$(CATEGORY)/%.h,$(wildcard $(CATEGORY)/*.json))
	touch src/$(CATEGORY).cpp

src/preset/$(FILE).h: preset/$(FILE).json
	xxd -i $< >$@

preset/iupacdna.json: js/iupacdna.js
	node $< >$@

preset/iupacaa.json: js/iupacaa.js
	node $< >$@

preset/protpsw.json constraints/protpsw.json: js/makepsw.js
	js/makepsw.js -w -a ACDEFGHIKLMNPQRSTVWY -n protpsw >$@

preset/dnapsw.json constraints/dnapsw.json: js/makepsw.js
	js/makepsw.js -w -a ACGT -n dnapsw >$@

preset/dnapsw_mix2.json: js/makepsw.js
	js/makepsw.js -a ACGT -n dnapsw_mix2 -m 2 >$@

preset/translate.json: js/translate.js
	node $< -C constraints/translate.json -P params/translate.json >$@

preset/translate-spliced.json: js/translate.js
	node $< -e base -e intron >$@

preset/%.json: js/%.js
	node $< >$@

preset/prot2dna.json: preset/flankbase.json preset/pint.json preset/translate-spliced.json preset/simple_introns.json preset/base2acgt.json
	$(WRAPBOSS) -v6 preset/flankbase.json '.' '(' preset/pint.json '=>' preset/translate-spliced.json '=>' preset/simple_introns.json ')' '.' preset/flankbase.json '=>' preset/base2acgt.json >$@

preset/psw2dna.json: preset/flankbase.json preset/pswint.json preset/translate-spliced.json preset/simple_introns.json preset/base2acgt.json
	$(WRAPBOSS) -v6 preset/flankbase.json '.' '(' preset/pswint.json '=>' preset/translate-spliced.json '=>' preset/simple_introns.json ')' '.' preset/flankbase.json '=>' preset/base2acgt.json >$@

preset/dnapswnbr.json: js/dna2.js
	$< >$@

# valijson doesn't like the URLs, but other schema validators demand them, so strip them out for xxd
src/schema/$(FILE).h: schema/$(FILE).json.nourl
	xxd -i $< | sed 's/.nourl//' >$@

schema/$(FILE).json.nourl: schema/$(FILE).json
	grep -v '"id": "http' $< >$@

# peglib grammars

# Transducer composition tests
COMPOSE_TESTS = test-echo test-echo2 test-echo2-expr test-echo-stutter test-stutter2 test-noise2 test-unitindel2 test-machine-params
test-echo:
	@$(TEST) $(WRAPBOSS) t/machine/bitecho.json t/expect/bitecho.json

test-echo2:
	@$(TEST) $(WRAPBOSS) t/machine/bitecho.json t/machine/bitecho.json t/expect/bitecho-bitecho.json

test-echo2-expr:
	@$(TEST) $(WRAPBOSS) t/machine/compose-bitecho-bitecho.json t/expect/bitecho-bitecho.json

test-echo-stutter:
	@$(TEST) $(WRAPBOSS) t/machine/bitecho.json t/machine/bitstutter.json t/expect/bitecho-bitstutter.json

test-stutter2:
	@$(TEST) $(WRAPBOSS) t/machine/bitstutter.json t/machine/bitstutter.json t/expect/bitstutter-bitstutter.json

test-noise2:
	@$(TEST) $(WRAPBOSS) t/machine/bitnoise.json t/machine/bitnoise.json --show-params t/expect/bitnoise-bitnoise.json

test-unitindel2:
	@$(TEST) $(WRAPBOSS) t/machine/unitindel.json t/machine/unitindel.json --show-params t/expect/unitindel-unitindel.json

test-machine-params:
	@$(TEST) $(WRAPBOSS) t/machine/params.json -idem

# Transducer construction tests
CONSTRUCT_TESTS = test-generator test-recognizer test-wild-generator test-wild-recognizer test-union test-intersection test-brackets test-kleene test-loop test-noisy-loop test-concat test-eliminate test-merge test-reverse test-revcomp test-transpose test-weight test-shorthand test-hmmer test-hmmer-plan7 test-hmmer-multihit test-jphmm test-csv test-csv-tiny test-csv-tiny-fail test-csv-tiny-empty test-nanopore test-nanopore-prefix test-nanopore-decode
test-generator:
	@$(TEST) $(WRAPBOSS) --generate-json t/io/seq101.json t/expect/generator101.json

test-recognizer:
	@$(TEST) $(WRAPBOSS) --recognize-json t/io/seq001.json t/expect/recognizer001.json

test-wild-generator:
	@$(TEST) $(WRAPBOSS) --generate-wild ACGT t/expect/ACGT_generator.json
	@$(TEST) $(WRAPBOSS) --generate-wild-dna t/expect/ACGT_generator.json

test-wild-recognizer:
	@$(TEST) $(WRAPBOSS) --recognize-wild ACGT t/expect/ACGT_recognizer.json
	@$(TEST) $(WRAPBOSS) --recognize-wild-dna t/expect/ACGT_recognizer.json

test-union:
	@$(TEST) $(WRAPBOSS) --generate-json t/io/seq001.json -u t/expect/generator101.json t/expect/generate-101-or-001.json

test-intersection:
	@$(TEST) $(WRAPBOSS) t/machine/bitnoise.json -m --recognize-json t/io/seq001.json -i --recognize-json t/io/seq101.json t/expect/noise-001-and-101.json

test-brackets:
	@$(TEST) $(WRAPBOSS) --begin t/machine/bitnoise.json --recognize-json t/io/seq001.json --end -i --recognize-json t/io/seq101.json t/expect/noise-001-and-101.json

test-kleene:
	@$(TEST) $(WRAPBOSS) --generate-json t/io/seq001.json -K t/expect/generate-multiple-001.json

test-loop:
	@$(TEST) $(WRAPBOSS) --recognize-json t/io/seq101.json -o --recognize-json t/io/seq001.json t/expect/101-loop-001.json

test-noisy-loop:
	@$(TEST) $(WRAPBOSS) t/machine/bitnoise.json --begin --recognize-json t/io/seq101.json -o --recognize-json t/io/seq001.json --end t/expect/noisy-101-loop-001.json

test-concat:
	@$(TEST) $(WRAPBOSS) --generate-json t/io/seq001.json -c t/expect/generator101.json t/expect/concat-001-101.json

test-eliminate:
	@$(TEST) $(WRAPBOSS) t/machine/silent.json -n t/expect/silent-elim.json
	@$(TEST) $(WRAPBOSS) t/machine/silent2.json -n t/expect/silent2-elim.json
	@$(TEST) $(WRAPBOSS) t/machine/silent3.json -n t/expect/silent3-elim.json
	@$(TEST) $(WRAPBOSS) t/machine/single-silent-incoming.json --eliminate-states t/expect/single-silent-incoming.json
	@$(TEST) $(WRAPBOSS) t/machine/single-silent-outgoing.json --eliminate-states t/expect/single-silent-outgoing.json

test-merge:
	@$(TEST) $(WRAPBOSS) t/machine/merge-parallel.json --merge-states t/expect/merge-parallel.json
	@$(TEST) $(WRAPBOSS) t/machine/merge-bubble.json --merge-states t/expect/merge-bubble.json
	@$(TEST) $(WRAPBOSS) t/machine/merge-noop.json --merge-states t/expect/merge-noop.json
	@$(TEST) $(WRAPBOSS) t/machine/merge-chain.json --merge-states t/expect/merge-chain.json

test-reverse:
	@$(TEST) $(WRAPBOSS) --generate-json t/io/seq001.json -e t/expect/generator001-reversed.json

test-revcomp:
	@$(TEST) $(WRAPBOSS) --generate-json t/io/seqAGC.json -r t/expect/generatorAGC-revcomp.json

test-transpose:
	@$(TEST) $(WRAPBOSS) --generate-json t/io/seq001.json -t t/expect/recognizer001.json

test-weight:
	@$(TEST) $(WRAPBOSS) -w '$$p' t/expect/null-p.json
	@$(TEST) $(WRAPBOSS) -w 2 t/expect/null-2.json
	@$(TEST) $(WRAPBOSS) -w .5 t/expect/null-0.5.json
	@$(TEST) $(WRAPBOSS) -w '$$p*$$q' t/expect/null-pq.json
	@$(TEST) $(WRAPBOSS) -w '1*2' t/expect/null-2.json
	@$(TEST) $(WRAPBOSS) -w '1/2' t/expect/null-1div2.json
	@$(TEST) $(WRAPBOSS) --recognize-wild ACGT --weight-input '$$p%' --reciprocal t/expect/null-weight-recip.json
	@$(TEST) $(WRAPBOSS) --recognize-wild ACGT --weight-input '1/$$p%' t/expect/null-weight-recip.json

test-shorthand:
	@$(TEST) $(WRAPBOSS) '(' t/machine/bitnoise.json '>>' 101 ')' '&&' '>>' 001 '.' '>>' AGC '#' '$$x' t/expect/shorthand.json

test-hmmer:
	@$(TEST) python3 t/roundfloats.py 3 $(WRAPBOSS) --hmmer-global t/hmmer/fn3.hmm t/expect/fn3.json

test-hmmer-plan7:
	@$(TEST) python3 t/roundfloats.py 3 $(WRAPBOSS) --hmmer-plan7 t/hmmer/fn3.hmm t/expect/fn3-plan7.json

test-hmmer-multihit:
	@$(TEST) python3 t/roundfloats.py 3 $(WRAPBOSS) --hmmer-multihit t/hmmer/fn3.hmm t/expect/fn3-multihit.json

test-jphmm:
	@$(TEST) $(WRAPBOSS) --jphmm t/seq/jphmmtest.fa t/expect/jphmmtest.json

test-csv:
	@$(TEST) $(WRAPBOSS) --generate-csv t/csv/test.csv t/expect/csvtest.json
	@$(TEST) $(WRAPBOSS) --generate-csv t/csv/test.csv --cond-norm t/expect/normcsvtest.json
	@$(TEST) $(WRAPBOSS) --recognize-csv t/csv/test.csv --transpose t/expect/csvtest.json
	@$(TEST) $(WRAPBOSS) --recognize-csv t/csv/test.csv --transpose --joint-norm t/expect/normcsvtest.json

test-csv-tiny:
	@$(TEST) js/stripnames.js $(WRAPBOSS) -L --generate-json t/io/tiny_uc.json --recognize-csv t/csv/tiny_uc.csv t/expect/tiny_uc.json

test-csv-tiny-fail:
	@$(TEST) js/stripnames.js $(WRAPBOSS) -L --generate-json t/io/tiny_lc.json --recognize-csv t/csv/tiny_uc.csv t/expect/tiny_uc_fail.json

test-csv-tiny-empty:
	@$(TEST) js/stripnames.js $(WRAPBOSS) -L --generate-json t/io/empty.json --recognize-csv t/csv/tiny_uc.csv t/expect/tiny_empty.json

test-nanopore:
	@$(TEST) js/stripnames.js $(WRAPBOSS) -L --generate-json t/io/nanopore_test_seq.json --recognize-csv t/csv/nanopore_test.csv t/expect/nanopore_test.json

test-nanopore-prefix:
	@$(TEST) js/stripnames.js $(WRAPBOSS) -L --generate-json t/io/nanopore_test_seq.json --concat t/machine/acgt_wild.json --recognize-csv t/csv/nanopore_test.csv t/expect/nanopore_test_prefix.json

test-nanopore-decode:
	@$(TEST) $(WRAPBOSS) --recognize-csv t/csv/nanopore_test.csv --beam-decode t/expect/nanopore_beam_decode.json

# Invalid transducer construction tests
INVALID_CONSTRUCT_TESTS = test-unmatched-begin test-unmatched-end test-empty-brackets test-impossible-intersect test-missing-machine
test-unmatched-begin:
	@$(TEST) $(WRAPBOSS) --begin -fail

test-unmatched-end:
	@$(TEST) $(WRAPBOSS) --end -fail

test-empty-brackets:
	@$(TEST) $(WRAPBOSS) --begin --end -fail

test-missing-machine:
	@$(TEST) $(WRAPBOSS) t/machine/bitnoise.json -m -m t/machine/bitnoise.json t/machine/bitnoise.json -fail

test-impossible-intersect:
	@$(TEST) $(WRAPBOSS) t/machine/bitnoise.json --begin --recognize-json t/io/seq001.json -i --recognize-json t/io/seq101.json --end t/expect/zero.json

# Schema validation tests
VALID_SCHEMA_TESTS = test-echo-valid test-unitindel2-valid
test-echo-valid:
	@$(TEST) $(WRAPBOSS) t/expect/bitecho.json -idem

test-unitindel2-valid:
	@$(TEST) $(WRAPBOSS) --show-params t/expect/unitindel-unitindel.json -idem

# Schema validation failure tests
INVALID_SCHEMA_TESTS = test-not-json test-no-state test-bad-state test-bad-trans test-bad-weight test-cyclic
test-not-json:
	@$(TEST) $(WRAPBOSS) t/invalid/not_json.txt -fail

test-no-state:
	@$(TEST) $(WRAPBOSS) t/invalid/no_state.json -fail

test-bad-state:
	@$(TEST) $(WRAPBOSS) t/invalid/bad_state.json -fail

test-bad-trans:
	@$(TEST) $(WRAPBOSS) t/invalid/bad_trans.json -fail

test-bad-weight:
	@$(TEST) $(WRAPBOSS) t/invalid/bad_weight.json -fail

test-cyclic:
	@$(TEST) $(WRAPBOSS) t/invalid/cyclic.json -fail

# Non-transducer I/O tests
IO_TESTS = test-fastseq test-empty-fastseq test-seqpair test-seqpairlist test-env test-params test-constraints test-dot
test-fastseq: t/bin/testfastseq
	@$(WRAPTEST) t/bin/testfastseq t/tc1/CAA25498.fa t/expect/CAA25498.fa

test-empty-fastseq: t/bin/testfastseq
	@$(WRAPTEST) t/bin/testfastseq t/io/empty-1line.fa -idem
	@$(WRAPTEST) t/bin/testfastseq t/io/empty.fa t/io/empty-1line.fa

test-seqpair: t/bin/testseqpair
	@$(WRAPTEST) t/bin/testseqpair t/io/tiny.json -idem
	@$(WRAPTEST) t/bin/testseqpair t/io/tinypath.json -idem
	@$(WRAPTEST) t/bin/testseqpair t/io/tinyfail.json -fail
	@$(WRAPTEST) t/bin/testseqpair t/io/tinypathnames.json t/io/tinypath.json
	@$(WRAPTEST) t/bin/testseqpair t/io/tinypathonly.json t/expect/tinypathonly.json

test-seqpairlist: t/bin/testseqpairlist
	@$(WRAPTEST) t/bin/testseqpairlist t/io/seqpairlist.json -idem

test-env: t/bin/testenv
	@$(WRAPTEST) t/bin/testenv t/io/tinypath.json full t/expect/tinypath_full_env.json
	@$(WRAPTEST) t/bin/testenv t/io/tinypath.json path t/expect/tinypath_path_env.json
	@$(WRAPTEST) t/bin/testenv t/io/smallpath.json path t/expect/smallpath_path_env.json
	@$(WRAPTEST) t/bin/testenv t/io/smallpath.json 0 t/expect/smallpath_area0_env.json
	@$(WRAPTEST) t/bin/testenv t/io/smallpath.json 1 t/expect/smallpath_area1_env.json
	@$(WRAPTEST) t/bin/testenv t/io/smallpath.json 2 t/expect/smallpath_area2_env.json
	@$(WRAPTEST) t/bin/testenv t/io/smallpath.json 3 t/expect/smallpath_area3_env.json
	@$(WRAPTEST) t/bin/testenv t/io/smallpath.json 4 t/expect/smallpath_area4_env.json
	@$(WRAPTEST) t/bin/testenv t/io/smallpath.json 5 t/expect/smallpath_area4_env.json
	@$(WRAPTEST) t/bin/testenv t/io/asympath.json 0 t/expect/asympath_area0_env.json
	@$(WRAPTEST) t/bin/testenv t/io/asympath.json 1 t/expect/asympath_area1_env.json
	@$(WRAPTEST) t/bin/testenv t/io/asympath.json p t/expect/asympath_area0_env.json

test-params: t/bin/testparams
	@$(WRAPTEST) t/bin/testparams t/io/params.json -idem

test-constraints: t/bin/testconstraints
	@$(WRAPTEST) t/bin/testconstraints t/io/constraints.json -idem

test-dot:
	@$(TEST) $(WRAPBOSS) t/machine/bitnoise.json --graphviz t/expect/bitnoise.dot
	@$(TEST) $(WRAPBOSS) t/machine/bitnoise.json t/machine/bitnoise.json --graphviz t/expect/bitnoise2.dot
	@$(TEST) $(WRAPBOSS) t/machine/bitnoise.json --graphviz --dot-no-merge t/expect/bitnoise-no-merge.dot
	@$(TEST) $(WRAPBOSS) t/machine/bitnoise.json --graphviz --dot-show-io t/expect/bitnoise-show-io.dot
	@$(TEST) $(WRAPBOSS) tutorial/metalhead.json --graphviz t/expect/metalhead.dot

# Symbolic algebra tests
ALGEBRA_TESTS = test-list-params test-deriv-xplusy-x test-deriv-xy-x test-eval-1plus2
test-list-params: t/bin/testlistparams
	@$(WRAPTEST) t/bin/testlistparams t/algebra/x_plus_y.json t/expect/xy_params.txt

test-deriv-xplusy-x: t/bin/testderiv
	@$(WRAPTEST) t/bin/testderiv t/algebra/x_plus_y.json x t/expect/dxplusy_dx.json

test-deriv-xy-x: t/bin/testderiv
	@$(WRAPTEST) t/bin/testderiv t/algebra/x_times_y.json x t/expect/dxy_dx.json

test-eval-1plus2: t/bin/testeval
	@$(WRAPTEST) t/bin/testeval t/algebra/x_plus_y.json t/algebra/params.json t/expect/1_plus_2.json

# Dynamic programming tests
DP_TESTS = test-fwd-bitnoise-params-tiny test-back-bitnoise-params-tiny test-fb-bitnoise-params-tiny test-max-bitnoise-params-tiny test-fit-bitnoise-seqpairlist test-funcs test-single-param test-align-stutter-noise test-counts test-counts2 test-counts3 test-count-motif
test-fwd-bitnoise-params-tiny: t/bin/testforward
	@$(WRAPTEST) t/bin/testforward t/machine/bitnoise.json t/io/params.json t/io/tiny.json t/expect/fwd-bitnoise-params-tiny.json

test-back-bitnoise-params-tiny: t/bin/testbackward
	@$(WRAPTEST) t/bin/testbackward t/machine/bitnoise.json t/io/params.json t/io/tiny.json t/expect/back-bitnoise-params-tiny.json

test-fb-bitnoise-params-tiny: t/bin/testcounts
	@$(WRAPTEST) t/bin/testcounts t/machine/bitnoise.json t/io/params.json t/io/tiny.json t/expect/fwdback-bitnoise-params-tiny.json

test-max-bitnoise-params-tiny: t/bin/testmaximize
	@$(TEST) python3 t/roundfloats.py 4 t/bin/testmaximize t/machine/bitnoise.json t/io/params.json t/io/tiny.json t/io/pqcons.json t/expect/max-bitnoise-params-tiny.json

test-fit-bitnoise-seqpairlist:
	@$(TEST) python3 t/roundfloats.py 4 $(WRAPBOSS) t/machine/bitnoise.json -N t/io/pqcons.json -D t/io/seqpairlist.json -T t/expect/fit-bitnoise-seqpairlist.json
	@$(TEST) python3 t/roundfloats.py 4 $(WRAPBOSS) t/machine/bitnoise.json -N t/io/pqcons.json -D t/io/pathlist.json -T t/expect/fit-bitnoise-seqpairlist.json

test-funcs:
	@$(TEST) python3 t/roundfloats.py 4 $(WRAPBOSS) -F t/io/e=0.json t/machine/bitnoise.json t/machine/bsc.json -N t/io/pqcons.json -D t/io/seqpairlist.json -T t/expect/test-funcs.json

test-single-param:
	@$(TEST) python3 t/roundfloats.py 4 $(WRAPBOSS) t/machine/bitnoise.json t/machine/bsc.json -N t/io/econs.json -D t/io/seqpairlist.json -T -F t/io/params.json t/expect/single-param.json

test-align-stutter-noise:
	@$(TEST) $(WRAPBOSS) t/machine/bitstutter.json t/machine/bitnoise.json -P t/io/params.json -D t/io/difflen.json -A t/expect/align-stutter-noise-difflen.json

test-counts:
	@$(TEST) $(WRAPBOSS) --generate-chars 101 -m t/machine/bitnoise.json --recognize-chars 001 -P t/io/params.json -N t/io/pqcons.json -C t/expect/counts.json

test-counts2:
	@$(TEST) $(WRAPBOSS) t/machine/bitnoise.json --input-chars 101 --output-chars 001 -P t/io/params.json -N t/io/pqcons.json -C t/expect/counts.json

test-counts3:
	@$(TEST) $(WRAPBOSS) t/machine/counter.json --output-chars xxx -C t/expect/counter.json
	@$(TEST) $(WRAPBOSS) --generate-one x --count-copies p --output-chars xxx -C t/expect/counter.json

test-count-motif:
	@$(TEST) $(WRAPBOSS) --generate-uniform ACGT --concat --generate-chars CATCAG --concat --begin --generate-one A --count-copies n --end --concat --generate-chars TATA --concat --generate-uniform ACGT --recognize-json t/io/nanopore_test_seq.json -C t/expect/count11.json
	@$(TEST) python3 t/roundfloats.py 1 $(WRAPBOSS) --generate-uniform ACGT --concat --generate-chars CATCAG --concat --begin --generate-one A --count-copies n --end --concat --generate-chars TATA --concat --generate-uniform ACGT --recognize-csv t/csv/nanopore_test.csv -C t/expect/count9.json
	@$(TEST) python3 t/roundfloats.py 1 $(WRAPBOSS) --generate-uniform ACGT --concat --generate-chars CAT --concat --begin --generate-one T --count-copies n --end --concat --generate-chars GG --concat --generate-uniform ACGT --recognize-csv t/csv/nanopore_test.csv -C t/expect/count4.json

# Code generation tests
CODEGEN_TESTS = test-101-bitnoise-001 test-101-bitstutternoise-0011 test-101-bitnoise-001-compiled test-101-bitnoise-001-compiled-seq test-101-bitstutternoise-0011-compiled-seq-forward test-101-bitstutternoise-0011-compiled-seq-viterbi test-101-bitnoise-001-compiled-seq2prof test-101-bitnoise-001-compiled-js test-101-bitnoise-001-compiled-js-seq test-101-bitnoise-001-compiled-js-seq2prof

# C++
t/src/%/prof/test.cpp: t/machine/%.json $(BOSSTARGET) src/softplus.h src/getparams.h t/src/testcompiledprof.cpp
	test -e $(dir $@) || mkdir -p $(dir $@)
	$(WRAPBOSS) t/machine/$*.json --cpp64 --inseq profile --outseq profile --codegen $(dir $@)
	cp t/src/testcompiledprof.cpp $@

t/src/%/seq/test.cpp: t/machine/%.json $(BOSSTARGET) src/softplus.h src/getparams.h t/src/testcompiledseq.cpp
	@test -e $(dir $@) || mkdir -p $(dir $@)
	@$(WRAPBOSS) t/machine/$*.json --cpp64 --inseq string --outseq string --codegen $(dir $@)
	@cp t/src/testcompiledseq.cpp $@

t/src/%/seqvit/test.cpp: t/machine/%.json $(BOSSTARGET) src/softplus.h src/getparams.h t/src/testcompiledseq.cpp
	@test -e $(dir $@) || mkdir -p $(dir $@)
	@$(WRAPBOSS) t/machine/$*.json --cpp64 --inseq string --outseq string --codegen $(dir $@) --compileviterbi
	@cp t/src/testcompiledseq.cpp $@

t/src/%/seq2prof/test.cpp: t/machine/%.json $(BOSSTARGET) src/softplus.h src/getparams.h t/src/testcompiledseq2prof.cpp
	@test -e $(dir $@) || mkdir -p $(dir $@)
	@$(WRAPBOSS) t/machine/$*.json --cpp64 --inseq string --outseq profile --codegen $(dir $@)
	@cp t/src/testcompiledseq2prof.cpp $@

t/src/%/fasta/test.cpp: t/machine/%.json $(BOSSTARGET) src/softplus.h src/getparams.h t/src/testcompiledfasta.cpp
	@test -e $(dir $@) || mkdir -p $(dir $@)
	@$(WRAPBOSS) t/machine/$*.json --cpp64 --inseq string --outseq string --codegen $(dir $@)
	@cp t/src/testcompiledfasta.cpp $@

t/src/%/fasta2strand/test.cpp: t/machine/%.json $(BOSSTARGET) src/softplus.h src/getparams.h t/src/testcompiledfasta2strand.cpp
	@test -e $(dir $@) || mkdir -p $(dir $@)
	@$(WRAPBOSS) t/machine/$*.json --cpp64 --inseq string --outseq string --codegen $(dir $@)
	@cp t/src/testcompiledfasta2strand.cpp $@

test-101-bitnoise-001:
	@$(TEST) python3 t/roundfloats.py 4 js/stripnames.js $(WRAPBOSS) --generate-json t/io/seq101.json -m t/machine/bitnoise.json --recognize-json t/io/seq001.json -P t/io/params.json -N t/io/pqcons.json -L t/expect/101-bitnoise-001.json

test-101-bitstutternoise-0011:
	@$(TEST) python3 t/roundfloats.py 3 js/stripnames.js $(WRAPBOSS) --generate-json t/io/seq101.json -m t/machine/bitstutter-noise.json --recognize-chars 0011 -P t/io/params.json -N t/io/pqcons.json -L t/expect/101-bitstutternoise-fwd-0011.json
	@$(TEST) python3 t/roundfloats.py 3 js/stripnames.js $(WRAPBOSS) --generate-json t/io/seq101.json -m t/machine/bitstutter-noise.json --recognize-chars 0011 -P t/io/params.json -N t/io/pqcons.json -V t/expect/101-bitstutternoise-vit-0011.json

test-101-bitnoise-001-compiled: t/codegen/bitnoise/prof/test
	@$(TEST) python3 t/roundfloats.py 4 js/stripnames.js $< t/csv/prof101.csv t/csv/prof001.csv t/io/params.json t/expect/101-bitnoise-001.json

test-101-bitnoise-001-compiled-seq: t/codegen/bitnoise/seq/test
	@$(TEST) python3 t/roundfloats.py 4 js/stripnames.js $< 101 001 t/io/params.json t/expect/101-bitnoise-001.json

test-101-bitnoise-001-compiled-seq2prof: t/codegen/bitnoise/seq2prof/test
	@$(TEST) python3 t/roundfloats.py 4 js/stripnames.js $< 101 t/csv/prof001.csv t/io/params.json t/expect/101-bitnoise-001.json

test-101-bitstutternoise-0011-compiled-seq-forward: t/codegen/bitstutter-noise/seq/test
	@$(TEST) python3 t/roundfloats.py 3 js/stripnames.js $< 101 0011 t/io/params.json t/expect/101-bitstutternoise-fwd-0011.json

test-101-bitstutternoise-0011-compiled-seq-viterbi: t/codegen/bitstutter-noise/seqvit/test
	@$(TEST) python3 t/roundfloats.py 3 js/stripnames.js $< 101 0011 t/io/params.json t/expect/101-bitstutternoise-vit-0011.json

# JavaScript
js/lib/%/prof/test.js: t/machine/%.json $(BOSSTARGET) js/lib/softplus.js js/lib/getparams.js js/lib/testcompiledprof.js
	test -e $(dir $@) || mkdir -p $(dir $@)
	$(WRAPBOSS) t/machine/$*.json --js --inseq profile --outseq profile --codegen $(dir $@)
	cat js/lib/testcompiledprof.js $(dir $@)computeForward*.js >$@
	chmod +x $@
	cp js/lib/softplus.js js/lib/getparams.js $(dir $@)

js/lib/%/seq/test.js: t/machine/%.json $(BOSSTARGET) js/lib/softplus.js js/lib/getparams.js js/lib/testcompiledprof.js
	test -e $(dir $@) || mkdir -p $(dir $@)
	$(WRAPBOSS) t/machine/$*.json --js --inseq string --outseq string --codegen $(dir $@)
	cat js/lib/testcompiledprof.js $(dir $@)computeForward*.js >$@
	chmod +x $@
	cp js/lib/softplus.js js/lib/getparams.js $(dir $@)

js/lib/%/seq2prof/test.js: t/machine/%.json $(BOSSTARGET) js/lib/softplus.js js/lib/getparams.js js/lib/testcompiledprof.js
	test -e $(dir $@) || mkdir -p $(dir $@)
	$(WRAPBOSS) t/machine/$*.json --js --inseq string --outseq profile --codegen $(dir $@)
	cat js/lib/testcompiledprof.js $(dir $@)computeForward*.js >$@
	chmod +x $@
	cp js/lib/softplus.js js/lib/getparams.js $(dir $@)

test-101-bitnoise-001-compiled-js: js/lib/bitnoise/prof/test.js
	@$(TEST) python3 t/roundfloats.py 4 js/stripnames.js node $< --inprof t/csv/prof101.csv --outprof t/csv/prof001.csv --params t/io/params.json t/expect/101-bitnoise-001.json

test-101-bitnoise-001-compiled-js-seq: js/lib/bitnoise/seq/test.js
	@$(TEST) python3 t/roundfloats.py 4 js/stripnames.js node $< --inseq 101 --outseq 001 --params t/io/params.json t/expect/101-bitnoise-001.json

test-101-bitnoise-001-compiled-js-seq2prof: js/lib/bitnoise/seq2prof/test.js
	@$(TEST) python3 t/roundfloats.py 4 js/stripnames.js node $< --inseq 101 --outprof t/csv/prof001.csv --params t/io/params.json t/expect/101-bitnoise-001.json

# Encoding/decoding
DECODE_TESTS = test-decode-bitecho-101 test-bintern test-hamming

test-decode-bitecho-101:
	@$(TEST) $(WRAPBOSS) t/machine/bitecho.json --recognize-chars 101 --prefix-decode t/expect/decode-bitecho-101.json

test-bintern:
	@$(TEST) $(WRAPBOSS) --generate-chars 101 t/machine/bintern.json --prefix-encode t/expect/encode-g101-bintern.json
	@$(TEST) $(WRAPBOSS) --input-chars 101 t/machine/bintern.json --prefix-encode t/expect/encode-i101-bintern.json
	@$(TEST) $(WRAPBOSS) t/machine/bintern.json --recognize-chars 12222 --prefix-decode t/expect/decode-a12222-bintern.json
	@$(TEST) $(WRAPBOSS) t/machine/bintern.json --output-chars 12222 --prefix-decode t/expect/decode-o12222-bintern.json
	@$(TEST) $(WRAPBOSS) t/machine/bintern.json --recognize-chars 12222 --beam-decode t/expect/decode-a12222-bintern.json
	@$(TEST) $(WRAPBOSS) t/machine/bintern.json --output-chars 12222 --beam-decode t/expect/decode-o12222-bintern.json

test-hamming:
	@$(TEST) $(WRAPBOSS) --preset hamming74 --viterbi-encode --input-chars 0000000100100011010001010110011110001001101010111100110111101111 t/expect/hamming74.json
	@$(TEST) $(WRAPBOSS) --preset hamming74 --prefix-encode --input-chars 0000000100100011010001010110011110001001101010111100110111101111 t/expect/hamming74.json
	@$(TEST) $(WRAPBOSS) --preset hamming74 --beam-encode --input-chars 0000000100100011010001010110011110001001101010111100110111101111 t/expect/hamming74.json

# Expression parser tests
EXPR_TESTS = test-expr-exp test-expr-log test-expr-power test-expr-unary-neg test-expr-parens test-expr-scinotation
test-expr-exp:
	@$(TEST) $(WRAPBOSS) -w 'exp(0)' t/expect/null-1.json

test-expr-log:
	@$(TEST) $(WRAPBOSS) -w 'exp(log(2))' t/expect/null-2.json

test-expr-power:
	@$(TEST) $(WRAPBOSS) -w '2^3' t/expect/null-8.json

test-expr-unary-neg:
	@$(TEST) $(WRAPBOSS) -w '-(-(2))' t/expect/null-neg-neg-2.json

test-expr-parens:
	@$(TEST) $(WRAPBOSS) -w '(1+1)' t/expect/null-2.json

test-expr-scinotation:
	@$(TEST) $(WRAPBOSS) -w '2e0' t/expect/null-2.json

# Additional CLI tests
CLI_TESTS = test-viterbi-decode-bitecho test-cool-decode-bitecho test-mcmc-decode-bitecho test-random-encode-bitecho test-evaluate test-regex
test-viterbi-decode-bitecho:
	@$(TEST) $(WRAPBOSS) t/machine/bitecho.json --recognize-chars 101 --viterbi-decode t/expect/decode-bitecho-101.json

test-cool-decode-bitecho:
	@$(TEST) $(WRAPBOSS) t/machine/bitecho.json --recognize-chars 101 --cool-decode --seed 42 t/expect/decode-bitecho-101.json

test-mcmc-decode-bitecho:
	@$(TEST) $(WRAPBOSS) t/machine/bitecho.json --recognize-chars 101 --mcmc-decode --seed 42 t/expect/decode-bitecho-101.json

test-random-encode-bitecho:
	@$(TEST) $(WRAPBOSS) t/machine/bitecho.json --input-chars 101 --random-encode --seed 42 t/expect/random-encode-bitecho-101.json

test-evaluate:
	@$(TEST) $(WRAPBOSS) t/machine/bitnoise.json -P t/io/params.json --evaluate t/expect/evaluate-bitnoise.json

test-regex:
	@$(TEST) $(WRAPBOSS) --regex '[01]+' t/expect/regex-01plus.json

# Preset load tests
PRESETS = null compdna comprna dnapsw protpsw translate prot2dna psw2dna iupacdna iupacaa dna2rna rna2dna bintern terndna jukescantor dnapswnbr tkf91root tkf91branch tolower toupper hamming31 hamming74
PRESET_TESTS = $(addprefix test-preset-,$(PRESETS))
$(PRESET_TESTS): test-preset-%:
	@$(WRAPBOSS) --preset $* >t/expect/preset-$*.tmp.json 2>/dev/null
	@$(TEST) $(WRAPBOSS) t/expect/preset-$*.tmp.json -idem

# JSON API operation tests
JSON_API_TESTS = test-json-concat test-json-union test-json-intersect test-json-intersect-sum test-json-intersect-unsort test-json-compose-sum test-json-compose-unsort test-json-loop test-json-opt test-json-star test-json-plus test-json-eliminate test-json-merge test-json-reverse test-json-revcomp test-json-transpose
test-json-concat:
	@$(TEST) $(WRAPBOSS) t/machine/concat-001-101.json t/expect/json-concat.json

test-json-union:
	@$(TEST) $(WRAPBOSS) t/machine/union-001-101.json t/expect/json-union.json

test-json-intersect:
	@$(TEST) $(WRAPBOSS) t/machine/intersect-r001-r101.json t/expect/json-intersect.json

test-json-intersect-sum:
	@$(TEST) $(WRAPBOSS) t/machine/intersect-sum-r001-r101.json t/expect/json-intersect-sum.json

test-json-intersect-unsort:
	@$(TEST) $(WRAPBOSS) t/machine/intersect-unsort-r001-r101.json t/expect/json-intersect-unsort.json

test-json-compose-sum:
	@$(TEST) $(WRAPBOSS) t/machine/compose-sum-bitecho.json t/expect/json-compose-sum.json

test-json-compose-unsort:
	@$(TEST) $(WRAPBOSS) t/machine/compose-unsort-bitecho.json t/expect/json-compose-unsort.json

test-json-loop:
	@$(TEST) $(WRAPBOSS) t/machine/loop-gen1.json t/expect/json-loop.json

test-json-opt:
	@$(TEST) $(WRAPBOSS) t/machine/opt-gen1.json t/expect/json-opt.json

test-json-star:
	@$(TEST) $(WRAPBOSS) t/machine/star-gen1.json t/expect/json-star.json

test-json-plus:
	@$(TEST) $(WRAPBOSS) t/machine/plus-gen1.json t/expect/json-plus.json

test-json-eliminate:
	@$(TEST) $(WRAPBOSS) t/machine/eliminate-silent.json t/expect/json-eliminate.json

test-json-merge:
	@$(TEST) $(WRAPBOSS) t/machine/merge-json.json t/expect/merge-json.json

test-json-reverse:
	@$(TEST) $(WRAPBOSS) t/machine/reverse-gen001.json t/expect/json-reverse.json

test-json-revcomp:
	@$(TEST) $(WRAPBOSS) t/machine/revcomp-genAGC.json t/expect/json-revcomp.json

test-json-transpose:
	@$(TEST) $(WRAPBOSS) t/machine/transpose-gen001.json t/expect/json-transpose.json

# Top-level test target
TESTS = $(INVALID_SCHEMA_TESTS) $(VALID_SCHEMA_TESTS) $(COMPOSE_TESTS) $(CONSTRUCT_TESTS) $(INVALID_CONSTRUCT_TESTS) $(IO_TESTS) $(ALGEBRA_TESTS) $(DP_TESTS) $(CODEGEN_TESTS) $(DECODE_TESTS) $(EXPR_TESTS) $(CLI_TESTS) $(PRESET_TESTS) $(JSON_API_TESTS)
TESTLEN = $(shell python3 -c "print(max(len(s) for s in '$(TESTS)'.split()))")

TEST = python3 t/testexpect.py $@ $(TESTLEN)
WRAPTEST = $(TEST)

test: $(BOSSTARGET) $(TESTS)

# WebGPU tests (CPU fallback, Node.js)
WEBGPU_TESTS = test-webgpu-cpu test-webgpu-agreement

test-webgpu-cpu:
	@node js/webgpu/test/test-cpu.mjs

test-webgpu-agreement:
	@node js/webgpu/test/test-gpu-cpu-agreement.mjs

test-webgpu: $(WEBGPU_TESTS)

# Schema validator
ajv:
	npm install ajv-cli

validate-$D-$F:
	ajv -s schema/machine.json -r schema/expr.json -d t/$D/$F.json

# README
README.md: $(BOSSTARGET)
	$(WRAPBOSS) -h | python3 -c "import sys,html; helptext=html.escape(sys.stdin.read()); lines=open('README.md').readlines(); idx=next((i for i,l in enumerate(lines) if '<pre>' in l),len(lines)); sys.stdout.write(''.join(lines[:idx+1])+'<code>\n'+helptext+'</code></pre>\n')" >temp.md
	mv temp.md $@
