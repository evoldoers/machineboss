.SECONDARY:

MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MAKEFILE_DIR := $(dir $(MAKEFILE_PATH))

# try to figure out where GSL is
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
GSL_LIBS = -L$(GSL_PREFIX)/lib -lgsl -lgslcblas -lm
endif

# NB pkg-config support for Boost is lacking; see https://svn.boost.org/trac/boost/ticket/1094
BOOST_PREFIX = /usr
ifeq (,$(wildcard $(BOOST_PREFIX)/include/boost/regex.h))
BOOST_PREFIX = /usr/local
ifeq (,$(wildcard $(BOOST_PREFIX)/include/boost/regex.h))
BOOST_PREFIX =
endif
endif

BOOST_FLAGS =
BOOST_LIBS =
ifneq (,$(BOOST_PREFIX))
BOOST_FLAGS := -I$(BOOST_PREFIX)/include
BOOST_LIBS := -L$(BOOST_PREFIX)/lib -lboost_regex -lboost_program_options
endif

# HTSlib
HTS_FLAGS = $(shell pkg-config --cflags htslib)
HTS_LIBS = $(shell pkg-config --libs htslib)

# HDF5
HDF5_DIR ?= /usr/local
HDF5_INCLUDE_DIR ?= $(HDF5_DIR)/include
HDF5_LIB_DIR ?= $(HDF5_DIR)/lib
HDF5_LIB ?= hdf5
HDF5_FLAGS = -isystem $(HDF5_INCLUDE_DIR)
HDF5_LIBS = -L$(HDF5_LIB_DIR) -l$(HDF5_LIB)

# Uncomment if no HDF5
#HDF5_FLAGS =
#HDF5_LIBS =

# install dir
PREFIX = /usr/local
INSTALL_BIN = $(PREFIX)/bin

# other flags
ifneq (,$(findstring 32bit,$(MAKECMDGOALS)))
BUILD_FLAGS = -DIS32BIT
else
BUILD_FLAGS =
endif

ifneq (,$(findstring debug,$(MAKECMDGOALS)))
CPP_FLAGS = -std=c++11 -g -DUSE_VECTOR_GUARDS -DDEBUG $(GSL_FLAGS) $(BOOST_FLAGS) $(BUILD_FLAGS)
else
ifneq (,$(findstring unoptimized,$(MAKECMDGOALS)))
CPP_FLAGS = -std=c++11 -g $(GSL_FLAGS) $(BOOST_FLAGS) $(BUILD_FLAGS)
else
CPP_FLAGS = -std=c++11 -g -O3 $(GSL_FLAGS) $(BOOST_FLAGS) $(BUILD_FLAGS)
endif
endif
CPP_FLAGS += -Isrc -Iext -Iext/nlohmann_json
LD_FLAGS = -lstdc++ -lz $(GSL_LIBS) $(BOOST_LIBS)

# files
CPP_FILES = $(wildcard src/*.cpp)
OBJ_FILES = $(subst src/,obj/,$(subst .cpp,.o,$(CPP_FILES)))

# try clang++, fall back to g++
CPP = clang++
ifeq (, $(shell which $(CPP)))
CPP = g++
endif

# pwd
PWD = $(shell pwd)

# /bin/sh
SH = /bin/sh

# Targets

BOSS = bossmachine

all: $(BOSS)

install: $(BOSS)
	cp bin/$(BOSS) $(INSTALL_BIN)/$(BOSS)

# Main build rules
bin/%: $(OBJ_FILES) obj/%.o target/%.cpp
	@test -e $(dir $@) || mkdir -p $(dir $@)
	$(CPP) $(LD_FLAGS) -o $@ obj/$*.o $(OBJ_FILES)

obj/%.o: src/%.cpp
	@test -e $(dir $@) || mkdir -p $(dir $@)
	$(CPP) $(CPP_FLAGS) -c -o $@ $<

obj/%.o: target/%.cpp
	@test -e $(dir $@) || mkdir -p $(dir $@)
	$(CPP) $(CPP_FLAGS) -c -o $@ $<

t/bin/%: $(OBJ_FILES) obj/%.o t/src/%.cpp
	@test -e $(dir $@) || mkdir -p $(dir $@)
	$(CPP) $(LD_FLAGS) -o $@ obj/$*.o $(OBJ_FILES)

t/codegen/%: $(OBJ_FILES) obj/%.o
	$(MAKE) `ls $(dir t/src/$*)computeForward*.cpp | perl -pe 's/t\/src/obj/;s/\.cpp/.o/'`
	@test -e $(dir $@) || mkdir -p $(dir $@)
	$(CPP) $(LD_FLAGS) -o $@ $^ `ls $(dir t/src/$*)computeForward*.cpp | perl -pe 's/t\/src/obj/;s/\.cpp/.o/'`

obj/%.o: t/src/%.cpp
	@test -e $(dir $@) || mkdir -p $(dir $@)
	$(CPP) $(CPP_FLAGS) -c -o $@ $<

$(BOSS): bin/$(BOSS)

clean:
	rm -rf bin/* t/bin/* obj/*

debug unoptimized 32bit:

# Schemas & presets
# The relevant pseudotargets are generate-schemas and generate-presets (biomake required)
# Get biomake here: https://github.com/evoldoers/biomake
generate-$(CATEGORY)s: $(patsubst $(CATEGORY)/%.json,src/$(CATEGORY)/%.h,$(wildcard $(CATEGORY)/*.json))
	touch src/$(CATEGORY).cpp

src/preset/$(FILE).h: preset/$(FILE).json
	xxd -i $< >$@

preset/protpsw.json constraints/protpsw.json: js/makepsw.js
	js/makepsw.js -a ACDEFGHIKLMNPQRSTVWY -n protpsw >$@

preset/dnapsw.json constraints/dnapsw.json: js/makepsw.js
	js/makepsw.js -a ACGT -n dnapsw >$@

preset/%.json: js/%.js
	node $< >$@

preset/prot2dna.json: preset/flankbase.json preset/pint.json preset/translate.json preset/simple_introns.json preset/base2acgt.json
	bin/$(BOSS) -v6 preset/flankbase.json '.' '(' preset/pint.json '=>' preset/translate.json '=>' preset/simple_introns.json ')' '.' preset/flankbase.json '=>' preset/base2acgt.json >$@

preset/psw2dna.json: preset/flankbase.json preset/pswint.json preset/translate.json preset/simple_introns.json preset/base2acgt.json
	bin/$(BOSS) -v6 preset/flankbase.json '.' '(' preset/pswint.json '=>' preset/translate.json '=>' preset/simple_introns.json ')' '.' preset/flankbase.json '=>' preset/base2acgt.json >$@

# valijson doesn't like the URLs, but other schema validators demand them, so strip them out for xxd
src/schema/$(FILE).h: schema/$(FILE).json.nourl
	xxd -i $< | sed 's/.nourl//' >$@

schema/$(FILE).json.nourl: schema/$(FILE).json
	grep -v '"id": "http' $< >$@

# Transducer composition tests
COMPOSE_TESTS = test-echo test-echo2 test-echo2-expr test-echo-stutter test-stutter2 test-noise2 test-unitindel2 test-machine-params
test-echo:
	@$(TEST) bin/$(BOSS) t/machine/bitecho.json t/expect/bitecho.json

test-echo2:
	@$(TEST) bin/$(BOSS) t/machine/bitecho.json t/machine/bitecho.json t/expect/bitecho-bitecho.json

test-echo2-expr:
	@$(TEST) bin/$(BOSS) t/machine/compose-bitecho-bitecho.json t/expect/bitecho-bitecho.json

test-echo-stutter:
	@$(TEST) bin/$(BOSS) t/machine/bitecho.json t/machine/bitstutter.json t/expect/bitecho-bitstutter.json

test-stutter2:
	@$(TEST) bin/$(BOSS) t/machine/bitstutter.json t/machine/bitstutter.json t/expect/bitstutter-bitstutter.json

test-noise2:
	@$(TEST) bin/$(BOSS) t/machine/bitnoise.json t/machine/bitnoise.json --showparams t/expect/bitnoise-bitnoise.json

test-unitindel2:
	@$(TEST) bin/$(BOSS) t/machine/unitindel.json t/machine/unitindel.json --showparams t/expect/unitindel-unitindel.json

test-machine-params:
	@$(TEST) bin/$(BOSS) t/machine/params.json -idem

# Transducer construction tests
CONSTRUCT_TESTS = test-generator test-acceptor test-wild-generator test-wild-acceptor test-union test-intersection test-brackets test-kleene test-loop test-noisy-loop test-concat test-eliminate test-reverse test-revcomp test-transpose test-weight test-shorthand test-hmmer test-csv test-csv-tiny test-csv-tiny-fail test-csv-tiny-empty test-nanopore test-nanopore-prefix
test-generator:
	@$(TEST) bin/$(BOSS) --generate t/io/seq101.json t/expect/generator101.json

test-acceptor:
	@$(TEST) bin/$(BOSS) --accept t/io/seq001.json t/expect/acceptor001.json

test-wild-generator:
	@$(TEST) bin/$(BOSS) --generate-wild ACGT t/expect/ACGT_generator.json

test-wild-acceptor:
	@$(TEST) bin/$(BOSS) --accept-wild ACGT t/expect/ACGT_acceptor.json

test-union:
	@$(TEST) bin/$(BOSS) --generate t/io/seq001.json -u t/expect/generator101.json t/expect/generate-101-or-001.json

test-intersection:
	@$(TEST) bin/$(BOSS) t/machine/bitnoise.json -m --accept t/io/seq001.json -i --accept t/io/seq101.json t/expect/noise-001-and-101.json

test-brackets:
	@$(TEST) bin/$(BOSS) --begin t/machine/bitnoise.json --accept t/io/seq001.json --end -i --accept t/io/seq101.json t/expect/noise-001-and-101.json

test-kleene:
	@$(TEST) bin/$(BOSS) --generate t/io/seq001.json -K t/expect/generate-multiple-001.json

test-loop:
	@$(TEST) bin/$(BOSS) --accept t/io/seq101.json -o --accept t/io/seq001.json t/expect/101-loop-001.json

test-noisy-loop:
	@$(TEST) bin/$(BOSS) t/machine/bitnoise.json --accept t/io/seq101.json -o --accept t/io/seq001.json t/expect/noisy-101-loop-001.json

test-concat:
	@$(TEST) bin/$(BOSS) --generate t/io/seq001.json -c t/expect/generator101.json t/expect/concat-001-101.json

test-eliminate:
	@$(TEST) bin/$(BOSS) -n t/machine/silent.json t/expect/silent-elim.json
	@$(TEST) bin/$(BOSS) -n t/machine/silent2.json t/expect/silent2-elim.json
	@$(TEST) bin/$(BOSS) -n t/machine/silent3.json t/expect/silent3-elim.json

test-reverse:
	@$(TEST) bin/$(BOSS) -e --generate t/io/seq001.json t/expect/generator001-reversed.json

test-revcomp:
	@$(TEST) bin/$(BOSS) -r --generate t/io/seqAGC.json t/expect/generatorAGC-revcomp.json

test-transpose:
	@$(TEST) bin/$(BOSS) -t --generate t/io/seq001.json t/expect/acceptor001.json

test-weight:
	@$(TEST) bin/$(BOSS) -w p t/expect/null-p.json
	@$(TEST) bin/$(BOSS) -w 2 t/expect/null-2.json
	@$(TEST) bin/$(BOSS) -w .5 t/expect/null-0.5.json
	@$(TEST) bin/$(BOSS) -w '{"*":["p","q"]}' t/expect/null-pq.json
	@$(TEST) bin/$(BOSS) -w '{"*":[1,2]}' t/expect/null-2.json
	@$(TEST) bin/$(BOSS) -w '{"/":[1,2]}' t/expect/null-0.5.json
	@$(TEST) bin/$(BOSS) --accept-wild ACGT --weight-input p --reciprocal t/expect/null-weight-recip.json

test-shorthand:
	@$(TEST) bin/$(BOSS) '(' t/machine/bitnoise.json '>>' 101 ')' '&&' '>>' 001 '.' '>>' AGC '#' x t/expect/shorthand.json

test-hmmer:
	@$(TEST) bin/$(BOSS) --hmmer t/hmmer/fn3.hmm t/expect/fn3.json

test-csv:
	@$(TEST) bin/$(BOSS) --csv t/csv/test.csv t/expect/csvtest.json

test-csv-tiny:
	@$(TEST) js/stripnames.js bin/$(BOSS) -L --generate t/io/tiny_uc.json --transpose --csv t/csv/tiny_uc.csv t/expect/tiny_uc.json

test-csv-tiny-fail:
	@$(TEST) js/stripnames.js bin/$(BOSS) -L --generate t/io/tiny_lc.json --transpose --csv t/csv/tiny_uc.csv -fail

test-csv-tiny-empty:
	@$(TEST) js/stripnames.js bin/$(BOSS) -L --generate t/io/empty.json --transpose --csv t/csv/tiny_uc.csv t/expect/tiny_empty.json

test-nanopore:
	@$(TEST) js/stripnames.js bin/$(BOSS) -L --generate t/io/nanopore_test_seq.json --transpose --csv t/csv/nanopore_test.csv t/expect/nanopore_test.json

test-nanopore-prefix:
	@$(TEST) js/stripnames.js bin/$(BOSS) -L --generate t/io/nanopore_test_seq.json --concat t/machine/acgt_wild.json --transpose --csv t/csv/nanopore_test.csv t/expect/nanopore_test_prefix.json

# Invalid transducer construction tests
INVALID_CONSTRUCT_TESTS = test-unmatched-begin test-unmatched-end test-empty-brackets test-impossible-intersect test-missing-machine
test-unmatched-begin:
	@$(TEST) bin/$(BOSS) --begin -fail

test-unmatched-end:
	@$(TEST) bin/$(BOSS) --end -fail

test-empty-brackets:
	@$(TEST) bin/$(BOSS) --begin --end -fail

test-missing-machine:
	@$(TEST) bin/$(BOSS) t/machine/bitnoise.json -m -m t/machine/bitnoise.json t/machine/bitnoise.json -fail

test-impossible-intersect:
	@$(TEST) bin/$(BOSS) t/machine/bitnoise.json --accept t/io/seq001.json -i --accept t/io/seq101.json -fail

# Schema validation tests
VALID_SCHEMA_TESTS = test-echo-valid test-unitindel2-valid
test-echo-valid:
	@$(TEST) bin/$(BOSS) t/expect/bitecho.json -idem

test-unitindel2-valid:
	@$(TEST) bin/$(BOSS) --showparams t/expect/unitindel-unitindel.json -idem

# Schema validation failure tests
INVALID_SCHEMA_TESTS = test-not-json test-no-state test-bad-state test-bad-trans test-bad-weight
test-not-json:
	@$(TEST) bin/$(BOSS) t/invalid/not_json.txt -fail

test-no-state:
	@$(TEST) bin/$(BOSS) t/invalid/no_state.json -fail

test-bad-state:
	@$(TEST) bin/$(BOSS) t/invalid/bad_state.json -fail

test-bad-trans:
	@$(TEST) bin/$(BOSS) t/invalid/bad_trans.json -fail

test-bad-weight:
	@$(TEST) bin/$(BOSS) t/invalid/bad_weight.json -fail

test-cyclic:
	@$(TEST) bin/$(BOSS) t/invalid/cyclic.json -fail

# Non-transducer I/O tests
IO_TESTS = test-fastseq test-seqpair test-seqpairlist test-env test-params test-constraints test-dot
test-fastseq: t/bin/testfastseq
	@$(TEST) t/bin/testfastseq t/tc1/CAA25498.fa t/expect/CAA25498.fa

test-seqpair: t/bin/testseqpair
	@$(TEST) t/bin/testseqpair t/io/tiny.json -idem
	@$(TEST) t/bin/testseqpair t/io/tinypath.json -idem
	@$(TEST) t/bin/testseqpair t/io/tinyfail.json -fail
	@$(TEST) t/bin/testseqpair t/io/tinypathnames.json t/io/tinypath.json
	@$(TEST) t/bin/testseqpair t/io/tinypathonly.json t/expect/tinypathonly.json

test-seqpairlist: t/bin/testseqpairlist
	@$(TEST) t/bin/testseqpairlist t/io/seqpairlist.json -idem

test-env: t/bin/testenv
	@$(TEST) t/bin/testenv t/io/tinypath.json full t/expect/tinypath_full_env.json
	@$(TEST) t/bin/testenv t/io/tinypath.json path t/expect/tinypath_path_env.json
	@$(TEST) t/bin/testenv t/io/smallpath.json path t/expect/smallpath_path_env.json
	@$(TEST) t/bin/testenv t/io/smallpath.json 0 t/expect/smallpath_area0_env.json
	@$(TEST) t/bin/testenv t/io/smallpath.json 1 t/expect/smallpath_area1_env.json
	@$(TEST) t/bin/testenv t/io/smallpath.json 2 t/expect/smallpath_area2_env.json
	@$(TEST) t/bin/testenv t/io/smallpath.json 3 t/expect/smallpath_area3_env.json
	@$(TEST) t/bin/testenv t/io/smallpath.json 4 t/expect/smallpath_area4_env.json
	@$(TEST) t/bin/testenv t/io/smallpath.json 5 t/expect/smallpath_area4_env.json
	@$(TEST) t/bin/testenv t/io/asympath.json 0 t/expect/asympath_area0_env.json
	@$(TEST) t/bin/testenv t/io/asympath.json 1 t/expect/asympath_area1_env.json
	@$(TEST) t/bin/testenv t/io/asympath.json p t/expect/asympath_area0_env.json

test-params: t/bin/testparams
	@$(TEST) t/bin/testparams t/io/params.json -idem

test-constraints: t/bin/testconstraints
	@$(TEST) t/bin/testconstraints t/io/constraints.json -idem

test-dot:
	@$(TEST) bin/$(BOSS) t/machine/bitnoise.json --graphviz t/expect/bitnoise.dot
	@$(TEST) bin/$(BOSS) t/machine/bitnoise.json t/machine/bitnoise.json --graphviz t/expect/bitnoise2.dot

# Symbolic algebra tests
ALGEBRA_TESTS = test-list-params test-deriv-xplusy-x test-deriv-xy-x test-eval-1plus2
test-list-params: t/bin/testlistparams
	@$(TEST) t/bin/testlistparams t/algebra/x_plus_y.json t/expect/xy_params.txt

test-deriv-xplusy-x: t/bin/testderiv
	@$(TEST) t/bin/testderiv t/algebra/x_plus_y.json x t/expect/dxplusy_dx.json

test-deriv-xy-x: t/bin/testderiv
	@$(TEST) t/bin/testderiv t/algebra/x_times_y.json x t/expect/dxy_dx.json

test-eval-1plus2: t/bin/testeval
	@$(TEST) t/bin/testeval t/algebra/x_plus_y.json t/algebra/params.json t/expect/1_plus_2.json

# Dynamic programming tests
DP_TESTS = test-fwd-bitnoise-params-tiny test-back-bitnoise-params-tiny test-fb-bitnoise-params-tiny test-max-bitnoise-params-tiny test-fit-bitnoise-seqpairlist test-funcs test-single-param test-align-stutter-noise test-counts test-counts2
test-fwd-bitnoise-params-tiny: t/bin/testforward
	@$(TEST) t/bin/testforward t/machine/bitnoise.json t/io/params.json t/io/tiny.json t/expect/fwd-bitnoise-params-tiny.json

test-back-bitnoise-params-tiny: t/bin/testbackward
	@$(TEST) t/bin/testbackward t/machine/bitnoise.json t/io/params.json t/io/tiny.json t/expect/back-bitnoise-params-tiny.json

test-fb-bitnoise-params-tiny: t/bin/testcounts
	@$(TEST) t/bin/testcounts t/machine/bitnoise.json t/io/params.json t/io/tiny.json t/expect/fwdback-bitnoise-params-tiny.json

test-max-bitnoise-params-tiny: t/bin/testmaximize
	@$(TEST) t/roundfloats.pl 4 t/bin/testmaximize t/machine/bitnoise.json t/io/params.json t/io/tiny.json t/io/pqcons.json t/expect/max-bitnoise-params-tiny.json

test-fit-bitnoise-seqpairlist:
	@$(TEST) t/roundfloats.pl 4 bin/$(BOSS) t/machine/bitnoise.json -N t/io/pqcons.json -D t/io/seqpairlist.json -T t/expect/fit-bitnoise-seqpairlist.json
	@$(TEST) t/roundfloats.pl 4 bin/$(BOSS) t/machine/bitnoise.json -N t/io/pqcons.json -D t/io/pathlist.json -T t/expect/fit-bitnoise-seqpairlist.json

test-funcs:
	@$(TEST) t/roundfloats.pl 4 bin/$(BOSS) -F t/io/e=0.json t/machine/bitnoise.json t/machine/bsc.json -N t/io/pqcons.json -D t/io/seqpairlist.json -T t/expect/test-funcs.json

test-single-param:
	@$(TEST) t/roundfloats.pl 4 bin/$(BOSS) t/machine/bitnoise.json t/machine/bsc.json -N t/io/econs.json -D t/io/seqpairlist.json -T -F t/io/params.json t/expect/single-param.json

test-align-stutter-noise:
	@$(TEST) bin/$(BOSS) t/machine/bitstutter.json t/machine/bitnoise.json -P t/io/params.json -D t/io/difflen.json -A t/expect/align-stutter-noise-difflen.json

test-counts:
	@$(TEST) bin/$(BOSS) --generate-chars 101 -m t/machine/bitnoise.json --accept-chars 001 -P t/io/params.json -N t/io/pqcons.json -C t/expect/counts.json

test-counts2:
	@$(TEST) bin/$(BOSS) t/machine/bitnoise.json --input-chars 101 --output-chars 001 -P t/io/params.json -N t/io/pqcons.json -C t/expect/counts.json

# Code generation tests
CODEGEN_TESTS = test-101-bitnoise-001 test-101-bitnoise-001-compiled test-101-bitnoise-001-compiled-seq test-101-bitnoise-001-compiled-seq2prof test-101-bitnoise-001-compiled-js test-101-bitnoise-001-compiled-js-seq test-101-bitnoise-001-compiled-js-seq2prof

# C++
t/src/%/prof/test.cpp: t/machine/%.json bin/$(BOSS) src/softplus.h src/getparams.h t/src/testcompiledprof.cpp
	test -e $(dir $@) || mkdir -p $(dir $@)
	bin/$(BOSS) t/machine/$*.json --cpp64 --inseq profile --outseq profile --codegen $(dir $@)
	cp t/src/testcompiledprof.cpp $@

t/src/%/seq/test.cpp: t/machine/%.json bin/$(BOSS) src/softplus.h src/getparams.h t/src/testcompiledseq.cpp
	@test -e $(dir $@) || mkdir -p $(dir $@)
	@bin/$(BOSS) t/machine/$*.json --cpp64 --inseq string --outseq string --codegen $(dir $@)
	@cp t/src/testcompiledseq.cpp $@

t/src/%/seq2prof/test.cpp: t/machine/%.json bin/$(BOSS) src/softplus.h src/getparams.h t/src/testcompiledseq2prof.cpp
	@test -e $(dir $@) || mkdir -p $(dir $@)
	@bin/$(BOSS) t/machine/$*.json --cpp64 --inseq string --outseq profile --codegen $(dir $@)
	@cp t/src/testcompiledseq2prof.cpp $@

t/src/%/fasta/test.cpp: t/machine/%.json bin/$(BOSS) src/softplus.h src/getparams.h t/src/testcompiledfasta.cpp
	@test -e $(dir $@) || mkdir -p $(dir $@)
	@bin/$(BOSS) t/machine/$*.json --cpp64 --inseq string --outseq string --codegen $(dir $@)
	@cp t/src/testcompiledfasta.cpp $@

t/src/%/fasta2strand/test.cpp: t/machine/%.json bin/$(BOSS) src/softplus.h src/getparams.h t/src/testcompiledfasta2strand.cpp
	@test -e $(dir $@) || mkdir -p $(dir $@)
	@bin/$(BOSS) t/machine/$*.json --cpp64 --inseq string --outseq string --codegen $(dir $@)
	@cp t/src/testcompiledfasta2strand.cpp $@

test-101-bitnoise-001:
	@$(TEST) t/roundfloats.pl 4 js/stripnames.js bin/$(BOSS) --generate t/io/seq101.json -m t/machine/bitnoise.json --accept t/io/seq001.json -P t/io/params.json -N t/io/pqcons.json -L t/expect/101-bitnoise-001.json

test-101-bitnoise-001-compiled: t/codegen/bitnoise/prof/test
	@$(TEST) t/roundfloats.pl 4 js/stripnames.js $< t/csv/prof101.csv t/csv/prof001.csv t/io/params.json t/expect/101-bitnoise-001.json

test-101-bitnoise-001-compiled-seq: t/codegen/bitnoise/seq/test
	@$(TEST) t/roundfloats.pl 4 js/stripnames.js $< 101 001 t/io/params.json t/expect/101-bitnoise-001.json

test-101-bitnoise-001-compiled-seq2prof: t/codegen/bitnoise/seq2prof/test
	@$(TEST) t/roundfloats.pl 4 js/stripnames.js $< 101 t/csv/prof001.csv t/io/params.json t/expect/101-bitnoise-001.json

# JavaScript
js/lib/%/prof/test.js: t/machine/%.json bin/$(BOSS) js/lib/softplus.js js/lib/getparams.js js/lib/testcompiledprof.js
	test -e $(dir $@) || mkdir -p $(dir $@)
	bin/$(BOSS) t/machine/$*.json --js --inseq profile --outseq profile --codegen $(dir $@)
	cat js/lib/testcompiledprof.js $(dir $@)computeForward*.js >$@
	cp js/lib/softplus.js js/lib/getparams.js $(dir $@)

js/lib/%/seq/test.js: t/machine/%.json bin/$(BOSS) js/lib/softplus.js js/lib/getparams.js js/lib/testcompiledprof.js
	test -e $(dir $@) || mkdir -p $(dir $@)
	bin/$(BOSS) t/machine/$*.json --js --inseq string --outseq string --codegen $(dir $@)
	cat js/lib/testcompiledprof.js $(dir $@)computeForward*.js >$@
	cp js/lib/softplus.js js/lib/getparams.js $(dir $@)

js/lib/%/seq2prof/test.js: t/machine/%.json bin/$(BOSS) js/lib/softplus.js js/lib/getparams.js js/lib/testcompiledprof.js
	test -e $(dir $@) || mkdir -p $(dir $@)
	bin/$(BOSS) t/machine/$*.json --js --inseq string --outseq profile --codegen $(dir $@)
	cat js/lib/testcompiledprof.js $(dir $@)computeForward*.js >$@
	cp js/lib/softplus.js js/lib/getparams.js $(dir $@)

test-101-bitnoise-001-compiled-js: js/lib/bitnoise/prof/test.js
	@$(TEST) t/roundfloats.pl 4 js/stripnames.js node $< --inprof t/csv/prof101.csv --outprof t/csv/prof001.csv --params t/io/params.json t/expect/101-bitnoise-001.json

test-101-bitnoise-001-compiled-js-seq: js/lib/bitnoise/seq/test.js
	@$(TEST) t/roundfloats.pl 4 js/stripnames.js node $< --inseq 101 --outseq 001 --params t/io/params.json t/expect/101-bitnoise-001.json

test-101-bitnoise-001-compiled-js-seq2prof: js/lib/bitnoise/seq2prof/test.js
	@$(TEST) t/roundfloats.pl 4 js/stripnames.js node $< --inseq 101 --outprof t/csv/prof001.csv --params t/io/params.json t/expect/101-bitnoise-001.json

# Top-level test target
TESTS = $(INVALID_SCHEMA_TESTS) $(VALID_SCHEMA_TESTS) $(COMPOSE_TESTS) $(CONSTRUCT_TESTS) $(INVALID_CONSTRUCT_TESTS) $(IO_TESTS) $(ALGEBRA_TESTS) $(DP_TESTS) $(CODEGEN_TESTS)
TESTLEN = $(shell perl -e 'use List::Util qw(max);print max(map(length,qw($(TESTS))))')
TEST = t/testexpect.pl $@ $(TESTLEN)

test: $(BOSS) $(TESTS)

# Schema validator
ajv:
	npm install ajv-cli

validate-$D-$F:
	ajv -s schema/machine.json -r schema/expr.json -d t/$D/$F.json

# README
README.md: bin/$(BOSS)
	bin/$(BOSS) -h | perl -pe 's/</&lt;/g;s/>/&gt;/g;' | perl -e 'open FILE,"<README.md";while(<FILE>){last if/<pre>/;print}close FILE;print"<pre><code>\n";while(<>){print};print"</code></pre>\n"' >temp.md
	mv temp.md $@
