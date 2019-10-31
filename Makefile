.SECONDARY:

MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MAKEFILE_DIR := $(dir $(MAKEFILE_PATH))

# Pseudotargets that control compilation
NO_SSL = $(findstring no-ssl,$(MAKECMDGOALS))
USING_EMSCRIPTEN = $(findstring emscripten,$(MAKECMDGOALS))
IS_32BIT = $(findstring 32bit,$(MAKECMDGOALS))
IS_DEBUG = $(findstring debug,$(MAKECMDGOALS))
IS_UNOPTIMIZED = $(findstring unoptimized,$(MAKECMDGOALS))

# C++ compiler: Emscripten, Clang, or GCC?
ifneq (,$(USING_EMSCRIPTEN))
CPP = emcc
else
# try clang++, fall back to g++
CPP = clang++
ifeq (, $(shell which $(CPP)))
CPP = g++
endif
endif

# If using emscripten, we need to compile gsl-js ourselves
ifneq (,$(USING_EMSCRIPTEN))
GSL_PREFIX = gsl-js
GSL_SOURCE = $(GSL_PREFIX)/gsl-js
GSL_LIB = $(GSL_PREFIX)/lib
GSL_FLAGS = -I$(GSL_SOURCE)
GSL_LIBS =
GSL_SUBDIRS = vector matrix utils linalg blas cblas block err multimin permutation sys poly
GSL_OBJ_FILES = $(foreach dir,$(GSL_SUBDIRS),$(wildcard $(GSL_SOURCE)/$(dir)/*.o))
GSL_DEPS = $(GSL_LIB)
else
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
endif

# If using emscripten, don't link to Boost
BOOST_PROGRAM_OPTIONS = program_options
ifneq (,$(USING_EMSCRIPTEN))
BOOST_FLAGS = -s USE_BOOST_HEADERS=1
BOOST_LIBS = -s USE_BOOST_HEADERS=1
BOOST_OBJ_FILES = $(subst $(BOOST_PROGRAM_OPTIONS)/src/,obj/boost/,$(subst .cpp,.o,$(wildcard $(BOOST_PROGRAM_OPTIONS)/src/*.cpp)))
else
BOOST_OBJ_FILES =
# Try to figure out where Boost is
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
BOOST_LIBS := -L$(BOOST_PREFIX)/lib -lboost_regex -lboost_$(BOOST_PROGRAM_OPTIONS)
endif
endif

# SSL
# Compile with "no-ssl" as a target to skip SSL (use with emscripten)
ifneq (,$(USING_EMSCRIPTEN))
SSL_FLAGS = -DNO_SSL
SSL_LIBS =
else
SSL_FLAGS = -I/usr/local/opt/openssl/include
SSL_LIBS = -lssl -lcrypto -L/usr/local/opt/openssl/lib
endif

# install dir
PREFIX = /usr/local
INSTALL_BIN = $(PREFIX)/bin

# other flags
ifneq (,$(IS_32BIT))
BUILD_FLAGS = -DIS32BIT
else
BUILD_FLAGS =
endif

ALL_FLAGS = $(GSL_FLAGS) $(BOOST_FLAGS) $(BUILD_FLAGS) $(SSL_FLAGS)
ALL_LIBS = $(GSL_LIBS) $(BOOST_LIBS) $(BUILD_LIBS) $(SSL_LIBS)

ifneq (,$(IS_DEBUG))
CPP_FLAGS = -std=c++11 -g -DUSE_VECTOR_GUARDS -DDEBUG
else
ifneq (,$(IS_UNOPTIMIZED))
CPP_FLAGS = -std=c++11 -g
else
CPP_FLAGS = -std=c++11 -g -O3
endif
endif
CPP_FLAGS += $(ALL_FLAGS) -Isrc -Iext -Iext/nlohmann_json
LD_FLAGS = -lstdc++ -lz -lm $(ALL_LIBS)

ifneq (,$(USING_EMSCRIPTEN))
EMCC_FLAGS = -s USE_ZLIB=1 -s EXTRA_EXPORTED_RUNTIME_METHODS="['FS', 'callMain']" -s ALLOW_MEMORY_GROWTH=1 -s EXIT_RUNTIME=1 --pre-js emcc/pre.js
CPP_FLAGS += $(EMCC_FLAGS)
LD_FLAGS += $(EMCC_FLAGS)
endif

# files
CPP_FILES = $(wildcard src/*.cpp)
OBJ_FILES = $(subst src/,obj/,$(subst .cpp,.o,$(CPP_FILES)))

# pwd
PWD = $(shell pwd)

# /bin/sh
SH = /bin/sh

# Targets

BOSS = boss
AUTOWAX = autowax

ifneq (,$(USING_EMSCRIPTEN))
WRAP = node wasm/wrap.js
BOSSTARGET = wasm/boss.js
WRAPBOSS = $(WRAP) $(BOSSTARGET)
TESTSUFFIX = .js
else
WRAP =
BOSSTARGET = bin/$(BOSS)
WRAPBOSS = $(BOSSTARGET)
TESTSUFFIX =
endif

all: $(BOSS)

install: $(BOSS)
	cp bin/$(BOSS) $(INSTALL_BIN)/$(BOSS)

# Main build rules
bin/% wasm/%.js: $(OBJ_FILES) obj/%.o target/%.cpp $(GSL_DEPS) $(BOOST_OBJ_FILES)
	@test -e $(dir $@) || mkdir -p $(dir $@)
	$(CPP) $(LD_FLAGS) -o $@ obj/$*.o $(OBJ_FILES) $(GSL_OBJ_FILES) $(BOOST_OBJ_FILES)

obj/%.o: src/%.cpp $(GSL_DEPS)
	@test -e $(dir $@) || mkdir -p $(dir $@)
	$(CPP) $(CPP_FLAGS) -c -o $@ $<

obj/%.o: target/%.cpp $(GSL_DEPS)
	@test -e $(dir $@) || mkdir -p $(dir $@)
	$(CPP) $(CPP_FLAGS) -c -o $@ $<

obj/boost/%.o: $(BOOST_PROGRAM_OPTIONS) $(BOOST_PROGRAM_OPTIONS)/src/%.cpp
	@test -e $(dir $@) || mkdir -p $(dir $@)
	$(CPP) $(CPP_FLAGS) -c -o $@ $(BOOST_PROGRAM_OPTIONS)/src/$*.cpp

t/bin/%: $(OBJ_FILES) obj/%.o t/src/%.cpp $(GSL_DEPS) $(BOOST_OBJ_FILES)
	@test -e $(dir $@) || mkdir -p $(dir $@)
	$(CPP) $(LD_FLAGS) -o $@$(TESTSUFFIX) obj/$*.o $(OBJ_FILES) $(GSL_OBJ_FILES) $(BOOST_OBJ_FILES)
	mv $@$(TESTSUFFIX) $@

t/codegen/%: $(OBJ_FILES) obj/%.o
	$(MAKE) $(USING_EMSCRIPTEN) `ls $(dir t/src/$*)computeForward*.cpp | perl -pe 's/t\/src/obj/;s/\.cpp/.o/'`
	@test -e $(dir $@) || mkdir -p $(dir $@)
	$(CPP) $(LD_FLAGS) -o $@$(TESTSUFFIX) $^ `ls $(dir t/src/$*)computeForward*.cpp | perl -pe 's/t\/src/obj/;s/\.cpp/.o/'`
	mv $@$(TESTSUFFIX) $@

obj/%.o: t/src/%.cpp
	@test -e $(dir $@) || mkdir -p $(dir $@)
	$(CPP) $(CPP_FLAGS) -c -o $@ $<

$(BOSS): bin/$(BOSS)

$(AUTOWAX): bin/$(AUTOWAX)

emscripten: $(BOSSTARGET)

clean:
	rm -rf bin/$(BOSS) bin/$(AUTOWAX) wasm/$(BOSS).js t/bin/* obj/*

# Fake pseudotargets
debug unoptimized 32bit no-ssl:

# emscripten source files
# gsl-js
$(GSL_LIB):
	mkdir $(GSL_PREFIX)
	cd $(GSL_PREFIX); git clone https://github.com/GSL-for-JS/gsl-js.git
	cd $(GSL_SOURCE); emconfigure ./configure --prefix=$(abspath $(CURDIR)/$(GSL_PREFIX)); emmake make -k install

# boost::program_options
$(BOOST_PROGRAM_OPTIONS):
	git clone https://github.com/boostorg/program_options.git

# Schemas & presets
# The relevant pseudotargets are generate-schemas and generate-presets (biomake required)
# Get biomake here: https://github.com/evoldoers/biomake
generate-$(CATEGORY)s: $(patsubst $(CATEGORY)/%.json,src/$(CATEGORY)/%.h,$(wildcard $(CATEGORY)/*.json))
	touch src/$(CATEGORY).cpp

src/preset/$(FILE).h: preset/$(FILE).json
	xxd -i $< >$@

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
CONSTRUCT_TESTS = test-generator test-recognizer test-wild-generator test-wild-recognizer test-union test-intersection test-brackets test-kleene test-loop test-noisy-loop test-concat test-eliminate test-reverse test-revcomp test-transpose test-weight test-shorthand test-hmmer test-csv test-csv-tiny test-csv-tiny-fail test-csv-tiny-empty test-nanopore test-nanopore-prefix test-nanopore-decode
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

test-reverse:
	@$(TEST) $(WRAPBOSS) --generate-json t/io/seq001.json -e t/expect/generator001-reversed.json

test-revcomp:
	@$(TEST) $(WRAPBOSS) --generate-json t/io/seqAGC.json -r t/expect/generatorAGC-revcomp.json

test-transpose:
	@$(TEST) $(WRAPBOSS) --generate-json t/io/seq001.json -t t/expect/recognizer001.json

test-weight:
	@$(TEST) $(WRAPBOSS) -w p t/expect/null-p.json
	@$(TEST) $(WRAPBOSS) -w 2 t/expect/null-2.json
	@$(TEST) $(WRAPBOSS) -w .5 t/expect/null-0.5.json
	@$(TEST) $(WRAPBOSS) -w '{"*":["p","q"]}' t/expect/null-pq.json
	@$(TEST) $(WRAPBOSS) -w '{"*":[1,2]}' t/expect/null-2.json
	@$(TEST) $(WRAPBOSS) -w '{"/":[1,2]}' t/expect/null-1div2.json
	@$(TEST) $(WRAPBOSS) --recognize-wild ACGT --weight-input '"p$$"' --reciprocal t/expect/null-weight-recip.json

test-shorthand:
	@$(TEST) $(WRAPBOSS) '(' t/machine/bitnoise.json '>>' 101 ')' '&&' '>>' 001 '.' '>>' AGC '#' x t/expect/shorthand.json

test-hmmer:
	@$(TEST) t/roundfloats.pl 3 $(WRAPBOSS) --hmmer t/hmmer/fn3.hmm t/expect/fn3.json

test-csv:
	@$(TEST) $(WRAPBOSS) --generate-csv t/csv/test.csv t/expect/csvtest.json
	@$(TEST) $(WRAPBOSS) --generate-csv t/csv/test.csv --cond-norm t/expect/normcsvtest.json
	@$(TEST) $(WRAPBOSS) --recognize-csv t/csv/test.csv --transpose t/expect/csvtest.json
	@$(TEST) $(WRAPBOSS) --recognize-csv t/csv/test.csv --transpose --joint-norm t/expect/normcsvtest.json

test-csv-tiny:
	@$(TEST) js/stripnames.js $(WRAPBOSS) -L --generate-json t/io/tiny_uc.json --recognize-csv t/csv/tiny_uc.csv t/expect/tiny_uc.json

test-csv-tiny-fail:
	@$(TEST) js/stripnames.js $(WRAPBOSS) -L --generate-json t/io/tiny_lc.json --recognize-csv t/csv/tiny_uc.csv -fail

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
	@$(TEST) $(WRAPBOSS) t/machine/bitnoise.json --begin --recognize-json t/io/seq001.json -i --recognize-json t/io/seq101.json --end -fail

# Schema validation tests
VALID_SCHEMA_TESTS = test-echo-valid test-unitindel2-valid
test-echo-valid:
	@$(TEST) $(WRAPBOSS) t/expect/bitecho.json -idem

test-unitindel2-valid:
	@$(TEST) $(WRAPBOSS) --show-params t/expect/unitindel-unitindel.json -idem

# Schema validation failure tests
INVALID_SCHEMA_TESTS = test-not-json test-no-state test-bad-state test-bad-trans test-bad-weight
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
IO_TESTS = test-fastseq test-seqpair test-seqpairlist test-env test-params test-constraints test-dot
test-fastseq: t/bin/testfastseq
	@$(WRAPTEST) t/bin/testfastseq t/tc1/CAA25498.fa t/expect/CAA25498.fa

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
	@$(TEST) t/roundfloats.pl 4 $(WRAP) t/bin/testmaximize t/machine/bitnoise.json t/io/params.json t/io/tiny.json t/io/pqcons.json t/expect/max-bitnoise-params-tiny.json

test-fit-bitnoise-seqpairlist:
	@$(TEST) t/roundfloats.pl 4 $(WRAPBOSS) t/machine/bitnoise.json -N t/io/pqcons.json -D t/io/seqpairlist.json -T t/expect/fit-bitnoise-seqpairlist.json
	@$(TEST) t/roundfloats.pl 4 $(WRAPBOSS) t/machine/bitnoise.json -N t/io/pqcons.json -D t/io/pathlist.json -T t/expect/fit-bitnoise-seqpairlist.json

test-funcs:
	@$(TEST) t/roundfloats.pl 4 $(WRAPBOSS) -F t/io/e=0.json t/machine/bitnoise.json t/machine/bsc.json -N t/io/pqcons.json -D t/io/seqpairlist.json -T t/expect/test-funcs.json

test-single-param:
	@$(TEST) t/roundfloats.pl 4 $(WRAPBOSS) t/machine/bitnoise.json t/machine/bsc.json -N t/io/econs.json -D t/io/seqpairlist.json -T -F t/io/params.json t/expect/single-param.json

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
	@$(TEST) t/roundfloats.pl 1 $(WRAPBOSS) --generate-uniform ACGT --concat --generate-chars CATCAG --concat --begin --generate-one A --count-copies n --end --concat --generate-chars TATA --concat --generate-uniform ACGT --recognize-csv t/csv/nanopore_test.csv -C t/expect/count9.json
	@$(TEST) t/roundfloats.pl 1 $(WRAPBOSS) --generate-uniform ACGT --concat --generate-chars CAT --concat --begin --generate-one T --count-copies n --end --concat --generate-chars GG --concat --generate-uniform ACGT --recognize-csv t/csv/nanopore_test.csv -C t/expect/count4.json

# Code generation tests
CODEGEN_TESTS = test-101-bitnoise-001 test-101-bitnoise-001-compiled test-101-bitnoise-001-compiled-seq test-101-bitnoise-001-compiled-seq2prof test-101-bitnoise-001-compiled-js test-101-bitnoise-001-compiled-js-seq test-101-bitnoise-001-compiled-js-seq2prof

# C++
t/src/%/prof/test.cpp: t/machine/%.json $(BOSSTARGET) src/softplus.h src/getparams.h t/src/testcompiledprof.cpp
	test -e $(dir $@) || mkdir -p $(dir $@)
	$(WRAPBOSS) t/machine/$*.json --cpp64 --inseq profile --outseq profile --codegen $(dir $@)
	cp t/src/testcompiledprof.cpp $@

t/src/%/seq/test.cpp: t/machine/%.json $(BOSSTARGET) src/softplus.h src/getparams.h t/src/testcompiledseq.cpp
	@test -e $(dir $@) || mkdir -p $(dir $@)
	@$(WRAPBOSS) t/machine/$*.json --cpp64 --inseq string --outseq string --codegen $(dir $@)
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
	@$(TEST) t/roundfloats.pl 4 js/stripnames.js $(WRAPBOSS) --generate-json t/io/seq101.json -m t/machine/bitnoise.json --recognize-json t/io/seq001.json -P t/io/params.json -N t/io/pqcons.json -L t/expect/101-bitnoise-001.json

test-101-bitnoise-001-compiled: t/codegen/bitnoise/prof/test
	@$(TEST) t/roundfloats.pl 4 js/stripnames.js $(WRAP) $< t/csv/prof101.csv t/csv/prof001.csv t/io/params.json t/expect/101-bitnoise-001.json

test-101-bitnoise-001-compiled-seq: t/codegen/bitnoise/seq/test
	@$(TEST) t/roundfloats.pl 4 js/stripnames.js $(WRAP) $< 101 001 t/io/params.json t/expect/101-bitnoise-001.json

test-101-bitnoise-001-compiled-seq2prof: t/codegen/bitnoise/seq2prof/test
	@$(TEST) t/roundfloats.pl 4 js/stripnames.js $(WRAP) $< 101 t/csv/prof001.csv t/io/params.json t/expect/101-bitnoise-001.json

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
	@$(TEST) t/roundfloats.pl 4 js/stripnames.js $(WRAP) $< --inprof t/csv/prof101.csv --outprof t/csv/prof001.csv --params t/io/params.json t/expect/101-bitnoise-001.json

test-101-bitnoise-001-compiled-js-seq: js/lib/bitnoise/seq/test.js
	@$(TEST) t/roundfloats.pl 4 js/stripnames.js $(WRAP) $< --inseq 101 --outseq 001 --params t/io/params.json t/expect/101-bitnoise-001.json

test-101-bitnoise-001-compiled-js-seq2prof: js/lib/bitnoise/seq2prof/test.js
	@$(TEST) t/roundfloats.pl 4 js/stripnames.js $(WRAP) $< --inseq 101 --outprof t/csv/prof001.csv --params t/io/params.json t/expect/101-bitnoise-001.json

# Decoding
DECODE_TESTS = test-decode-bitecho-101 test-bintern

test-decode-bitecho-101:
	@$(TEST) $(WRAPBOSS) t/machine/bitecho.json --recognize-chars 101 --prefix-decode t/expect/decode-bitecho-101.json

test-bintern:
	@$(TEST) $(WRAPBOSS) --generate-chars 101 t/machine/bintern.json --prefix-encode t/expect/encode-g101-bintern.json
	@$(TEST) $(WRAPBOSS) --input-chars 101 t/machine/bintern.json --prefix-encode t/expect/encode-i101-bintern.json
	@$(TEST) $(WRAPBOSS) t/machine/bintern.json --recognize-chars 12222 --prefix-decode t/expect/decode-a12222-bintern.json
	@$(TEST) $(WRAPBOSS) t/machine/bintern.json --output-chars 12222 --prefix-decode t/expect/decode-o12222-bintern.json
	@$(TEST) $(WRAPBOSS) t/machine/bintern.json --recognize-chars 12222 --beam-decode t/expect/decode-a12222-bintern.json
	@$(TEST) $(WRAPBOSS) t/machine/bintern.json --output-chars 12222 --beam-decode t/expect/decode-o12222-bintern.json

# Top-level test target
TESTS = $(INVALID_SCHEMA_TESTS) $(VALID_SCHEMA_TESTS) $(COMPOSE_TESTS) $(CONSTRUCT_TESTS) $(INVALID_CONSTRUCT_TESTS) $(IO_TESTS) $(ALGEBRA_TESTS) $(DP_TESTS) $(CODEGEN_TESTS) $(DECODE_TESTS)
TESTLEN = $(shell perl -e 'use List::Util qw(max);print max(map(length,qw($(TESTS))))')

TEST = t/testexpect.pl $@ $(TESTLEN)
WRAPTEST = $(TEST) $(WRAP)

test: $(BOSSTARGET) $(TESTS)

# Schema validator
ajv:
	npm install ajv-cli

validate-$D-$F:
	ajv -s schema/machine.json -r schema/expr.json -d t/$D/$F.json

# README
README.md: $(BOSSTARGET)
	$(WRAPBOSS) -h | perl -pe 's/</&lt;/g;s/>/&gt;/g;' | perl -e 'open FILE,"<README.md";while(<FILE>){last if/<pre>/;print}close FILE;print"<pre><code>\n";while(<>){print};print"</code></pre>\n"' >temp.md
	mv temp.md $@
