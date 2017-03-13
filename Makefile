.SECONDARY:

MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MAKEFILE_DIR := $(dir $(MAKEFILE_PATH))

# try to figure out where GSL is
# autoconf would be better but we just need a quick hack for now :)
# Thanks to Torsten Seemann for gsl-config and pkg-config formulae
GSLPREFIX = $(shell gsl-config --prefix)
ifeq (,$(wildcard $(GSLPREFIX)/include/gsl/gsl_sf.h))
GSLPREFIX = /usr
ifeq (,$(wildcard $(GSLPREFIX)/include/gsl/gsl_sf.h))
GSLPREFIX = /usr/local
endif
endif

GSLFLAGS = $(shell pkg-config --cflags gsl)
ifeq (, $(GSLFLAGS))
GSLFLAGS = -I$(GSLPREFIX)/include
endif

GSLLIBS = $(shell pkg-config --libs gsl)
ifeq (, $(GSLLIBS))
GSLLIBS = -L$(GSLPREFIX)/lib -lgsl -lgslcblas -lm
endif

# NB pkg-config support for Boost is lacking; see https://svn.boost.org/trac/boost/ticket/1094
BOOSTPREFIX = /usr
ifeq (,$(wildcard $(BOOSTPREFIX)/include/boost/regex.h))
BOOSTPREFIX = /usr/local
ifeq (,$(wildcard $(BOOSTPREFIX)/include/boost/regex.h))
BOOSTPREFIX =
endif
endif

BOOSTFLAGS =
BOOSTLIBS =
ifneq (,$(BOOSTPREFIX))
BOOSTFLAGS := -I$(BOOSTPREFIX)/include
BOOSTLIBS := -L$(BOOSTPREFIX)/lib -lboost_regex -lboost_program_options
endif

# install dir
PREFIX = /usr/local

# other flags
ifneq (,$(findstring debug,$(MAKECMDGOALS)))
CPPFLAGS = -std=c++11 -g -DUSE_VECTOR_GUARDS -DDEBUG $(GSLFLAGS) $(BOOSTFLAGS)
else
CPPFLAGS = -std=c++11 -g -O3 $(GSLFLAGS) $(BOOSTFLAGS)
endif
CPPFLAGS += -Iext -Iext/nlohmann_json
LIBFLAGS = -lstdc++ -lz $(GSLLIBS) $(BOOSTLIBS)

CPPFILES = $(wildcard src/*.cpp)
OBJFILES = $(subst src/,obj/,$(subst .cpp,.o,$(CPPFILES)))

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

MAIN = bossmachine

all: $(MAIN)

# Main build rules
bin/%: $(OBJFILES) obj/%.o target/%.cpp
	@test -e bin || mkdir bin
	$(CPP) $(LIBFLAGS) -o $@ obj/$*.o $(OBJFILES)

obj/%.o: src/%.cpp
	@test -e obj || mkdir obj
	$(CPP) $(CPPFLAGS) -c -o $@ $<

obj/%.o: target/%.cpp
	@test -e obj || mkdir obj
	$(CPP) $(CPPFLAGS) -c -o $@ $<

bin/%: $(OBJFILES) obj/%.o t/src/%.cpp
	@test -e bin || mkdir bin
	@$(CPP) $(LIBFLAGS) -o $@ obj/$*.o $(OBJFILES)

obj/%.o: t/src/%.cpp
	@test -e obj || mkdir obj
	@$(CPP) $(CPPFLAGS) -c -o $@ $<

$(MAIN): bin/$(MAIN)

clean:
	rm -rf bin/$(MAIN) obj/*

debug: all

# Schemas & presets
# Targets are generate-schemas and generate-presets (biomake required)
# Get biomake here: https://github.com/evoldoers/biomake
generate-$(CATEGORY)s: $(patsubst $(CATEGORY)/%.json,src/$(CATEGORY)/%.h,$(wildcard $(CATEGORY)/*.json))
	touch src/$(CATEGORY).cpp

src/preset/$(FILE).h: preset/$(FILE).json
	xxd -i $< >$@

# valijson doesn't like the URLs, but other schema validators demand them, so strip them out for xxd
src/schema/$(FILE).h: schema/$(FILE).json.nourl
	xxd -i $< | sed 's/.nourl//' >$@

schema/$(FILE).json.nourl: schema/$(FILE).json
	grep -v '"id": "http' $< >$@

# Transducer composition tests
COMPOSE_TESTS = test-echo test-echo2 test-echo-stutter test-stutter2 test-noise2 test-unitindel2
test-echo:
	@$(TEST) bin/$(MAIN) t/machine/bitecho.json t/expect/bitecho.json

test-echo2:
	@$(TEST) bin/$(MAIN) t/machine/bitecho.json t/machine/bitecho.json t/expect/bitecho-bitecho.json

test-echo-stutter:
	@$(TEST) bin/$(MAIN) t/machine/bitecho.json t/machine/bitstutter.json t/expect/bitecho-bitstutter.json

test-stutter2:
	@$(TEST) bin/$(MAIN) t/machine/bitstutter.json t/machine/bitstutter.json t/expect/bitstutter-bitstutter.json

test-noise2:
	@$(TEST) bin/$(MAIN) t/machine/bitnoise.json t/machine/bitnoise.json t/expect/bitnoise-bitnoise.json

test-unitindel2:
	@$(TEST) bin/$(MAIN) t/machine/unitindel.json t/machine/unitindel.json t/expect/unitindel-unitindel.json

# Transducer construction tests
CONSTRUCT_TESTS = test-generator test-acceptor test-union test-intersection test-brackets test-kleene test-concat test-reverse test-revcomp test-flip test-null
test-generator:
	@$(TEST) bin/$(MAIN) -g t/io/seq101.json t/expect/generator101.json

test-acceptor:
	@$(TEST) bin/$(MAIN) -a t/io/seq001.json t/expect/acceptor001.json

test-union:
	@$(TEST) bin/$(MAIN) -g t/io/seq001.json t/expect/generator101.json -W p t/expect/generate-101-or-001.json

test-intersection:
	@$(TEST) bin/$(MAIN) t/machine/bitnoise.json -a t/io/seq001.json -M -a t/io/seq101.json -I t/expect/noise-001-and-101.json

test-brackets:
	@$(TEST) bin/$(MAIN) --begin t/machine/bitnoise.json -a t/io/seq001.json --end -i -a t/io/seq101.json t/expect/noise-001-and-101.json

test-kleene:
	@$(TEST) bin/$(MAIN) -g t/io/seq001.json -l q t/expect/generate-multiple-001.json

test-concat:
	@$(TEST) bin/$(MAIN) -g t/io/seq001.json -c t/expect/generator101.json t/expect/concat-001-101.json

test-reverse:
	@$(TEST) bin/$(MAIN) -g t/io/seq001.json -e t/expect/generator001-reversed.json

test-revcomp:
	@$(TEST) bin/$(MAIN) -g t/io/seqAGC.json -r t/expect/generatorAGC-revcomp.json

test-flip:
	@$(TEST) bin/$(MAIN) -g t/io/seq001.json -f t/expect/acceptor001.json

test-null:
	@$(TEST) bin/$(MAIN) -n t/expect/null.json

# Invalid transducer construction tests
INVALID_CONSTRUCT_TESTS = test-unmatched-begin test-unmatched-end test-empty-brackets test-impossible-intersect
test-unmatched-begin:
	@$(TEST) bin/$(MAIN) --begin -fail

test-unmatched-end:
	@$(TEST) bin/$(MAIN) --end -fail

test-empty-brackets:
	@$(TEST) bin/$(MAIN) --begin --end -fail

test-impossible-intersect:
	@$(TEST) bin/$(MAIN) t/machine/bitnoise.json -a t/io/seq001.json -i -a t/io/seq101.json -fail

# Schema validation tests
VALID_SCHEMA_TESTS = test-echo-valid test-unitindel2-valid
test-echo-valid:
	@$(TEST) bin/$(MAIN) t/expect/bitecho.json -idem

test-unitindel2-valid:
	@$(TEST) bin/$(MAIN) t/expect/unitindel-unitindel.json -idem

# Schema validation failure tests
INVALID_SCHEMA_TESTS = test-not-json test-no-state test-bad-state test-bad-trans test-bad-weight
test-not-json:
	@$(TEST) bin/$(MAIN) t/invalid/not_json.txt -fail

test-no-state:
	@$(TEST) bin/$(MAIN) t/invalid/no_state.json -fail

test-bad-state:
	@$(TEST) bin/$(MAIN) t/invalid/bad_state.json -fail

test-bad-trans:
	@$(TEST) bin/$(MAIN) t/invalid/bad_trans.json -fail

test-bad-weight:
	@$(TEST) bin/$(MAIN) t/invalid/bad_weight.json -fail

# Non-transducer I/O tests
IO_TESTS = test-seqpair test-seqpairlist test-params test-constraints
test-seqpair: bin/testseqpair
	@$(TEST) bin/testseqpair t/io/tiny.json -idem

test-seqpairlist: bin/testseqpairlist
	@$(TEST) bin/testseqpairlist t/io/seqpairlist.json -idem

test-params: bin/testparams
	@$(TEST) bin/testparams t/io/params.json -idem

test-constraints: bin/testconstraints
	@$(TEST) bin/testconstraints t/io/constraints.json -idem

# Symbolic algebra tests
ALGEBRA_TESTS = test-list-params test-deriv-xplusy-x test-deriv-xy-x test-eval-1plus2
test-list-params: bin/testlistparams
	@$(TEST) bin/testlistparams t/algebra/x_plus_y.json t/expect/xy_params.txt

test-deriv-xplusy-x: bin/testderiv
	@$(TEST) bin/testderiv t/algebra/x_plus_y.json x t/expect/dxplusy_dx.json

test-deriv-xy-x: bin/testderiv
	@$(TEST) bin/testderiv t/algebra/x_times_y.json x t/expect/dxy_dx.json

test-eval-1plus2: bin/testeval
	@$(TEST) bin/testeval t/algebra/x_plus_y.json t/algebra/params.json t/expect/1_plus_2.json

# Dynamic programming tests
DP_TESTS = test-fwd-bitnoise-params-tiny test-back-bitnoise-params-tiny test-fb-bitnoise-params-tiny test-max-bitnoise-params-tiny test-fit-bitnoise-seqpairlist test-align-stutter-noise
test-fwd-bitnoise-params-tiny: bin/testforward
	@$(TEST) bin/testforward t/machine/bitnoise.json t/io/params.json t/io/tiny.json t/expect/fwd-bitnoise-params-tiny.json

test-back-bitnoise-params-tiny: bin/testbackward
	@$(TEST) bin/testbackward t/machine/bitnoise.json t/io/params.json t/io/tiny.json t/expect/back-bitnoise-params-tiny.json

test-fb-bitnoise-params-tiny: bin/testcounts
	@$(TEST) bin/testcounts t/machine/bitnoise.json t/io/params.json t/io/tiny.json t/expect/fwdback-bitnoise-params-tiny.json

test-max-bitnoise-params-tiny: bin/testmaximize
	@$(TEST) t/roundfloats.pl 4 bin/testmaximize t/machine/bitnoise.json t/io/params.json t/io/tiny.json t/io/pqcons.json t/expect/max-bitnoise-params-tiny.json

test-fit-bitnoise-seqpairlist:
	@$(TEST) t/roundfloats.pl 4 bin/$(MAIN) t/machine/bitnoise.json -C t/io/pqcons.json -D t/io/seqpairlist.json -F t/expect/fit-bitnoise-seqpairlist.json

test-align-stutter-noise:
	@$(TEST) bin/$(MAIN) t/machine/bitstutter.json t/machine/bitnoise.json -P t/io/params.json -D t/io/difflen.json -A t/expect/align-stutter-noise-difflen.json

# Top-level test target
TESTS = $(INVALID_SCHEMA_TESTS) $(VALID_SCHEMA_TESTS) $(COMPOSE_TESTS) $(CONSTRUCT_TESTS) $(INVALID_CONSTRUCT_TESTS) $(IO_TESTS) $(ALGEBRA_TESTS) $(DP_TESTS)
TESTLEN = $(shell perl -e 'use List::Util qw(max);print max(map(length,qw($(TESTS))))')
TEST = t/testexpect.pl $@ $(TESTLEN)

test: $(MAIN) $(TESTS)

# Schema validator
ajv:
	npm install ajv-cli

validate-$D-$F:
	ajv -s schema/machine.json -r schema/expr.json -d t/$D/$F.json

# README
README.md: bin/$(MAIN)
	bin/$(MAIN) -h | perl -pe 's/</&lt;/g;s/>/&gt;/g;' | perl -e 'open FILE,"<README.md";while(<FILE>){last if/<pre>/;print}close FILE;print"<pre><code>\n";while(<>){print};print"</code></pre>\n"' >temp.md
	mv temp.md $@
