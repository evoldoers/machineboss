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

MAIN = acidbot

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

# Schema
# valijson doesn't like the URLs, but other validators demand them, so strip them out for xxd
src/schema/%.h: schema/%.nourl.json
	xxd -i $< | sed 's/.nourl//' >$@

schema/%.nourl.json: schema/%.json
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
IO_TESTS = test-seqpair test-params test-constraints
test-seqpair: bin/testseqpair
	@$(TEST) bin/testseqpair t/io/tiny.json -idem

test-params: bin/testparams
	@$(TEST) bin/testparams t/io/params.json -idem

test-constraints: bin/testconstraints
	@$(TEST) bin/testconstraints t/io/constraints.json -idem

# Symbolic algebra tests
ALGEBRA_TESTS = test-deriv-xplusy-x test-deriv-xy-x test-eval-1plus2
test-deriv-xplusy-x: bin/testderiv
	@$(TEST) bin/testderiv t/algebra/x_plus_y.json x t/expect/dxplusy_dx.json

test-deriv-xy-x: bin/testderiv
	@$(TEST) bin/testderiv t/algebra/x_times_y.json x t/expect/dxy_dx.json

test-eval-1plus2: bin/testeval
	@$(TEST) bin/testeval t/algebra/x_plus_y.json t/algebra/params.json t/expect/1_plus_2.json

# Dynamic programming tests
DP_TESTS = test-bitnoise-params-tiny
test-bitnoise-params-tiny: bin/testforward
	@$(TEST) bin/testforward t/machine/bitnoise.json t/io/params.json t/io/tiny.json t/expect/bitnoise-params-tiny.json

# Top-level test target
TESTS = $(INVALID_SCHEMA_TESTS) $(VALID_SCHEMA_TESTS) $(COMPOSE_TESTS) $(IO_TESTS) $(ALGEBRA_TESTS) $(DP_TESTS)
TESTLEN = $(shell perl -e 'use List::Util qw(max);print max(map(length,qw($(TESTS))))')
TEST = t/testexpect.pl $@ $(TESTLEN)

test: $(MAIN) $(TESTS)

# Schema validator
ajv:
	npm install ajv-cli

validate-$D-$F:
	ajv -s schema/machine.json -r schema/expr.json -d t/$D/$F.json
