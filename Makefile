.SECONDARY:

MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MAKEFILE_DIR := $(dir $(MAKEFILE_PATH))

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
CPPFLAGS = -std=c++11 -g -DUSE_VECTOR_GUARDS -DDEBUG $(BOOSTFLAGS)
else
CPPFLAGS = -std=c++11 -g -O3 $(BOOSTFLAGS)
endif
CPPFLAGS += -Iext -Iext/nlohmann_json
LIBFLAGS = -lstdc++ -lz $(BOOSTLIBS)

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
bin/%: $(OBJFILES) obj/%.o
	@test -e bin || mkdir bin
	$(CPP) $(LIBFLAGS) -o $@ obj/$*.o $(OBJFILES)

obj/%.o: src/%.cpp
	@test -e obj || mkdir obj
	$(CPP) $(CPPFLAGS) -c -o $@ $<

obj/%.o: target/%.cpp
	@test -e obj || mkdir obj
	$(CPP) $(CPPFLAGS) -c -o $@ $<

$(MAIN): bin/$(MAIN)

clean:
	rm -rf bin/$(MAIN) obj/*

debug: all

# Schema
# valijson doesn't like the URLs, but other validators demand them, so strip them out for xxd
src/schema/%.h: schema/%.nourl.json
	xxd -i $< | sed 's/.nourl//' >$@

schema/%.nourl.json: schema/%.json
	grep -v http $< >$@

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

# Top-level test target
TESTS = $(INVALID_SCHEMA_TESTS) $(VALID_SCHEMA_TESTS) $(COMPOSE_TESTS)
TESTLEN = $(shell perl -e 'use List::Util qw(max);print max(map(length,qw($(TESTS))))')
TEST = t/testexpect.pl $@ $(TESTLEN)

test: $(MAIN) $(TESTS)
