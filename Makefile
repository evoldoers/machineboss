.SECONDARY:

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

# Tests
TEST = t/testexpect.pl

test: $(MAIN) test-echo test-echo2 test-echo-stutter test-stutter2 test-noise2

test-echo:
	@$(TEST) bin/$(MAIN) -v0 t/machine/bitecho.json t/expect/bitecho.json

test-echo2:
	@$(TEST) bin/$(MAIN) -v0 t/machine/bitecho.json t/machine/bitecho.json t/expect/bitecho-bitecho.json

test-echo-stutter:
	@$(TEST) bin/$(MAIN) -v0 t/machine/bitecho.json t/machine/bitstutter.json t/expect/bitecho-bitstutter.json

test-stutter2:
	@$(TEST) bin/$(MAIN) -v0 t/machine/bitstutter.json t/machine/bitstutter.json t/expect/bitstutter-bitstutter.json

test-noise2:
	@$(TEST) bin/$(MAIN) -v0 t/machine/bitnoise.json t/machine/bitnoise.json t/expect/bitnoise-bitnoise.json

