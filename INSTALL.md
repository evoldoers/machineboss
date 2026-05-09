# Installation

## Compiling from C++

On Mac:

~~~~
brew install gsl
brew install pkgconfig
make
~~~~

(htslib is *not* required — we use only `kseq.h`, which is vendored as
a single header in `ext/htslib/`. zlib is also linked, for transparent
reading of gzipped FASTA, but it ships pre-installed on macOS and most
Linux distributions and so is not listed above.)

### Testing

~~~~
npm install
make test
~~~~

## Installing via npm

The software is compiled to WebAssembly as [machineboss](https://www.npmjs.com/package/machineboss) on npm

~~~~
npm install machineboss
~~~~
