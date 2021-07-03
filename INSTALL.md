# Installation

## Compiling from C++

On Mac:

~~~~
brew install gsl
brew install boost
brew install openssl
brew install htslib
brew install pkgconfig
make
~~~~

Use `make no-ssl` to build without SSL support (only needed to retrieve models from PFam/DFam).

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
