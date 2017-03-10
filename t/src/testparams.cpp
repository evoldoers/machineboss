#include "../../src/params.h"

int main (int argc, char** argv) {
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " params.json" << endl;
    exit(1);
  }
  Params params = Params::fromFile (argv[1]);
  params.writeJson (cout);
  exit(0);
}
