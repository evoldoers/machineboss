#include "../../src/constraints.h"

int main (int argc, char** argv) {
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " constraints.json" << endl;
    exit(1);
  }
  Constraints cons = Constraints::fromFile (argv[1]);
  cons.writeJson (cout);
  exit(0);
}
