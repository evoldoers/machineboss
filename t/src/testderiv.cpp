#include <fstream>
#include "../../src/weight.h"
#include "../../src/schema.h"

int main (int argc, char** argv) {
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " expr.json param" << endl;
    exit(1);
  }
  json w;
  ifstream in (argv[1]);
  in >> w;
  MachineSchema::validateOrDie ("expr", w);
  WeightExpr d = WeightAlgebra::deriv (w, ParamDefs(), string(argv[2]));
  cout << d << endl;
  exit(0);
}
