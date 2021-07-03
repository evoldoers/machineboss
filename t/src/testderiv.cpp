#include <fstream>
#include <iostream>
#include "../../src/weight.h"
#include "../../src/schema.h"

using namespace std;
using namespace MachineBoss;

int main (int argc, char** argv) {
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " expr.json param" << endl;
    exit(1);
  }
  json w;
  ifstream in (argv[1]);
  in >> w;
  MachineSchema::validateOrDie ("expr", w);
  WeightExpr d = WeightAlgebra::deriv (WeightAlgebra::fromJson(w), ParamDefs(), string(argv[2]));
  cout << WeightAlgebra::toJsonString(d) << endl;
  exit(0);
}
