#include <fstream>
#include <iostream>
#include "../../src/weight.h"
#include "../../src/schema.h"
#include "../../src/util.h"

using namespace std;

int main (int argc, char** argv) {
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " expr.json" << endl;
    exit(1);
  }
  json w;
  ifstream in (argv[1]);
  in >> w;
  MachineSchema::validateOrDie ("expr", w);
  cout << join(WeightAlgebra::params(WeightAlgebra::fromJson(w),ParamDefs()),"\n") << endl;
  exit(0);
}
