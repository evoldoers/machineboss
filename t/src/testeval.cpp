#include <fstream>
#include "../../src/params.h"
#include "../../src/schema.h"

int main (int argc, char** argv) {
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " expr.json params.json" << endl;
    exit(1);
  }
  json w;
  ifstream in (argv[1]);
  in >> w;
  MachineSchema::validateOrDie ("expr", w);
  Params p = JsonLoader<ParamAssign>::fromFile (argv[2]);
  cout << WeightAlgebra::eval(WeightAlgebra::fromJson(w),p.defs) << endl;
  exit(0);
}
