#include <fstream>
#include "../../src/counts.h"

int main (int argc, char** argv) {
  if (argc != 5) {
    cerr << "Usage: " << argv[0] << " machine.json params.json seqs.json constraints.json" << endl;
    exit(1);
  }
  Machine machine = Machine::fromFile (argv[1]);
  Params params = Params::fromFile (argv[2]);
  SeqPair seqPair = SeqPair::fromFile (argv[3]);
  Constraints constraints = Constraints::fromFile (argv[4]);
  EvaluatedMachine evalMachine (machine, params);
  MachineCounts counts (evalMachine, seqPair);
  MachineLagrangian lagrangian (machine, counts, constraints);
  Params optParams = lagrangian.optimize (params);
  optParams.writeJson (cout);
  exit(0);
}
