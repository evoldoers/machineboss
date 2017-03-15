#include <fstream>
#include "../../src/counts.h"

int main (int argc, char** argv) {
  if (argc != 5) {
    cerr << "Usage: " << argv[0] << " machine.json params.json seqs.json constraints.json" << endl;
    exit(1);
  }
  Machine machine = MachineLoader::fromFile (argv[1]);
  Params params = JsonLoader<ParamAssign>::fromFile (argv[2]);
  SeqPair seqPair = JsonLoader<SeqPair>::fromFile (argv[3]);
  Constraints constraints = JsonLoader<Constraints>::fromFile (argv[4]);
  EvaluatedMachine evalMachine (machine, params);
  MachineCounts counts (evalMachine, seqPair);
  MachineObjective objective (machine, counts, constraints, Params());
  Params optParams = objective.optimize (params);
  optParams.writeJson (cout);
  exit(0);
}
