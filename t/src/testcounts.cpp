#include <fstream>
#include "../../src/counts.h"

int main (int argc, char** argv) {
  if (argc != 4) {
    cerr << "Usage: " << argv[0] << " machine.json params.json seqs.json" << endl;
    exit(1);
  }
  Machine machine = MachineLoader::fromFile (argv[1]);
  Params params = JsonLoader<ParamAssign>::fromFile (argv[2]);
  SeqPair seqPair = JsonLoader<SeqPair>::fromFile (argv[3]);
  EvaluatedMachine evalMachine (machine, params);
  MachineCounts counts (evalMachine, seqPair);
  counts.writeJson (cout);
  exit(0);
}
