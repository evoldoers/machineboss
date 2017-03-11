#include <fstream>
#include "../../src/forward.h"

int main (int argc, char** argv) {
  if (argc != 4) {
    cerr << "Usage: " << argv[0] << " machine.json params.json seqs.json" << endl;
    exit(1);
  }
  Machine machine = MachineLoader::fromFile (argv[1]);
  Params params = Params::fromFile (argv[2]);
  SeqPair seqpair = SeqPair::fromFile (argv[3]);
  EvaluatedMachine evalMachine (machine, params);
  ForwardMatrix forward (evalMachine, seqpair);
  forward.writeJson (cout);
  exit(0);
}
