#include <fstream>
#include "../../src/backward.h"

using namespace MachineBoss;

int main (int argc, char** argv) {
  if (argc != 4) {
    cerr << "Usage: " << argv[0] << " machine.json params.json seqs.json" << endl;
    exit(1);
  }
  Machine machine = MachineLoader::fromFile (argv[1]);
  Params params = JsonLoader<ParamAssign>::fromFile (argv[2]);
  SeqPair seqpair = JsonLoader<SeqPair>::fromFile (argv[3]);
  EvaluatedMachine evalMachine (machine, params);
  BackwardMatrix backward (evalMachine, seqpair);
  backward.writeJson (cout);
  exit(0);
}
