#include <fstream>
#include "../../src/counts.h"

int main (int argc, char** argv) {
  if (argc != 4) {
    cerr << "Usage: " << argv[0] << " machine.json params.json seqs.json" << endl;
    exit(1);
  }
  Machine machine = Machine::fromFile (argv[1]);
  Params params = Params::fromFile (argv[2]);
  SeqPair seqpair = SeqPair::fromFile (argv[3]);
  EvaluatedMachine evalMachine (machine, params);
  MachineCounts counts (evalMachine, seqpair);
  counts.writeJson (cout);
  exit(0);
}
