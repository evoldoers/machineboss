#include "../../src/seqpair.h"

using namespace MachineBoss;

int main (int argc, char** argv) {
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " seqpair.json" << endl;
    exit(1);
  }
  SeqPair seqpair = JsonLoader<SeqPair>::fromFile (argv[1]);
  seqpair.writeJson (cout);
  cout << endl;
  exit(0);
}
