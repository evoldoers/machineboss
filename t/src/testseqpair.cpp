#include "../../src/seqpair.h"

int main (int argc, char** argv) {
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " seqpair.json" << endl;
    exit(1);
  }
  SeqPair seqpair = SeqPair::fromFile (argv[1]);
  seqpair.writeJson (cout);
  cout << endl;
  exit(0);
}
