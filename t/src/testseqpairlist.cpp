#include "../../src/seqpair.h"

int main (int argc, char** argv) {
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " seqpairlist.json" << endl;
    exit(1);
  }
  SeqPairList seqPairList = JsonLoader<SeqPairList>::fromFile (argv[1]);
  seqPairList.writeJson (cout);
  cout << endl;
  exit(0);
}
