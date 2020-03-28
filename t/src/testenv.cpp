#include "../../src/seqpair.h"

using namespace MachineBoss;

int main (int argc, char** argv) {
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " seqpair.json [full|path|<width>]" << endl;
    exit(1);
  }
  SeqPair seqPair = JsonLoader<SeqPair>::fromFile (argv[1]);
  Envelope env;
  if (argv[2][0] == 'f')
    env.initFull (seqPair);
  else if (argv[2][0] == 'p')
    env.initPath (seqPair.alignment);
  else {
    int width = atoi (argv[2]);
    env.initPathArea (seqPair.alignment, width);
  }
  env.writeJson (cout);
  cout << endl;
  exit(0);
}
