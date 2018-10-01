#include "../../src/fastseq.h"

int main (int argc, char** argv) {
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " sequences.fa" << endl;
    exit(1);
  }
  writeFastaSeqs (cout, readFastSeqs (argv[1]));
  exit(0);
}
