// #include "../../src/softplus.h"
// include a compiled computeForward() as well

#include "../../src/csv.h"
#include "../../src/params.h"
#include <fstream>

int main (int argc, char** argv) {
  CSVProfile inProf, outProf;
  if (argc != 4) {
    cerr << "Usage: " << argv[2] << " inputProfile.csv outputProfile.csv params.json" << endl;
    exit(1);
  }
  ifstream inFile (argv[1]);
  ifstream outFile (argv[2]);
  ifstream paramFile (argv[3]);
  inProf.read (inFile);
  outProf.read (outFile);
  const json params = json::parse (paramFile);
  cout << computeForward (inProf.row, outProf.row, params) << endl;
  return 0;
}
