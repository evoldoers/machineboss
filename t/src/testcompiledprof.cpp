// #include "../../src/softplus.h"
// include a compiled computeForward() as well

#include "../../src/csv.h"
#include "../../src/params.h"
#include <fstream>

int main (int argc, char** argv) {
  CSVProfile inProf, outProf;
  if (argc != 3 && argc != 4) {
    cerr << "Usage: " << argv[0] << " inputProfile.csv outputProfile.csv [params.json]" << endl;
    exit(1);
  }
  ifstream inFile (argv[1]);
  ifstream outFile (argv[2]);
  inProf.read (inFile);
  outProf.read (outFile);
  json params;
  if (argc > 3) {
    ifstream paramFile (argv[3]);
    params = json::parse (paramFile);
  } else
    params = json::object();
  cout << "[" << computeForward (inProf.row, outProf.row, params) << "]" << endl;
  return 0;
}
