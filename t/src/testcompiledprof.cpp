#include <fstream>

#include "csv.h"
#include "params.h"
#include "computeForward.h"

using namespace MachineBoss;

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
  cout << "[[\"" << escaped_str(argv[1]) << "\",\"" << escaped_str(argv[2]) << "\"," << computeForward (inProf.row, outProf.row, params) << "]]" << endl;
  return 0;
}
