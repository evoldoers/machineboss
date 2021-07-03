#include <fstream>

#include "csv.h"
#include "params.h"
#include "computeForward.h"

using namespace MachineBoss;

int main (int argc, char** argv) {
  CSVProfile outProf;
  if (argc != 3 && argc != 4) {
    cerr << "Usage: " << argv[0] << " inputSequence outputProfile.csv [params.json]" << endl;
    exit(1);
  }
  string inStr (argv[1]);
  ifstream outFile (argv[2]);
  outProf.read (outFile);
  json params;
  if (argc > 3) {
    ifstream paramFile (argv[3]);
    params = json::parse (paramFile);
  } else
    params = json::object();
  cout << "[[\"input\",\"output\"," << computeForward (inStr, outProf.row, params) << "]]" << endl;
  return 0;
}
