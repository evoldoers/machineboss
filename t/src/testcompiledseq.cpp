#include <fstream>

#include "csv.h"
#include "params.h"
#include "computeForward.h"

int main (int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    cerr << "Usage: " << argv[0] << " inputSequence outputSequence [params.json]" << endl;
    exit(1);
  }
  string inStr (argv[1]);
  string outStr (argv[2]);
  json params;
  if (argc > 3) {
    ifstream paramFile (argv[3]);
    params = json::parse (paramFile);
  } else
    params = json::object();
  cout << "[" << computeForward (inStr, outStr, params) << "]" << endl;
  return 0;
}
