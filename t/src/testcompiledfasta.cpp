#include <fstream>

#include "csv.h"
#include "params.h"
#include "computeForward.h"

using namespace MachineBoss;

int main (int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    cerr << "Usage: " << argv[0] << " inputSequences.fa outputSequences.fa [params.json]" << endl;
    exit(1);
  }
  const vguard<FastSeq> inSeqs = readFastSeqs (argv[1]);
  const vguard<FastSeq> outSeqs = readFastSeqs (argv[2]);
  json params;
  if (argc > 3) {
    ifstream paramFile (argv[3]);
    params = json::parse (paramFile);
  } else
    params = json::object();
  cout << "[";
  size_t n = 0;
  for (const auto& inSeq: inSeqs) {
    for (const auto& outSeq: outSeqs)
      cout << (n++ ? ",\n" : "")
	   << "[\"" << escaped_str(inSeq.name) << "\",\"" << escaped_str(outSeq.name) << "\","
	   << computeForward (inSeq.seq, outSeq.seq, params) << "]";
  }
  cout << "]\n";
  return 0;
}
