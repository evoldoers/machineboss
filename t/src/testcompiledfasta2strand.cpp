#include <fstream>

#include "csv.h"
#include "params.h"
#include "fastseq.h"
#include "computeForward.h"

using namespace MachineBoss;

string revcomp (const string& fwd) {
  string rev (fwd);
  for (size_t pos = 0; pos < fwd.size(); ++pos) {
    char& rc = rev[pos];
    const char c = fwd[fwd.size() - pos - 1];
    switch (c) {
    case 'A': rc = 'T'; break;
    case 'C': rc = 'G'; break;
    case 'G': rc = 'C'; break;
    case 'T': rc = 'A'; break;
    case 'a': rc = 't'; break;
    case 'c': rc = 'g'; break;
    case 'g': rc = 'c'; break;
    case 't': rc = 'a'; break;
    default:
      rc = 'N';
      break;
    }
  }
  return rev;
}

int main (int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    cerr << "Usage: " << argv[0] << " inputSequences.fa outputDNA.fa [params.json]" << endl;
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
  for (const auto& inSeq: inSeqs) {
    size_t nOut = 0;
    for (const auto& outSeq: outSeqs)
      cout << (nOut++ ? ",\n" : "")
	   << "[\"" << escaped_str(inSeq.name) << "\",\"" << escaped_str(outSeq.name) << "\","
	   << SoftPlus::slow_logsumexp (computeForward (inSeq.seq, outSeq.seq, params),
					computeForward (inSeq.seq, revcomp (outSeq.seq), params))
	   << "]";
  }
  cout << "]\n";
  return 0;
}
