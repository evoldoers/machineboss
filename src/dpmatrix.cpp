#include <iomanip>
#include "dpmatrix.h"
#include "logger.h"

vguard<long long> Envelope::offsets() const {
  vguard<long long> result;
  result.reserve (outLen + 2);
  result.push_back (0);
  for (OutputIndex y = 0; y <= outLen; ++y)
    result.push_back (result.back() + inEnd[y] - inStart[y]);
  return result;
}

DPMatrix::DPMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair) :
  machine (machine),
  seqPair (seqPair),
  input (machine.inputTokenizer.tokenize (seqPair.input.seq)),
  output (machine.outputTokenizer.tokenize (seqPair.output.seq)),
  inLen (input.size()),
  outLen (output.size()),
  nStates (machine.nStates()),
  env (inLen, outLen)
{
  alloc();
}

void DPMatrix::alloc() {
  LogThisAt(7,"Creating " << (inLen+1) << "*" << (outLen+1) << "*" << nStates << " matrix" << endl);
  LogThisAt(8,"Machine:" << endl << machine.toJsonString() << endl);
  cellStorage.resize (nCells(), -numeric_limits<double>::infinity());
}

void DPMatrix::writeJson (ostream& outs) const {
  outs << "{" << endl
       << " \"input\": \"" << seqPair.input.name << "\"," << endl
       << " \"output\": \"" << seqPair.output.name << "\"," << endl
       << " \"cell\": [";
  for (InputIndex i = 0; i <= inLen; ++i)
    for (OutputIndex o = 0; o <= outLen; ++o)
      for (StateIndex s = 0; s < nStates; ++s)
	outs << ((i || o || s) ? "," : "") << endl
	     << "  { \"inPos\": " << i << ", \"outPos\": " << o << ", \"state\": " << machine.state[s].name << ", \"logLike\": " << setprecision(5) << cell(i,o,s) << " }";
  outs << endl
       << " ]" << endl
       << "}" << endl;
}

ostream& operator<< (ostream& out, const DPMatrix& m) {
  m.writeJson (out);
  return out;
}
