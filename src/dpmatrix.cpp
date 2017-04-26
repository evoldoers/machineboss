#include <iomanip>
#include "dpmatrix.h"
#include "logger.h"

DPMatrix::DPMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair) :
  machine (machine),
  seqPair (seqPair),
  input (machine.inputTokenizer.tokenize (seqPair.input.seq)),
  output (machine.outputTokenizer.tokenize (seqPair.output.seq)),
  inLen (input.size()),
  outLen (output.size()),
  nStates (machine.nStates())
{
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

string DPMatrix::toJsonString() const {
  ostringstream outs;
  writeJson (outs);
  return outs.str();
}
