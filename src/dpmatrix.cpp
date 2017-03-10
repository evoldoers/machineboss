#include <iomanip>
#include "dpmatrix.h"

DPMatrix::DPMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair) :
  machine (machine),
  seqPair (seqPair),
  input (machine.inputTokenizer.tokenize (seqPair.input.seq)),
  output (machine.outputTokenizer.tokenize (seqPair.output.seq)),
  inLen (input.size()),
  outLen (output.size()),
  nStates (machine.nStates())
{
  cellStorage.resize (nCells());
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
