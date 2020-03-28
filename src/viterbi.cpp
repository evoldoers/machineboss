#include "viterbi.h"
#include "logger.h"

using namespace MachineBoss;

ViterbiMatrix::ViterbiMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair) :
  DPMatrix (machine, seqPair)
{
  fill();
}

ViterbiMatrix::ViterbiMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair, const Envelope& env) :
  DPMatrix (machine, seqPair, env)
{
  fill();
}

void ViterbiMatrix::fill() {
  ProgressLog(plogDP,6);
  plogDP.initProgress ("Filling Viterbi matrix (%lu cells)", nCellsComputed());
  CellIndex nCellsDone = 0;
  for (OutputIndex outPos = 0; outPos <= outLen; ++outPos) {
    const OutputToken outTok = outPos ? output[outPos-1] : OutputTokenizer::emptyToken();
    for (InputIndex inPos = env.inStart[outPos]; inPos < env.inEnd[outPos]; ++inPos, ++nCellsDone) {
      const InputToken inTok = inPos ? input[inPos-1] : InputTokenizer::emptyToken();
      for (StateIndex d = 0; d < nStates; ++d) {
	plogDP.logProgress (nCellsDone / (double) nCellsComputed(), "filled %lu cells", nCellsDone);
	++nCellsDone;
	const EvaluatedMachineState& state = machine.state[d];
	double ll = (inPos || outPos || d) ? -numeric_limits<double>::infinity() : 0;
	if (inPos && outPos)
	  accumulate (ll, state.incoming, inTok, outTok, inPos - 1, outPos - 1, max_reduce);
	if (inPos)
	  accumulate (ll, state.incoming, inTok, OutputTokenizer::emptyToken(), inPos - 1, outPos, max_reduce);
	if (outPos)
	  accumulate (ll, state.incoming, InputTokenizer::emptyToken(), outTok, inPos, outPos - 1, max_reduce);
	accumulate (ll, state.incoming, InputTokenizer::emptyToken(), OutputTokenizer::emptyToken(), inPos, outPos, max_reduce);
	cell(inPos,outPos,d) = ll;
      }
    }
  }
  LogThisAt(8,"Viterbi matrix:" << endl << *this);
}

double ViterbiMatrix::logLike() const {
  return cell (inLen, outLen, machine.endState());
}

MachinePath ViterbiMatrix::path (const Machine& m) const {
  return traceBack (m);
}
