#include "forward.h"
#include "logger.h"

ForwardMatrix::ForwardMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair) :
  DPMatrix (machine, seqPair)
{
  fill();
}

ForwardMatrix::ForwardMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair, const Envelope& env) :
  DPMatrix (machine, seqPair, env)
{
  fill();
}

void ForwardMatrix::fill() {
  for (OutputIndex outPos = 0; outPos <= outLen; ++outPos) {
    const OutputToken outTok = outPos ? output[outPos-1] : OutputTokenizer::emptyToken();
    for (InputIndex inPos = 0; inPos <= inLen; ++inPos) {
      const InputToken inTok = inPos ? input[inPos-1] : InputTokenizer::emptyToken();
      for (StateIndex d = 0; d < nStates; ++d) {
	const EvaluatedMachineState& state = machine.state[d];
	double ll = (inPos || outPos || d) ? -numeric_limits<double>::infinity() : 0;
	if (inPos && outPos)
	  accumulate (ll, state.incoming, inTok, outTok, inPos - 1, outPos - 1, sum_reduce);
	if (inPos)
	  accumulate (ll, state.incoming, inTok, OutputTokenizer::emptyToken(), inPos - 1, outPos, sum_reduce);
	if (outPos)
	  accumulate (ll, state.incoming, InputTokenizer::emptyToken(), outTok, inPos, outPos - 1, sum_reduce);
	accumulate (ll, state.incoming, InputTokenizer::emptyToken(), OutputTokenizer::emptyToken(), inPos, outPos, sum_reduce);
	cell(inPos,outPos,d) = ll;
      }
    }
  }
  LogThisAt(8,"Forward matrix:" << endl << *this);
}

double ForwardMatrix::logLike() const {
  return cell (inLen, outLen, machine.endState());
}
