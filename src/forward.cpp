#include "forward.h"

ForwardMatrix::ForwardMatrix (const EvaluatedMachine& machine, const vguard<InputToken>& input, const vguard<OutputToken>& output) :
  DPMatrix (machine, input, output)
{
  cell (0, 0, machine.startState()) = 0;
  for (InputIndex inPos = 0; inPos <= inLen; ++inPos) {
    const InputToken inTok = inPos ? input[inPos-1] : InputTokenizer::emptyToken();
    for (OutputIndex outPos = 0; outPos <= outLen; ++outPos) {
      const OutputToken outTok = outPos ? output[outPos-1] : OutputTokenizer::emptyToken();
      for (StateIndex d = 0; d < nStates; ++d) {
	const EvaluatedMachineState& state = machine.state[d];
	double ll = -numeric_limits<double>::infinity();
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
}

double ForwardMatrix::logLike() const {
  return cell (inLen, outLen, machine.endState());
}
