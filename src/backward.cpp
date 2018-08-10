#include "backward.h"
#include "logger.h"

BackwardMatrix::BackwardMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair) :
  DPMatrix (machine, seqPair)
{
  fill();
}

BackwardMatrix::BackwardMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair, const Envelope& env) :
  DPMatrix (machine, seqPair, env)
{
  fill();
}

void BackwardMatrix::fill() {
  for (OutputIndex outPos = outLen; outPos >= 0; --outPos) {
    const bool endOfOutput = (outPos == outLen);
    const OutputToken outTok = endOfOutput ? OutputTokenizer::emptyToken() : output[outPos];
    for (InputIndex inPos = inLen; inPos >= 0; --inPos) {
      const bool endOfInput = (inPos == inLen);
      const InputToken inTok = endOfInput ? InputTokenizer::emptyToken() : input[inPos];
      for (int s = nStates - 1; s >= 0; --s) {
	const bool endState = (s == nStates - 1);
	const EvaluatedMachineState& state = machine.state[(StateIndex) s];
	double ll = (endOfInput && endOfOutput && endState) ? 0 : -numeric_limits<double>::infinity();
	if (!endOfInput && !endOfOutput)
	  accumulate (ll, state.outgoing, inTok, outTok, inPos + 1, outPos + 1, sum_reduce);
	if (!endOfInput)
	  accumulate (ll, state.outgoing, inTok, OutputTokenizer::emptyToken(), inPos + 1, outPos, sum_reduce);
	if (!endOfOutput)
	  accumulate (ll, state.outgoing, InputTokenizer::emptyToken(), outTok, inPos, outPos + 1, sum_reduce);
	accumulate (ll, state.outgoing, InputTokenizer::emptyToken(), OutputTokenizer::emptyToken(), inPos, outPos, sum_reduce);
	cell(inPos,outPos,(StateIndex) s) = ll;
      }
    }
  }
  LogThisAt(8,"Backward matrix:" << endl << *this);
}

double BackwardMatrix::logLike() const {
  return cell (0, 0, machine.startState());
}

void BackwardMatrix::getCounts (const ForwardMatrix& forward, MachineCounts& counts) const {
  const double ll = logLike();
  for (InputIndex inPos = inLen; inPos >= 0; --inPos) {
    const bool endOfInput = (inPos == inLen);
    const InputToken inTok = endOfInput ? InputTokenizer::emptyToken() : input[inPos];
    for (OutputIndex outPos = outLen; outPos >= 0; --outPos) {
      const bool endOfOutput = (outPos == outLen);
      const OutputToken outTok = endOfOutput ? OutputTokenizer::emptyToken() : output[outPos];
      for (int s = nStates - 1; s >= 0; --s) {
	const bool endState = (s == nStates - 1);
	const EvaluatedMachineState& state = machine.state[(StateIndex) s];
	const double logOddsRatio = forward.cell(inPos,outPos,(StateIndex) s) - ll;
	vguard<double>& transCounts = counts.count[(StateIndex) s];
	if (!endOfInput && !endOfOutput)
	  accumulateCounts (logOddsRatio, transCounts, state.outgoing, inTok, outTok, inPos + 1, outPos + 1);
	if (!endOfInput)
	  accumulateCounts (logOddsRatio, transCounts, state.outgoing, inTok, OutputTokenizer::emptyToken(), inPos + 1, outPos);
	if (!endOfOutput)
	  accumulateCounts (logOddsRatio, transCounts, state.outgoing, InputTokenizer::emptyToken(), outTok, inPos, outPos + 1);
	accumulateCounts (logOddsRatio, transCounts, state.outgoing, InputTokenizer::emptyToken(), OutputTokenizer::emptyToken(), inPos, outPos);
      }
    }
  }
}
