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
  ProgressLog(plogDP,7);
  plogDP.initProgress ("Filling Backward matrix (%lu cells)", nCells());
  for (OutputIndex outPos = outLen; outPos >= 0; --outPos) {
    plogDP.logProgress (nStates * (offsets.back() - offsets[outPos+1]) / (double) nCells(), "filled %lu cells", nStates * (offsets.back() - offsets[outPos+1]));
    const bool endOfOutput = (outPos == outLen);
    const OutputToken outTok = endOfOutput ? OutputTokenizer::emptyToken() : output[outPos];
    for (InputIndex inPos = env.inEnd[outPos] - 1; inPos >= env.inStart[outPos]; --inPos) {
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
  ProgressLog(plogDP,6);
  plogDP.initProgress ("Calculating Forward-Backward counts (%lu cells)", nCells());
  const double ll = logLike();
  for (OutputIndex outPos = outLen; outPos >= 0; --outPos) {
    plogDP.logProgress (nStates * (offsets.back() - offsets[outPos+1]) / (double) nCells(), "counted %lu cells", nStates * (offsets.back() - offsets[outPos+1]));
    const bool endOfOutput = (outPos == outLen);
    const OutputToken outTok = endOfOutput ? OutputTokenizer::emptyToken() : output[outPos];
    for (InputIndex inPos = env.inEnd[outPos] - 1; inPos >= env.inStart[outPos]; --inPos) {
      const bool endOfInput = (inPos == inLen);
      const InputToken inTok = endOfInput ? InputTokenizer::emptyToken() : input[inPos];
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
