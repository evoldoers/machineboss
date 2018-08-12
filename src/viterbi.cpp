#include "viterbi.h"
#include "logger.h"

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
  plogDP.initProgress ("Filling Viterbi matrix (%lu cells)", nCells());
  for (OutputIndex outPos = 0; outPos <= outLen; ++outPos) {
    plogDP.logProgress (nStates * offsets[outPos] / nCells(), "filled %lu cells", nStates * offsets[outPos]);
    const OutputToken outTok = outPos ? output[outPos-1] : OutputTokenizer::emptyToken();
    for (InputIndex inPos = env.inStart[outPos]; inPos < env.inEnd[outPos]; ++inPos) {
      const InputToken inTok = inPos ? input[inPos-1] : InputTokenizer::emptyToken();
      for (StateIndex d = 0; d < nStates; ++d) {
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
  Assert (logLike() > -numeric_limits<double>::infinity(), "Can't do traceback: no finite-weight paths");
  MachinePath path;
  InputIndex inPos = inLen;
  OutputIndex outPos = outLen;
  StateIndex s = nStates - 1;
  while (inPos > 0 || outPos > 0 || s != 0) {
    const EvaluatedMachineState& state = machine.state[s];
    double bestLogLike = -numeric_limits<double>::infinity();
    StateIndex bestSource;
    EvaluatedMachineState::TransIndex bestTransIndex;
    const InputToken inTok = inPos ? input[inPos-1] : InputTokenizer::emptyToken();
    const OutputToken outTok = outPos ? output[outPos-1] : OutputTokenizer::emptyToken();
    if (inPos && outPos)
      pathIterate (bestLogLike, bestSource, bestTransIndex, state.incoming, inTok, outTok, inPos - 1, outPos - 1);
    if (inPos)
      pathIterate (bestLogLike, bestSource, bestTransIndex, state.incoming, inTok, OutputTokenizer::emptyToken(), inPos - 1, outPos);
    if (outPos)
      pathIterate (bestLogLike, bestSource, bestTransIndex, state.incoming, InputTokenizer::emptyToken(), outTok, inPos, outPos - 1);
    pathIterate (bestLogLike, bestSource, bestTransIndex, state.incoming, InputTokenizer::emptyToken(), OutputTokenizer::emptyToken(), inPos, outPos);
    const MachineTransition& bestTrans = m.state[bestSource].getTransition (bestTransIndex);
    if (!bestTrans.inputEmpty()) --inPos;
    if (!bestTrans.outputEmpty()) --outPos;
    s = bestSource;
    path.trans.push_front (bestTrans);
  }
  return path;
}
