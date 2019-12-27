#include "logger.h"

template<class IndexMapper>
MappedForwardMatrix<IndexMapper>::MappedForwardMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair) :
  DPMatrix<IndexMapper> (machine, seqPair)
{
  fill (machine.startState());
}

template<class IndexMapper>
MappedForwardMatrix<IndexMapper>::MappedForwardMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair, const Envelope& env) :
  DPMatrix<IndexMapper> (machine, seqPair, env)
{
  fill (machine.startState());
}

template<class IndexMapper>
MappedForwardMatrix<IndexMapper>::MappedForwardMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair, const Envelope& env, StateIndex startState) :
  DPMatrix<IndexMapper> (machine, seqPair, env)
{
  fill (startState);
}

template<class IndexMapper>
void MappedForwardMatrix<IndexMapper>::fill (StateIndex startState) {
  typedef DPMatrix<IndexMapper> DPM;
  ProgressLog(plogDP,6);
  plogDP.initProgress ("Filling Forward matrix (%lu cells)", DPM::nCells());
  typename DPM::CellIndex nCellsDone = 0;
  for (typename DPM::OutputIndex outPos = 0; outPos <= DPM::outLen; ++outPos) {
    const OutputToken outTok = outPos ? DPM::output[outPos-1] : OutputTokenizer::emptyToken();
    for (typename DPM::InputIndex inPos = DPM::env.inStart[outPos]; inPos < DPM::env.inEnd[outPos]; ++inPos) {
      const InputToken inTok = inPos ? DPM::input[inPos-1] : InputTokenizer::emptyToken();
      for (StateIndex d = 0; d < DPM::nStates; ++d) {
	plogDP.logProgress (nCellsDone / (double) DPM::nCells(), "filled %lu cells", nCellsDone);
	++nCellsDone;
	const EvaluatedMachineState& state = DPM::machine.state[d];
	double ll = (inPos || outPos || d != startState) ? -numeric_limits<double>::infinity() : 0;
	if (inPos && outPos)
	  DPM::accumulate (ll, state.incoming, inTok, outTok, inPos - 1, outPos - 1, DPM::sum_reduce);
	if (inPos)
	  DPM::accumulate (ll, state.incoming, inTok, OutputTokenizer::emptyToken(), inPos - 1, outPos, DPM::sum_reduce);
	if (outPos)
	  DPM::accumulate (ll, state.incoming, InputTokenizer::emptyToken(), outTok, inPos, outPos - 1, DPM::sum_reduce);
	DPM::accumulate (ll, state.incoming, InputTokenizer::emptyToken(), OutputTokenizer::emptyToken(), inPos, outPos, DPM::sum_reduce);
	DPM::cell(inPos,outPos,d) = ll;
      }
    }
  }
  LogThisAt(8,"Forward matrix:" << endl << *this);
}

template<class IndexMapper>
double MappedForwardMatrix<IndexMapper>::logLike() const {
  typedef DPMatrix<IndexMapper> DPM;
  return DPM::cell (DPM::inLen, DPM::outLen, DPM::machine.endState());
}
