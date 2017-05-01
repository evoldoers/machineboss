#include "vtrace.h"
#include "../logger.h"

ViterbiTraceMatrix::ViterbiTraceMatrix (const EvaluatedMachine& eval, const GaussianModelParams& modelParams, const TraceMoments& trace, const TraceParams& traceParams) :
  TraceDPMatrix (eval, modelParams, trace, traceParams)
{
  ProgressLog(plog,3);
  plog.initProgress ("Viterbi algorithm (%ld samples, %u transitions)", outLen, nTrans);

  cell(0,eval.startState()) = 0;
  for (const auto& it: nullTrans())
    update (0, it.dest, cell(0,it.src) + it.logWeight, it.in);

  for (OutputIndex outPos = 1; outPos <= outLen; ++outPos) {
    plog.logProgress ((outPos - 1) / (double) outLen, "sample %ld/%ld", outPos, outLen);
    for (OutputToken outTok = 1; outTok < nOutToks; ++outTok) {
      const double llEmit = logEmitProb(outPos,outTok);
      for (const auto& it: transByOut[outTok])
	update (outPos, it.dest, cell(outPos-1,it.src) + it.logWeight + llEmit, it.in);
    }

    for (const auto& it: nullTrans())
      update (outPos, it.dest, cell(outPos,it.src) + it.logWeight, it.in);
  }
  LogThisAt(6,"Viterbi log-likelihood: " << logLike() << endl);
  LogThisAt(10,"Viterbi matrix:" << endl << *this);
}

double ViterbiTraceMatrix::logLike() const {
  return cell (outLen, eval.endState());
}

MachinePath ViterbiTraceMatrix::path (const Machine& m) const {
  Assert (logLike() > -numeric_limits<double>::infinity(), "Can't do traceback: no finite-weight paths");
  MachinePath path;
  OutputIndex outPos = outLen;
  StateIndex s = nStates - 1;
  while (outPos > 0 || s != 0) {
    const EvaluatedMachineState& state = eval.state[s];
    double bestLogLike = -numeric_limits<double>::infinity();
    StateIndex bestSource;
    EvaluatedMachineState::TransIndex bestTransIndex;

    for (const auto& inTok_outStateTransMap: state.incoming)
      for (const auto& outTok_stateTransMap: inTok_outStateTransMap.second) {
	const OutputToken outTok = outTok_stateTransMap.first;
	if (outTok == 0 || outPos > 0)
	  for (const auto& src_trans: outTok_stateTransMap.second) {
	    const double tll = cell(outPos-(outTok?1:0),src_trans.first) + src_trans.second.logWeight + (outTok ? logEmitProb(outPos,outTok) : 0);
	    if (tll > bestLogLike) {
	      bestLogLike = tll;
	      bestSource = src_trans.first;
	      bestTransIndex = src_trans.second.transIndex;
	    }
	  }
      }
    const MachineTransition& bestTrans = m.state[bestSource].getTransition (bestTransIndex);
    if (!bestTrans.outputEmpty()) --outPos;
    s = bestSource;
    path.trans.push_front (bestTrans);
  }
  return path;
}
