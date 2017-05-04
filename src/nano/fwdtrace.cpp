#include "fwdtrace.h"
#include "../logger.h"

ForwardTraceMatrix::ForwardTraceMatrix (const EvaluatedMachine& eval, const GaussianModelParams& modelParams, const TraceMoments& trace, const TraceParams& traceParams, size_t blockBytes) :
  TraceDPMatrix (eval, modelParams, trace, traceParams, blockBytes)
{
  ProgressLog(plog,3);
  plog.initProgress ("Forward algorithm (%ld samples, %u states, %u transitions)", outLen, nStates, nTrans);

  cell(0,eval.startState()) = 0;
  for (const auto& it: nullTrans())
    log_accum_exp (cell(0,it.dest), cell(0,it.src) + it.logWeight);

  for (OutputIndex outPos = 1; outPos <= outLen; ++outPos) {
    plog.logProgress ((outPos - 1) / (double) outLen, "sample %ld/%ld", outPos, outLen);
    fillColumn (outPos);
  }

  logLike = cell (outLen, eval.endState());
  LogThisAt(6,"Forward log-likelihood: " << logLike << endl);
}

void ForwardTraceMatrix::fillColumn (OutputIndex outPos) {
  vguard<double>& thisColumn = column(outPos);
  initColumn (thisColumn);
  if (outPos == 0)
    thisColumn[eval.startState()] = 0;
  else {
    const vguard<double>& prevColumn = column(outPos-1);
    for (OutputToken outTok = 1; outTok < nOutToks; ++outTok) {
      const double llEmit = logEmitProb(outPos,outTok);
      for (const auto& it: transByOut[outTok])
	log_accum_exp (thisColumn[it.dest], prevColumn[it.src] + it.logWeight + llEmit);
    }
  }
  
  for (const auto& it: nullTrans())
    log_accum_exp (thisColumn[it.dest], thisColumn[it.src] + it.logWeight);
}

void ForwardTraceMatrix::refillBlock (OutputIndex blockStart) {
  Assert (blockStart % blockSize == 0, "In ForwardTraceMatrix: block start is not aligned with checkpoint");

  const OutputIndex blockEnd = min((OutputIndex)nColumns,blockStart+blockSize) - 1;
  LogThisAt(3,"Refilling Forward matrix from sample " << (blockStart+1) << " to " << blockEnd << endl);

  for (OutputIndex outPos = blockStart + 1; outPos <= blockEnd; ++outPos)
    fillColumn (outPos);
}

MachinePath ForwardTraceMatrix::samplePath (const Machine& machine, mt19937& generator) {
  Assert (logLike > -numeric_limits<double>::infinity(), "Can't sample Forward traceback: no finite-weight paths");
  uniform_real_distribution<double> distrib;
  MachinePath path;
  OutputIndex outPos = outLen;
  StateIndex s = nStates - 1;
  while (outPos > 0 || s != 0) {
    const EvaluatedMachineState& state = eval.state[s];
    vguard<double> transLogLike;
    vguard<EvaluatedMachineState::TransIndex> transIndex;
    vguard<StateIndex> transSource;
    for (const auto& inTok_outStateTransMap: state.incoming)
      for (const auto& outTok_stateTransMap: inTok_outStateTransMap.second) {
	const OutputToken outTok = outTok_stateTransMap.first;
	if (outTok == 0 || outPos > 0)
	  for (const auto& src_trans: outTok_stateTransMap.second) {
	    const double tll = cell(outPos-(outTok?1:0),src_trans.first) + src_trans.second.logWeight + (outTok ? logEmitProb(outPos,outTok) : 0);
	    transLogLike.push_back (tll);
	    transIndex.push_back (src_trans.second.transIndex);
	    transSource.push_back (src_trans.first);
	  }
      }
    const double tll_min = *(min_element (transLogLike.begin(), transLogLike.end()));
    vguard<double> transProb (transLogLike.size());
    double tpTotal = 0;
    for (auto& tll: transLogLike) {
      transProb.push_back (exp (tll - tll_min));
      tpTotal += transProb.back();
    }
    double p = distrib(generator) * tpTotal;
    size_t ti;
    for (ti = 0; p > 0 && ti < transProb.size(); ++ti)
      p -= transProb[ti];
    const MachineTransition& trans = machine.state[transSource[ti]].getTransition (transIndex[ti]);
    if (!trans.outputEmpty()) {
      if (outPos % blockSize == 0)
	refillBlock (outPos - blockSize);
      --outPos;
    }
    s = transSource[ti];
    path.trans.push_front (trans);
  }
  return path;
}
