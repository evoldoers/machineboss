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
