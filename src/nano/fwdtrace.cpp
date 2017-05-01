#include "fwdtrace.h"
#include "../logger.h"

ForwardTraceMatrix::ForwardTraceMatrix (const EvaluatedMachine& eval, const GaussianModelParams& modelParams, const TraceMoments& trace, const TraceParams& traceParams) :
  TraceDPMatrix (eval, modelParams, trace, traceParams)
{
  ProgressLog(plog,3);
  plog.initProgress ("Forward algorithm (%ld samples, %u transitions)", outLen, nTrans);

  cell(0,eval.startState()) = 0;
  for (const auto& it: nullTrans())
    log_accum_exp (cell(0,it.dest), cell(0,it.src) + it.logWeight);

  for (OutputIndex outPos = 1; outPos <= outLen; ++outPos) {
    plog.logProgress ((outPos - 1) / (double) outLen, "sample %ld/%ld", outPos, outLen);
    for (OutputToken outTok = 1; outTok < nOutToks; ++outTok) {
      const double llEmit = logEmitProb(outPos,outTok);
      for (const auto& it: transByOut[outTok])
	log_accum_exp (cell(outPos,it.dest), cell(outPos-1,it.src) + it.logWeight + llEmit);
    }

    for (const auto& it: nullTrans())
      log_accum_exp (cell(outPos,it.dest), cell(outPos,it.src) + it.logWeight);
  }
  LogThisAt(6,"Forward log-likelihood: " << logLike() << endl);
  LogThisAt(10,"Forward matrix:" << endl << *this);
}

double ForwardTraceMatrix::logLike() const {
  return cell (outLen, eval.endState());
}
