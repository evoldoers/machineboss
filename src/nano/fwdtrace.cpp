#include "fwdtrace.h"
#include "logger.h"

ForwardTraceMatrix::ForwardTraceMatrix (const EvaluatedMachine& eval, const GaussianModelParams& modelParams, const Trace& trace, const TraceParams& traceParams) :
  TraceDPMatrix (eval, modelParams, trace, traceParams)
{
  cell(0,eval.startState()) = 0;
  for (const auto& it: nullTrans())
    log_accum_exp (cell(0,it.dest), cell(0,it.src) + it.logWeight);

  for (OutputIndex outPos = 1; outPos <= outLen; ++outPos) {
    for (OutputToken out = 1; out < nOutToks; ++out) {
      const double llEmit = logEmitProb(outPos,outTok);
      for (const auto& it: transByOut[out])
	log_accum_exp (cell(outPos,it.dest), cell(outPos-1,it.src) + it.logWeight + llEmit);
    }

    for (const auto& it: nullTrans())
      log_accum_exp (cell(outPos,it.dest), cell(outPos,it.src) + it.logWeight);
  }
  LogThisAt(8,"Forward matrix:" << endl << toJsonString());
}

double ForwardMatrix::logLike() const {
  return cell (outLen, machine.endState());
}