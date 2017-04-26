#include "backtrace.h"
#include "logger.h"

BackwardTraceMatrix::BackwardTraceMatrix (const EvaluatedMachine& eval, const GaussianModelParams& modelParams, const Trace& trace, const TraceParams& traceParams) :
  TraceDPMatrix (eval, modelParams, trace, traceParams),
  nullTrans_rbegin (nullTrans().rbegin()),
  nullTrans_rend (nullTrans().rend())
{
  cell(outLen,eval.endState()) = 0;
  for (auto iter = nullTrans_rbegin; iter != nullTrans_rend; ++iter)
    log_accum_exp (cell(outLen,(*iter).src), cell(outLen,(*iter).dest) + it.logWeight);

  for (OutputIndex outPos = outLen - 1; outPos >= 0; --outPos) {
    for (OutputToken out = 1; out < nOutToks; ++out) {
      const double llEmit = logEmitProb(outPos+1,outTok);
      for (const auto& it: transByOut[out])
	log_accum_exp (cell(outPos,it.src), cell(outPos+1,it.dest) + it.logWeight + llEmit);
    }

    for (auto iter = nullTrans_rbegin; iter != nullTrans_rend; ++iter)
      log_accum_exp (cell(outPos,(*iter).src), cell(outPos,(*iter).dest) + it.logWeight);
  }
  LogThisAt(8,"Backward matrix:" << endl << toJsonString());
}

double BackwardMatrix::logLike() const {
  return cell (0, machine.startState());
}

void BackwardMatrix::getMachineCounts (const TraceForwardMatrix& fwd, MachineCounts& counts) const {
  const double llFinal = fwd.logLike();

  for (const auto& it: nullTrans())
    counts.count[it.src][it.transIndex] += exp (fwd.cell(0,it.src) + it.logWeight + cell(0,it.dest) - llFinal);

  for (OutputIndex outPos = 1; outPos <= outLen; ++outPos) {
    for (OutputToken out = 1; out < nOutToks; ++out) {
      const double llEmit = logEmitProb(outPos,outTok);
      for (const auto& it: transByOut[out])
	counts.count[it.src][it.transIndex] += exp (fwd.cell(outPos-1,it.src) + it.logWeight + llEmit + cell(outPos,it.dest) - llFinal);
    }

    for (const auto& it: transByOut.front())
      counts.count[it.src][it.transIndex] += exp (fwd.cell(outPos,it.src) + it.logWeight + cell(outPos,it.dest) - llFinal);
  }
}

void BackwardMatrix::getGaussianCounts (const TraceForwardMatrix& fwd, vguard<GaussianCounts>& counts) const {
  const double llFinal = fwd.logLike();

  for (OutputIndex outPos = 1; outPos <= outLen; ++outPos)
    for (OutputToken out = 1; out < nOutToks; ++out) {
      const double llEmit = logEmitProb(outPos,outTok);
      const auto& sample = moments.sample[outPos-1];
      for (const auto& it: transByOut[out])
	counts[outTok-1].inc (sample, exp (fwd.cell(outPos-1,it.src) + it.logWeight + llEmit + cell(outPos,it.dest) - llFinal));
    }
}

void BackwardMatrix::getCounts (const TraceForwardMatrix& fwd, GaussianModelCounts& modelCounts) const {
  getMachineCounts (fwd, modelCounts.machine);
  getGaussianCounts (fwd, modelCounts.gauss);
}
