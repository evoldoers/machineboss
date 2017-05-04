#include "../logger.h"
#include "backtrace.h"

BackwardTraceMatrix::BackwardTraceMatrix (ForwardTraceMatrix& fwd, MachineCounts* transCounts, vguard<GaussianCounts>* emitCounts) :
  TraceDPMatrix (fwd.eval, fwd.modelParams, fwd.moments, fwd.traceParams, fwd.blockBytes),
  nullTrans_rbegin (nullTrans().rbegin()),
  nullTrans_rend (nullTrans().rend())
{
  const double llFinal = fwd.logLike;

  ProgressLog(plog,3);
  plog.initProgress ("Backward algorithm (%ld samples, %u states, %u transitions)", outLen, nStates, nTrans);

  for (OutputIndex outPos = outLen; outPos >= 0; --outPos) {
    plog.logProgress ((outLen - outPos) / (double) outLen, "sample %ld/%ld", outPos, outLen);

    if (outPos < outLen && ((outPos + 1) % fwd.blockSize) == 0)
      fwd.refillBlock (outPos + 1 - fwd.blockSize);

    vguard<double>& thisColumn = column(outPos);
    initColumn (thisColumn);

    const vguard<double>& thisFwdColumn = fwd.column(outPos);

    if (outPos == outLen)
      thisColumn[eval.endState()] = 0;
    else {
      const auto& sample = moments.sample[outPos];
      const vguard<double>& nextColumn = column(outPos+1);
      for (OutputToken outTok = 1; outTok < nOutToks; ++outTok) {
	const double llEmit = logEmitProb(outPos+1,outTok);
	for (const auto& it: transByOut[outTok]) {
	  const double llTrans = nextColumn[it.dest] + it.logWeight + llEmit;
	  log_accum_exp (thisColumn[it.src], llTrans);
	  if (transCounts || emitCounts) {
	    const double ppEmit = exp (thisFwdColumn[it.src] + llTrans - llFinal);
	    if (transCounts)
	      transCounts->count[it.src][it.transIndex] += ppEmit;
	    if (emitCounts)
	      (*emitCounts)[outTok-1].inc (sample, ppEmit);
	  }
	}
      }
    }

    for (auto iter = nullTrans_rbegin; iter != nullTrans_rend; ++iter) {
      const double llTrans = thisColumn[(*iter).dest] + (*iter).logWeight;
      log_accum_exp (thisColumn[(*iter).src], llTrans);
      if (transCounts)
	transCounts->count[(*iter).src][(*iter).transIndex] += exp (thisFwdColumn[(*iter).src] + llTrans - llFinal);
    }
  }
  LogThisAt(6,"Backward log-likelihood: " << logLike() << endl);
}

double BackwardTraceMatrix::logLike() const {
  return cell (0, eval.startState());
}
