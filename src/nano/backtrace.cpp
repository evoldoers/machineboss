#include "../logger.h"
#include "backtrace.h"

BackwardTraceMatrix::BackwardTraceMatrix (ForwardTraceMatrix& fwd, MachineCounts* transCounts, vguard<GaussianCounts>* emitCounts) :
  TraceDPMatrix (fwd.eval, fwd.modelParams, fwd.moments, fwd.traceParams, fwd.blockBytes, fwd.bandWidth),
  nullTrans_rbegin (nullTrans.rbegin()),
  nullTrans_rend (nullTrans.rend())
{
  const double llFinal = fwd.logLike;
  Assert (llFinal > -numeric_limits<double>::infinity(), "Can't get Forward-Backward counts: Forward likelihood is zero");

  ProgressLog(plog,3);
  plog.initProgress ("Backward algorithm (%ld samples, %u states, %u transitions)", outLen, nStates, nTrans);

  for (OutputIndex outPos = outLen; outPos >= 0; --outPos) {
    plog.logProgress ((outLen - outPos) / (double) outLen, "sample %ld/%ld", outPos, outLen);

    vguard<double>& thisColumn = column(outPos);
    initColumn (thisColumn);

    fwd.readyColumn(outPos);
    const vguard<double>& thisFwdColumn = fwd.column(outPos);

    if (outPos == outLen)
      thisColumn[eval.endState()] = 0;
    else {
      const auto& sample = moments.sample[outPos];
      const vguard<double>& nextColumn = column(outPos+1);
      const auto itBegin = bandTransBegin(outPos), itEnd = bandTransEnd(outPos);
      for (auto itIter = itBegin; itIter != itEnd; ++itIter) {
	const auto& it = *itIter;
	const OutputToken outTok = it.out;
	const double llEmit = logEmitProb(outPos+1,outTok);
	const double llTrans = nextColumn[it.dest] + logTransProb(outPos+1,it) + llEmit;
	log_accum_exp (thisColumn[it.src], llTrans);
	if (transCounts || emitCounts) {
	  const double ppEmit = exp (thisFwdColumn[it.src] + llTrans - llFinal);
	  if (transCounts) {
	    transCounts->count[it.src][it.transIndex] += ppEmit;
	    if (it.loop) {
	      const auto& mom = moments.sample[outPos];
	      transCounts->count[it.dest][it.loopTransIndex] += (mom.m0 - 1) * ppEmit;
	    }
	  }
	  if (emitCounts)
	    (*emitCounts)[outTok-1].inc (sample, ppEmit);
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
  LogThisAt(10,"Backward matrix:" << endl << *this);
}

double BackwardTraceMatrix::logLike() const {
  return cell (0, eval.startState());
}
