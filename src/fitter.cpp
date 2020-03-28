#include "fitter.h"
#include "eval.h"
#include "counts.h"
#include "logger.h"

#define MaxEMIterations 1000
#define MinEMImprovement .001

using namespace MachineBoss;

Constraints MachineFitter::allConstraints() const {
  return machine.cons.combine (constraints);
}

Params MachineFitter::fit (const SeqPairList& trainingSet) const {
  return fit (trainingSet, trainingSet.envelopes());
}

Params MachineFitter::fit (const SeqPairList& trainingSet, size_t width) const {
  return fit (trainingSet, trainingSet.envelopes (width));
}

Params MachineFitter::fit (const SeqPairList& trainingSet, const list<Envelope>& envelopes) const {
  Assert (envelopes.size() == trainingSet.seqPairs.size(), "Envelope/training set mismatch");
  Params params = seed;
  double prev;
  for (size_t iter = 0; true; ++iter) {
    const Params allParams = machine.funcs.combine(constants).combine(params);
    const EvaluatedMachine eval (machine, allParams);
    const MachineCounts counts (eval, trainingSet, envelopes);
    LogThisAt(2,"Baum-Welch iteration #" << (iter+1) << ": log-likelihood " << counts.loglike << endl);
    LogThisAt(4,"Parameters:" << endl << JsonWriter<Params>::toJsonString(params) << endl);
    if (iter > 0) {
      if (iter == MaxEMIterations)
	break;
      const double improvement = (counts.loglike - prev) / abs(prev);
      if (improvement < MinEMImprovement)
	break;
    }
    LogThisAt(5,"Constructing M-step objective function" << endl);
    MachineObjective objective (machine, counts, constraints, constants);
    LogThisAt(5,"Optimizing M-step objective function" << endl);
    params = objective.optimize (params);
    prev = counts.loglike;
  }
  return params;
}
