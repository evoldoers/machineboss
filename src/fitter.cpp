#include "fitter.h"
#include "eval.h"
#include "counts.h"
#include "logger.h"

#define MaxEMIterations 1000
#define MinEMImprovement .001

Constraints MachineFitter::allConstraints() const {
  return machine.cons.combine (constraints);
}

Params MachineFitter::fit (const SeqPairList& trainingSet) const {
  Params params = seed;
  double prev;
  for (size_t iter = 0; true; ++iter) {
    const Params allParams = machine.defs.combine(constants).combine(params);
    const EvaluatedMachine eval (machine, allParams);
    MachineCounts counts (eval);
    double loglike = 0;
    for (const auto& seqPair: trainingSet.seqPairs)
      loglike += counts.add (eval, seqPair);
    LogThisAt(2,"Baum-Welch iteration #" << (iter+1) << ": log-likelihood " << loglike << endl);
    LogThisAt(4,"Parameters:" << endl << JsonWriter<Params>::toJsonString(params) << endl);
    if (iter > 0) {
      if (iter == MaxEMIterations)
	break;
      const double improvement = (loglike - prev) / abs(prev);
      if (improvement < MinEMImprovement)
	break;
    }
    MachineObjective objective (machine, counts, constraints, constants);
    params = objective.optimize (params);
    prev = loglike;
  }
  return params;
}
