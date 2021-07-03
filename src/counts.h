#ifndef COUNTS_INCLUDED
#define COUNTS_INCLUDED

#include "eval.h"
#include "seqpair.h"
#include "constraints.h"

// E-step
namespace MachineBoss {

struct MachineCounts {
  vguard<vguard<double> > count;  // indexed: count[state][nTrans]
  double loglike;
  MachineCounts();
  MachineCounts (const EvaluatedMachine&);
  MachineCounts (const EvaluatedMachine&, const SeqPair&);
  MachineCounts (const EvaluatedMachine&, const SeqPairList&, const list<Envelope>& = list<Envelope>());
  void init (const EvaluatedMachine&);
  double add (const EvaluatedMachine&, const SeqPair&);  // returns log-likelihood
  double add (const EvaluatedMachine&, const SeqPair&, const Envelope&);  // returns log-likelihood
  MachineCounts& operator+= (const MachineCounts&);
  map<string,double> paramCounts (const Machine&, const ParamAssign&) const;  // expectation of d(logLike)/d(logParam)
  void writeJson (ostream&) const;
  void writeParamCountsJson (ostream&, const Machine&, const ParamAssign&) const;
};

// M-step
struct MachineObjective {
  const Constraints constraints;
  vguard<string> transformedParam;
  map<string,size_t> transformedParamIndex;
  ParamDefs constantDefs, paramTransformDefs, allDefs;
  WeightExpr objective;
  vguard<WeightExpr> deriv;
  MachineObjective (const Machine&, const MachineCounts&, const Constraints&, const Params&);
  Params optimize (const Params& seed) const;
  string toString() const;
};

}  // end namespace

#endif /* COUNTS_INCLUDED */

