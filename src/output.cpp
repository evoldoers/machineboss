#include "output.h"
#include "util.h"

vguard<string> OutputAdapter::summaryStatParams() const {
  return extract_keys (summaryStatCoeff);
}

WeightExpr OutputAdapter::loglike (const SummaryStats& stats, const ParamDefs& symParamDefs, const ParamDefs& seqParamDefs) const {
  ParamDefs paramDefs (symParamDefs);
  paramDefs.insert (seqParamDefs.begin(), seqParamDefs.end());
  WeightExpr ll;
  for (auto& ss: stats)
    ll = WeightAlgebra::add (ll, WeightAlgebra::multiply (WeightAlgebra::expand (summaryStatCoeff.at(ss.first), paramDefs),
							  ss.second));
  return ll;
}
