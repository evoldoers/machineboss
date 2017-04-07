#ifndef OBJ_ADAPTER_INCLUDED
#define OBJ_ADAPTER_INCLUDED

#include "machine.h"

typedef json OutputObject;

struct OutputAdapter {
  typedef ParamDefs SummaryStats;

  ParamDefs summaryStatCoeff;
  vguard<string> symParams, seqParams;
  // TODO: each OutputAdapter should have a schema for validating OutputObjects
  
  vguard<string> summaryStatParams() const;
  WeightExpr loglike (const SummaryStats& stats, const ParamDefs& symParamDefs, const ParamDefs& seqParamDefs) const;

  virtual SummaryStats summaryStats (const OutputObject& outputUnit) const = 0;
};

struct OutputMap {
  map<OutputSymbol,ParamDefs> symParamDefs;

  void readJson (const json& json);
  void writeJson (ostream& out) const;
};

#endif /* OBJ_ADAPTER_INCLUDED */

