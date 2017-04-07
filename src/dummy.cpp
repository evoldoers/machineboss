#include "dummy.h"

DummyOutputAdapter::DummyOutputAdapter (const Machine& machine, const char* indicatorPrefix, const char* selectorPrefix)
  : outputAlphabet (machine.outputAlphabet()),
    indicatorPrefix (indicatorPrefix),
    selectorPrefix (selectorPrefix)
{
  for (const auto& s: outputAlphabet) {
    const string indicatorParam = indicatorPrefix + s;
    const string selectorParam = selectorPrefix + s;
    summaryStatCoeff[indicatorParam] = selectorParam;

    ParamDefs defs;
    for (const auto& t: outputAlphabet)
      defs[selectorPrefix+t] = (s == t ? 0 : -numeric_limits<double>::infinity());
    outputMap.symParamDefs[s] = defs;
  }
}

OutputAdapter::SummaryStats DummyOutputAdapter::summaryStats (const OutputObject& outputUnit) const {
  SummaryStats stats;
  const OutputSymbol outSym = outputUnit.get<string>();
  stats[indicatorPrefix + outSym] = true;
  return stats;
}
