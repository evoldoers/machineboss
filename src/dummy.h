#ifndef DUMMY_INCLUDED_INCLUDED
#define DUMMY_INCLUDED_INCLUDED

#include "output.h"

struct DummyOutputAdapter : OutputAdapter {
  vguard<OutputSymbol> outputAlphabet;
  string indicatorPrefix, selectorPrefix;
  OutputMap outputMap;
  DummyOutputAdapter (const Machine& machine, const char* indicatorPrefix = "indicator_", const char* selectorPrefix = "selector_");
  SummaryStats summaryStats (const OutputObject& outputUnit) const;
};

#endif /* DUMMY_INCLUDED_INCLUDED */

