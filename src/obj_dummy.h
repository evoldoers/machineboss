#ifndef OBJ_DUMMY_INCLUDED
#define OBJ_DUMMY_INCLUDED

#include "obj_adapter.h"

struct DummyOutputAdapter : OutputAdapter {
  vguard<OutputSymbol> outputAlphabet;
  string indicatorPrefix, selectorPrefix;
  OutputMap outputMap;
  DummyOutputAdapter (const Machine& machine, const char* indicatorPrefix = "indicator_", const char* selectorPrefix = "selector_");
  SummaryStats summaryStats (const OutputObject& outputUnit) const;
};

#endif /* OBJ_DUMMY_INCLUDED */

