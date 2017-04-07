#ifndef OBJ_GAUSSIAN_INCLUDED
#define OBJ_GAUSSIAN_INCLUDED

#include "obj_adapter.h"

struct GaussianOutputAdapter : OutputAdapter {
  string meanParam, sdParam;  // symbol parameters
  string shiftParam, scaleParam;  // sequence parameters
  string m0Param, m1Param, m2Param;  // summary statistic (moment) parameters
  GaussianOutputAdapter();
  SummaryStats summaryStats (const OutputObject& outputUnit) const;
};

#endif /* OBJ_GAUSSIAN_INCLUDED */
