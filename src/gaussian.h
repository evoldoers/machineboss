#ifndef GAUSSIAN_INCLUDED
#define GAUSSIAN_INCLUDED

#include "output.h"

struct GaussianOutputAdapter : OutputAdapter {
  string meanParam, sdParam;  // symbol parameters
  string shiftParam, scaleParam;  // sequence parameters
  string m0Param, m1Param, m2Param;  // summary statistic (moment) parameters
  GaussianOutputAdapter();
  SummaryStats summaryStats (const OutputObject& outputUnit) const;
};

#endif /* GAUSSIAN_INCLUDED */
