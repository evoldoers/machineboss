#ifndef JPHMM_INCLUDED
#define JPHMM_INCLUDED

#include "machine.h"
#include "fastseq.h"

namespace MachineBoss {

struct JPHMM : Machine {
  JPHMM (const vguard<FastSeq>&);
  const size_t rows, cols;
  static const string jumpParam;

  inline StateIndex emitState (int row, int col) const { return rows*col + row + 1; }
};

};

#endif /* JPHMM_INCLUDED */
