#ifndef SEGMENTER_INCLUDED
#define SEGMENTER_INCLUDED

#include "moments.h"

struct Segmenter {
  const Trace& trace;
  const size_t seqLen, maxSegLen;
  const double maxFracDiff;
  // minSegs[seqPos][segLen-1] = minimum# of segments for samples (0..seqPos) ending in a segment of segLen
  vguard<vguard<int> > minSegs;
  Segmenter (const Trace& trace, double maxFracDiff, size_t maxSegLen);
  TraceMoments segments() const;
  friend ostream& operator<< (ostream&, const Segmenter&);
};

#endif /* SEGMENTER_INCLUDED */
