#include "segmenter.h"
#include "../logger.h"

Segmenter::Segmenter (const Trace& trace, double maxFracDiff, size_t maxSegLen) :
  trace (trace),
  seqLen (trace.sample.size()),
  maxSegLen (maxSegLen),
  maxFracDiff (maxFracDiff),
  minSegs (seqLen, vguard<int> (maxSegLen + 1, seqLen))
{
  minSegs[0][0] = 1;
  for (size_t seqPos = 1; seqPos < seqLen; ++seqPos) {
    for (size_t segLen = 1; segLen <= maxSegLen; ++segLen)
      minSegs[seqPos][0] = min (minSegs[seqPos][0], 1 + minSegs[seqPos-1][segLen-1]);
    const double x = trace.sample[seqPos];
    for (size_t segLen = 2; segLen < seqPos && segLen <= maxSegLen; ++segLen) {
      const double xPrev = trace.sample[seqPos-segLen-1];
      const double diff = x - xPrev, xMin = min(abs(x),abs(xPrev)), fracDiff = diff/xMin;
      if (fracDiff > maxFracDiff)
	break;
      minSegs[seqPos][segLen-1] = minSegs[seqPos-1][segLen-2];
    }
  }
}

TraceMoments Segmenter::segments() const {
  list<SampleMoments> s;
  size_t seqPos = seqLen;
  while (seqPos > 0) {
    size_t bestMinSegLen, prevSegLen;
    for (size_t segLen = 1; segLen <= maxSegLen; ++segLen)
      if (segLen == 1 || minSegs[seqPos-1][segLen-1] < bestMinSegLen) {
	bestMinSegLen = minSegs[seqPos-1][segLen-1];
	prevSegLen = segLen;
      }
    s.push_front (SampleMoments (trace, seqPos - prevSegLen, prevSegLen));
    seqPos -= prevSegLen;
  }
  TraceMoments m;
  m.name = trace.name;
  m.sample = vguard<SampleMoments> (s.begin(), s.end());
  LogThisAt(5,"Trace " << trace.name << ": segmented " << trace.sample.size() << " samples into " << m.sample.size() << " events" << endl);
  return m;
}
