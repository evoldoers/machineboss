#ifndef BACKWARD_INCLUDED
#define BACKWARD_INCLUDED

#include "forward.h"
#include "counts.h"

class BackwardMatrix : public DPMatrix {
private:
  inline void accumulateCounts (double logOddsRatio, vguard<double>& transCounts, const EvaluatedMachineState::InOutTransMap& inOutTransMap, InputToken inTok, OutputToken outTok, InputIndex inPos, OutputIndex outPos) const {
    auto visit = [&] (StateIndex, EvaluatedMachineState::TransIndex ti, double tll) { transCounts[ti] += exp (logOddsRatio + tll); };
    iterate (inOutTransMap, inTok, outTok, inPos, outPos, visit);
  }

public:
  BackwardMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair);
  void getCounts (const ForwardMatrix&, MachineCounts&) const;
  double logLike() const;
};

#endif /* BACKWARD_INCLUDED */
