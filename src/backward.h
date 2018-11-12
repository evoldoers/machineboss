#ifndef BACKWARD_INCLUDED
#define BACKWARD_INCLUDED

#include "forward.h"
#include "counts.h"

class BackwardMatrix : public DPMatrix {
private:
  inline void accumulateCounts (double logOddsRatio, vguard<double>& transCounts, const EvaluatedMachineState::InOutStateTransMap& inOutStateTransMap, InputToken inTok, OutputToken outTok, InputIndex inPos, OutputIndex outPos) const {
    auto visit = [&] (StateIndex, EvaluatedMachineState::TransIndex ti, double tll) {
      transCounts[ti] += exp (logOddsRatio + tll);
    };
    iterate (inOutStateTransMap, inTok, outTok, inPos, outPos, visit);
  }

  void fill();
  
public:
  BackwardMatrix (const EvaluatedMachine&, const SeqPair&);
  BackwardMatrix (const EvaluatedMachine&, const SeqPair&, const Envelope&);
  void getCounts (const ForwardMatrix&, MachineCounts&) const;
  double logLike() const;
  MachinePath traceFrom (const Machine&, const ForwardMatrix&, InputIndex, OutputIndex, StateIndex) const;
  MachinePath traceFrom (const Machine&, const ForwardMatrix&, InputIndex, OutputIndex, StateIndex, EvaluatedMachineState::TransIndex) const;
};

#endif /* BACKWARD_INCLUDED */
