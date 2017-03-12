#ifndef VITERBI_INCLUDED
#define VITERBI_INCLUDED

#include "dpmatrix.h"

class ViterbiMatrix : public DPMatrix {
private:
  inline void traceIterate (double& bestLogLike, StateIndex& bestSource, EvaluatedMachineState::TransIndex& bestTransIndex, const EvaluatedMachineState::InOutTransMap& inOutTransMap, InputToken inTok, OutputToken outTok, InputIndex inPos, OutputIndex outPos) const {
    auto visit = [&] (StateIndex src, EvaluatedMachineState::TransIndex ti, double tll) {
      if (tll > bestLogLike) {
	bestLogLike = tll;
	bestSource = src;
	bestTransIndex = ti;
      }
    };
    iterate (inOutTransMap, inTok, outTok, inPos, outPos, visit);
  }

public:
  ViterbiMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair);
  double logLike() const;
  MachinePath trace (const Machine&) const;
};

#endif /* VITERBI_INCLUDED */
