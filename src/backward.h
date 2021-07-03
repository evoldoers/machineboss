#ifndef BACKWARD_INCLUDED
#define BACKWARD_INCLUDED

#include <queue>
#include "forward.h"
#include "counts.h"

namespace MachineBoss {

class BackwardMatrix : public DPMatrix<IdentityIndexMapper> {
public:
  typedef function<void(StateIndex,EvaluatedMachineState::TransIndex,InputIndex,OutputIndex,double)> BackTransVisitor;
  static BackTransVisitor transitionCounter (MachineCounts& counts) {
    BackTransVisitor tv = [&] (StateIndex s, EvaluatedMachineState::TransIndex ti, InputIndex, OutputIndex, double postProb) {
      counts.count[s][ti] += postProb;
    };
    return tv;
  }

  struct PostTrans {
    InputIndex inPos;
    OutputIndex outPos;
    StateIndex src;
    EvaluatedMachineState::TransIndex transIndex;
    double weight;
    bool operator< (const PostTrans& ptq) const { return weight < ptq.weight; }
  };
  typedef priority_queue<PostTrans> PostTransQueue;
  static BackTransVisitor transitionSorter (PostTransQueue& ptq) {
    BackTransVisitor tv = [&] (StateIndex s, EvaluatedMachineState::TransIndex ti, InputIndex ip, OutputIndex op, double postProb) {
      ptq.push (PostTrans ({ .inPos = ip, .outPos = op, .src = s, .transIndex = ti, .weight = postProb }));
    };
    return tv;
  }

private:
  inline void accumulateCounts (double logOddsRatio, const BackTransVisitor& tv, StateIndex src, const EvaluatedMachineState::InOutStateTransMap& inOutStateTransMap, InputToken inTok, OutputToken outTok, InputIndex inPos, OutputIndex outPos) const {
    auto visit = [&] (StateIndex, EvaluatedMachineState::TransIndex ti, double tll) {
      tv (src, ti, inPos, outPos, exp (logOddsRatio + tll));
    };
    iterate (inOutStateTransMap, inTok, outTok, inPos, outPos, visit);
  }

  void fill();
  
public:
  BackwardMatrix (const EvaluatedMachine&, const SeqPair&);
  BackwardMatrix (const EvaluatedMachine&, const SeqPair&, const Envelope&);
  void getCounts (const ForwardMatrix&, const BackTransVisitor&) const;
  void getCounts (const ForwardMatrix&, MachineCounts&) const;
  double logLike() const;
  PostTransQueue postTransQueue (const ForwardMatrix&) const;
  MachinePath traceFrom (const Machine&, const ForwardMatrix&, InputIndex, OutputIndex, StateIndex) const;
  MachinePath traceFrom (const Machine&, const ForwardMatrix&, InputIndex, OutputIndex, StateIndex, EvaluatedMachineState::TransIndex) const;
  void traceFrom (const Machine&, const ForwardMatrix&, InputIndex, OutputIndex, StateIndex, EvaluatedMachineState::TransIndex, TraceTerminator) const;
};

}  // end namespace

#endif /* BACKWARD_INCLUDED */
