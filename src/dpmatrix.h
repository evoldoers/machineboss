#ifndef DPMATRIX_INCLUDED
#define DPMATRIX_INCLUDED

#include "eval.h"
#include "seqpair.h"
#include "logsumexp.h"

struct Envelope {
  typedef long InputIndex;
  typedef long OutputIndex;

  InputIndex inLen;
  OutputIndex outLen;
  vguard<InputIndex> inStart, inEnd;

  inline bool contains (InputIndex x, OutputIndex y) const {
    return y >= 0 && y <= outLen && x >= inStart[y] && x < inEnd[y];
  }

  vguard<long long> offsets() const;
  
  Envelope (InputIndex x, OutputIndex y)
    : inLen (x), outLen (y), inStart (y + 1, 0), inEnd (y + 1, x + 2)
  { }
};

class DPMatrix {
public:
  typedef Envelope::InputIndex InputIndex;
  typedef Envelope::OutputIndex OutputIndex;

private:
  typedef long long CellIndex;

  vguard<double> cellStorage;

  inline CellIndex nCells() const {
    return nStates * (inLen + 1) * (outLen + 1);
  }

  inline CellIndex cellIndex (InputIndex inPos, OutputIndex outPos, StateIndex state) const {
    return (inPos * (outLen + 1) + outPos) * nStates + state;
  }

protected:
  inline void accumulate (double& ll, const EvaluatedMachineState::InOutStateTransMap& inOutStateTransMap, InputToken inTok, OutputToken outTok, InputIndex inPos, OutputIndex outPos, function<double(double,double)> reduce) const {
    auto visit = [&] (StateIndex, EvaluatedMachineState::TransIndex, double t) { ll = reduce(ll,t); };
    iterate (inOutStateTransMap, inTok, outTok, inPos, outPos, visit);
  }

  inline void iterate (const EvaluatedMachineState::InOutStateTransMap& inOutStateTransMap, InputToken inTok, OutputToken outTok, InputIndex inPos, OutputIndex outPos, function<void(StateIndex,EvaluatedMachineState::TransIndex,double)> visit) const {
    if (inOutStateTransMap.count (inTok)) {
      const EvaluatedMachineState::OutStateTransMap& outStateTransMap = inOutStateTransMap.at (inTok);
      if (outStateTransMap.count (outTok))
	for (const auto& st: outStateTransMap.at (outTok)) {
	  const EvaluatedMachineState::Trans& trans = st.second;
	  visit (st.first, trans.transIndex, cell(inPos,outPos,st.first) + trans.logWeight);
	}
    }
  }

  static inline double sum_reduce (double x, double y) { return log_sum_exp(x,y); }
  static inline double max_reduce (double x, double y) { return max(x,y); }

public:
  const EvaluatedMachine& machine;
  const SeqPair& seqPair;
  const vguard<InputToken> input;
  const vguard<OutputToken> output;
  const InputIndex inLen;
  const OutputIndex outLen;
  const StateIndex nStates;

  DPMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair);

  void writeJson (ostream& out) const;
  friend ostream& operator<< (ostream&, const DPMatrix&);
  
  inline double& cell (InputIndex inPos, OutputIndex outPos, StateIndex state) { return cellStorage[cellIndex(inPos,outPos,state)]; }
  inline const double cell (InputIndex inPos, OutputIndex outPos, StateIndex state) const { return cellStorage[cellIndex(inPos,outPos,state)]; }
};

#endif /* DPMATRIX_INCLUDED */
