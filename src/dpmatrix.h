#ifndef DPMATRIX_INCLUDED
#define DPMATRIX_INCLUDED

#include <random>
#include "eval.h"
#include "seqpair.h"
#include "logsumexp.h"

struct IndexMapperBase {
  typedef typename Envelope::InputIndex InputIndex;
  typedef typename Envelope::OutputIndex OutputIndex;
};

struct IdentityIndexMapper : IndexMapperBase {
  static inline InputIndex mappedInput (InputIndex i) { return i; }
  static inline OutputIndex mappedOutput (OutputIndex j) { return j; }
};

struct RollingInputIndexMapper : IndexMapperBase {
  static inline InputIndex mappedInput (InputIndex i) { return i % 2; }
  static inline OutputIndex mappedOutput (OutputIndex j) { return j; }
};

struct RollingOutputIndexMapper : IndexMapperBase {
  static inline InputIndex mappedInput (InputIndex i) { return i; }
  static inline OutputIndex mappedOutput (OutputIndex j) { return j % 2; }
};

template<class IndexMapper>
class DPMatrix : protected IndexMapper {
public:
  typedef typename IndexMapper::InputIndex InputIndex;
  typedef typename IndexMapper::OutputIndex OutputIndex;

  typedef function<double(double,double)> Reducer;
  typedef function<bool(InputIndex,OutputIndex,StateIndex,EvaluatedMachineState::TransIndex)> TraceTerminator;
  typedef function<void(StateIndex,EvaluatedMachineState::TransIndex,double)> TransVisitor;
  typedef function<size_t(const vguard<double>&)> TransSelector;

protected:
  typedef Envelope::Offset CellIndex;
  vguard<CellIndex> offsets;
  
  inline CellIndex nCells() const {
    return nStates * offsets.back();
  }

private:
  vguard<double> cellStorage;

  inline CellIndex cellIndex (InputIndex inPos, OutputIndex outPos, StateIndex state) const {
#ifdef USE_VECTOR_GUARDS
    if (!env.contains (inPos, outPos))
      throw runtime_error ("Envelope out-of-bounds access error");
#endif
    const OutputIndex mappedOutPos = IndexMapper::mappedOutput(outPos);
    return (offsets[mappedOutPos] + IndexMapper::mappedInput(inPos) - env.inStart[mappedOutPos]) * nStates + state;
  }

  void alloc();
  
protected:
  inline void accumulate (double& ll, const EvaluatedMachineState::InOutStateTransMap& inOutStateTransMap, InputToken inTok, OutputToken outTok, InputIndex inPos, OutputIndex outPos, Reducer reduce) const {
    auto visit = [&] (StateIndex, EvaluatedMachineState::TransIndex, double t) { ll = reduce(ll,t); };
    iterate (inOutStateTransMap, inTok, outTok, inPos, outPos, visit);
  }

  inline void iterate (const EvaluatedMachineState::InOutStateTransMap& inOutStateTransMap, InputToken inTok, OutputToken outTok, InputIndex inPos, OutputIndex outPos, TransVisitor visit) const {
    if (inOutStateTransMap.count (inTok)) {
      const EvaluatedMachineState::OutStateTransMap& outStateTransMap = inOutStateTransMap.at (inTok);
      if (outStateTransMap.count (outTok))
	for (const auto& st: outStateTransMap.at (outTok)) {
	  const EvaluatedMachineState::Trans& trans = st.second;
	  visit (st.first, trans.transIndex, cell(inPos,outPos,st.first) + trans.logWeight);
	}
    }
  }
  
  inline void pathIterate (TransVisitor visit, const EvaluatedMachineState::InOutStateTransMap& inOutStateTransMap, InputToken inTok, OutputToken outTok, InputIndex inPos, OutputIndex outPos) const {
    iterate (inOutStateTransMap, inTok, outTok, inPos, outPos, visit);
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
  const Envelope env;

  DPMatrix (const EvaluatedMachine&, const SeqPair&);
  DPMatrix (const EvaluatedMachine&, const SeqPair&, const Envelope&);

  void writeJson (ostream& out) const;
  
  inline double& cell (InputIndex inPos, OutputIndex outPos, StateIndex state) {
    return cellStorage[cellIndex(inPos,outPos,state)];
  }

  inline const double cell (InputIndex inPos, OutputIndex outPos, StateIndex state) const {
    return env.contains(inPos,outPos) ? cellStorage[cellIndex(inPos,outPos,state)] : -numeric_limits<double>::infinity();
  }

  double startCell() const { return cell (0, 0, machine.startState()); }
  double endCell() const { return cell (inLen, outLen, machine.endState()); }

  static TransVisitor addTransToTraceOptions (vguard<StateIndex>&, vguard<EvaluatedMachineState::TransIndex>&, vguard<double>&);
  static size_t selectMaxTrans (const vguard<double>&);
  static TransSelector randomTransSelector (mt19937&);
  
  MachinePath traceBack (const Machine& m, TransSelector ts = DPMatrix::selectMaxTrans) const;
  MachinePath traceBack (const Machine& m, StateIndex s, TransSelector ts = DPMatrix::selectMaxTrans) const;
  MachinePath traceBack (const Machine& m, InputIndex inPos, OutputIndex outPos, StateIndex s, TransSelector ts = DPMatrix::selectMaxTrans) const;
  void traceBack (const Machine& m, InputIndex inPos, OutputIndex outPos, StateIndex s, TraceTerminator stopTrace, TransSelector ts = DPMatrix::selectMaxTrans) const;

  MachinePath traceForward (const Machine& m, TransSelector ts = DPMatrix::selectMaxTrans) const;
  MachinePath traceForward (const Machine& m, InputIndex inPos, OutputIndex outPos, StateIndex s, TransSelector ts = DPMatrix::selectMaxTrans) const;
  void traceForward (const Machine& m, InputIndex inPos, OutputIndex outPos, StateIndex s, TraceTerminator stopTrace, TransSelector ts = DPMatrix::selectMaxTrans) const;
};

#include "dpmatrix_defs.h"

#endif /* DPMATRIX_INCLUDED */
