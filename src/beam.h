#ifndef BEAM_INCLUDED
#define BEAM_INCLUDED

#include <list>
#include <algorithm>
#include <random>

#include "dpmatrix.h"
#include "logger.h"

struct BeamSearchMatrix {
  typedef Envelope::InputIndex InputIndex;
  typedef Envelope::OutputIndex OutputIndex;

  struct SeqNode {
    typedef SeqNode* SeqNodePtr;
    const InputToken inTok;
    const SeqNodePtr parent;
    vguard<SeqNodePtr> child;
    SeqNode();
    SeqNode (SeqNodePtr p, InputToken t);
  };

  typedef SeqNode::SeqNodePtr SeqNodePtr;
  struct Cell {
    typedef map<SeqNodePtr,double> LogWeightMap;
    LogWeightMap logWeight;
    bool operator() (SeqNodePtr a, SeqNodePtr b) const {
      return logWeight.at(a) > logWeight.at(b);
    }
  };
  
  const EvaluatedMachine& machine;
  const vguard<OutputToken> output;
  const OutputIndex outLen;
  const StateIndex nStates;
  const InputToken inToks;
  const size_t beamWidth;

  list<SeqNode> seqNodeStore;
  vguard<Cell> cellStore;

  inline size_t nCells() const {
    return (outLen + 1) * nStates;
  }
  
  inline size_t cellIndex (OutputIndex outPos, StateIndex state) const {
    return outPos * nStates + state;
  }

  inline Cell& cell (OutputIndex outPos, StateIndex state) {
    return cellStore [cellIndex (outPos, state)];
  }

  inline const Cell& cell (OutputIndex outPos, StateIndex state) const {
    return cellStore [cellIndex (outPos, state)];
  }

  inline void accumulate (Cell& destCell, const EvaluatedMachineState::OutStateTransMap& outStateTransMap, InputToken inTok, OutputToken outTok, OutputIndex outPos) {
    if (outStateTransMap.count (outTok)) {
      for (const auto& st: outStateTransMap.at (outTok)) {
	const EvaluatedMachineState::Trans& trans = st.second;
	const Cell& srcCell = cell (outPos, st.first);
	for (const auto& seq_lw: srcCell.logWeight) {
	  const SeqNodePtr node = extendSeq (seq_lw.first, inTok);
	  const LogWeight lw = seq_lw.second + trans.logWeight;
	  destCell.logWeight[node] = destCell.logWeight.count(node) ? log_sum_exp (destCell.logWeight.at(node), lw) : lw;
	}
      }
    }
  }

  inline SeqNodePtr extendSeq (SeqNodePtr node, InputToken inTok) {
    if (node->child.empty())
      node->child.insert (node->child.end(), inToks, NULL);
    if (!node->child[inTok]) {
      seqNodeStore.push_back (SeqNode (node, inTok));
      node->child[inTok] = &seqNodeStore.back();
    }
    return node->child[inTok];
  }

  BeamSearchMatrix (const EvaluatedMachine& machine, const vguard<OutputSymbol>& outSym, size_t beamWidth);

  vguard<InputSymbol> bestSeq();
  vguard<InputSymbol> getSeq (SeqNodePtr) const;
};

#endif /* BEAM_INCLUDED */
