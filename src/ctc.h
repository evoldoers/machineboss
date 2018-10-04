#ifndef CTC_INCLUDED
#define CTC_INCLUDED

#include <list>
#include <deque>
#include <queue>

#include "dpmatrix.h"

struct PrefixTree {
  typedef Envelope::OutputIndex OutputIndex;
  struct Node {
    const InputToken inTok;
    const Node* parent;
    const StateIndex nStates;
    const OutputIndex outLen;
    vguard<Node*> child;
    vguard<double> cellStorage;
    double logPrefixProb;

    Node();
    Node (const PrefixTree& tree, const Node* parent, InputToken inTok);

    void fill (const PrefixTree& tree);
    double logSeqProb() const;

    inline void accumulateSeqCell (double& ll, const EvaluatedMachineState::OutStateTransMap& outStateTransMap, const Node& inNode, OutputToken outTok, OutputIndex outPos) const {
      if (outStateTransMap.count (outTok))
	for (const auto& st: outStateTransMap.at (outTok)) {
	  const EvaluatedMachineState::Trans& trans = st.second;
	  log_accum_exp (ll, inNode.seqCell(outPos,st.first) + trans.logWeight);
	}
    }

    inline size_t nCells() const {
      return 2 * (outLen + 1) * nStates;
    }

    inline size_t seqCellIndex (OutputIndex outPos, StateIndex state) const {
      return 2 * (outLen * state + outPos);
    }

    inline size_t prefixCellIndex (OutputIndex outPos, StateIndex state) const {
      return seqCellIndex (outPos, state) + 1;
    }

    inline double seqCell (OutputIndex outPos, StateIndex state) const {
      return cellStorage [seqCellIndex (outPos, state)];
    }

    inline double& seqCell (OutputIndex outPos, StateIndex state) {
      return cellStorage [seqCellIndex (outPos, state)];
    }

    inline double prefixCell (OutputIndex outPos, StateIndex state) const {
      return cellStorage [prefixCellIndex (outPos, state)];
    }

    inline double& prefixCell (OutputIndex outPos, StateIndex state) {
      return cellStorage [prefixCellIndex (outPos, state)];
    }
  };
  struct NodeComparator {
    bool operator() (const Node* x, const Node* y) const { return x->logPrefixProb < y->logPrefixProb; }
  };

  typedef list<Node> NodeStorage;
  typedef priority_queue<PrefixTree::Node*,deque<PrefixTree::Node*>,PrefixTree::NodeComparator> NodePtrQueue;

  const EvaluatedMachine& machine;
  const vguard<vguard<LogWeight> > sumInTrans;
  const vguard<OutputToken> output;
  const OutputIndex outLen;
  const StateIndex nStates;

  NodeStorage nodeStore;
  NodePtrQueue nodeQueue;
  Node* bestSeqNode;
  double bestLogSeqProb;
  
  PrefixTree (const EvaluatedMachine& machine, const vguard<OutputSymbol>& outSym);

  Node* addNode (const Node* parent, InputToken inTok);
  Node* bestPrefixNode() const { return nodeQueue.top(); }

  vguard<InputSymbol> bestSeq() const { return seqTraceback (bestSeqNode); }
  vguard<InputSymbol> bestPrefix() const { return seqTraceback (bestPrefixNode()); }
  vguard<InputSymbol> seqTraceback (const Node* node) const;
};

#endif /* CTC_INCLUDED */
