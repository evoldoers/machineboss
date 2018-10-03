#ifndef CTC_INCLUDED
#define CTC_INCLUDED

#include <deque>
#include <queue>
#include "dpmatrix.h"

struct PrefixTree {
  typedef Envelope::OutputIndex OutputIndex;
  struct Node {
    const Node* parent;
    const InputToken inTok;
    vguard<NodeIndex> child;
    vguard<double> cellStorage;
    double logPrefixProb, logSeqProb;
    Node (const PrefixTree& tree, const Node* parent, InputToken inTok);
  };
  struct NodeComparator {
    bool operator() (const Node& x, const Node& y) const { return x.logPrefixProb > y.logPrefixProb; }
  };
  
  typedef deque<Node> NodeStorage;
  typedef priority_queue<PrefixTree::Node*,deque<PrefixTree::Node*>,PrefixTree::NodeComparator> NodePtrQueue;
  
  const EvaluatedMachine& machine;
  const vguard<OutputToken> output;
  const OutputIndex outLen;
  const StateIndex nStates;

  NodeStorage node;
  NodePtrQueue nodeQueue;
  Node *bestPrefix, *bestSeq;
  
  PrefixTree (const EvaluatedMachine& machine, const vguard<OutputToken>& output);

  double logSeqProb (NodeIndex) const;
  double logPrefixProb (NodeIndex) const;

  inline size_t nCells() const {
    return 2 * (outLen + 1) * nStates;
  }

  inline size_t cellIndex (OutputIndex outPos, StateIndex state) const {
    return 2 * (outPos * state + outPos);
  }

  inline double& seqCell (NodeIndex inNodeIdx, OutputIndex outPos, StateIndex state) {
    return node[n].cellStorage [cellIndex (outPos, state)];
  }

  inline double& prefixCell (NodeIndex inNodeIdx, OutputIndex outPos, StateIndex state) {
    return node[n].cellStorage [cellIndex (outPos, state) + 1];
  }
};

#endif /* CTC_INCLUDED */
