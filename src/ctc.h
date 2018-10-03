#ifndef CTC_INCLUDED
#define CTC_INCLUDED

#include <deque>
#include <queue>
#include "dpmatrix.h"

struct PrefixTree {
  typedef Envelope::OutputIndex OutputIndex;
  struct Node {
    const InputToken inTok;
    const Node* parent;
    vguard<Node*> child;
    vguard<double> cellStorage;
    double logPrefixProb;
    Node (const PrefixTree& tree, const Node* parent, InputToken inTok);
  };
  struct NodeComparator {
    bool operator() (const Node* x, const Node* y) const { return x->logPrefixProb > y->logPrefixProb; }
  };
  
  typedef deque<Node> NodeStorage;
  typedef priority_queue<PrefixTree::Node*,deque<PrefixTree::Node*>,PrefixTree::NodeComparator> NodePtrQueue;
  
  const EvaluatedMachine& machine;
  const vguard<OutputToken> output;
  const OutputIndex outLen;
  const StateIndex nStates;

  NodeStorage node;
  NodePtrQueue nodeQueue;
  Node* bestSeqNode;
  
  PrefixTree (const EvaluatedMachine& machine, const vguard<OutputToken>& output);

  double logSeqProb (const Node&) const;
  double logPrefixProb (const Node&) const;

  Node* bestPrefixNode() const;
  vguard<InputToken> bestSeq() const { return seqTraceback (bestSeqNode); }
  vguard<InputToken> bestPrefix() const { return seqTraceback (bestPrefixNode()); }
  vguard<InputToken> seqTraceback (const Node* node) const;

  inline size_t nCells() const {
    return 2 * (outLen + 1) * nStates;
  }

  inline size_t cellIndex (OutputIndex outPos, StateIndex state) const {
    return 2 * (outPos * state + outPos);
  }

  inline double& seqCell (const Node& inNode, OutputIndex outPos, StateIndex state) {
    return inNode.cellStorage [cellIndex (outPos, state)];
  }

  inline double& prefixCell (const Node& inNode, OutputIndex outPos, StateIndex state) {
    return inNode.cellStorage [cellIndex (outPos, state) + 1];
  }
};

#endif /* CTC_INCLUDED */
