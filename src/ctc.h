#ifndef CTC_INCLUDED
#define CTC_INCLUDED

#include <list>
#include <algorithm>
#include <random>

#include "dpmatrix.h"
#include "logger.h"

// If G is a generator with output alphabet A, then
// prefix(G) = (echo(A) + wild(A)) * G is its prefix machine.
// (Here + is concatenation, * is composition, echo(A) copies A-symbols from input to output, and wild(A) emits A-symbols.)
// The PrefixTree uses this construction implicitly:
// seqCell(outPos,state) and prefixCell(outPos,state) track states in, respectively, echo(A)*G and wild(A)*G.
struct PrefixTree {
  typedef Envelope::InputIndex InputIndex;
  typedef Envelope::OutputIndex OutputIndex;
  struct Node {
    const InputToken inTok;
    const Node* parent;
    const StateIndex nStates;
    const OutputIndex outLen;
    const InputIndex length;
    vguard<double> cellStorage;
    double logPrefixProb;
    bool extended;
    list<Node*> child;
    list<Node>::iterator iter;
    list<Node*>::iterator parentIter;
    
    Node();
    Node (const PrefixTree& tree, const Node* parent, InputToken inTok);

    void fill (const PrefixTree& tree);
    double logSeqProb() const;

    inline void accumulateSeqCell (double& ll, const EvaluatedMachineState::OutStateTransMap& outStateTransMap, const Node& inNode, OutputToken outTok, OutputIndex outPos) const {
      if (outStateTransMap.count (outTok))
	for (const auto& st: outStateTransMap.at (outTok)) {
	  const EvaluatedMachineState::Trans& trans = st.second;
	  LogThisAt(9,"seqCell("<<outPos<<",d) logsum+= "<<inNode.seqCell(outPos,st.first)<<" + "<<trans.logWeight<<" ("<<st.first<<"->d)"<<endl);
	  log_accum_exp (ll, inNode.seqCell(outPos,st.first) + trans.logWeight);
	}
    }

    inline size_t nCells() const {
      return 2 * (outLen + 1) * nStates;
    }

    inline size_t seqCellIndex (OutputIndex outPos, StateIndex state) const {
      return 2 * (outPos * nStates + state);
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

    vguard<InputToken> traceback() const;
    Node* randomChild (mt19937&) const;
  };
  struct NodeComparator {
    bool operator() (const Node* x, const Node* y) const { return x->logPrefixProb < y->logPrefixProb; }
  };

  typedef list<Node> NodeStorage;
  typedef vector<PrefixTree::Node*> NodePtrQueue;

  const EvaluatedMachine& machine;
  const vguard<vguard<LogWeight> > logSumInTrans;
  const vguard<OutputToken> output;
  const OutputIndex outLen;
  const StateIndex nStates;
  const InputIndex maxBacktrack;

  NodeStorage nodeStore;
  NodePtrQueue nodeQueue;
  NodeComparator nodeComparator;
  Node* bestSeqNode;
  double bestLogSeqProb;
  InputIndex maxPrefixLen;
  
  PrefixTree (const EvaluatedMachine& machine, const vguard<OutputSymbol>& outSym, InputIndex maxBacktrack);
  void clear();
  
  vguard<InputSymbol> doPrefixSearch();  // finds most likely input
  vguard<InputSymbol> doAnnealedSearch (mt19937&, int stepsPerTok, bool doCooling = true);

  vguard<InputSymbol> sampleSeq (mt19937&);  // samples from posterior distribution over inputs
  vguard<InputToken> sampleTokSeq (mt19937&);

  double logSeqProb (const list<InputToken>&, bool humble = false);

  Node* rootNode();
  void extendNode (Node* parent);
  Node* addNode (Node* parent, InputToken inTok, bool humble = false);
  void removeNode (Node* node);
  Node* bestPrefixNode() const { return nodeQueue.front(); }
  string nodeQueueDebugString() const;
  
  vguard<InputSymbol> bestSeq() const { return seqTraceback (bestSeqNode); }
  vguard<InputSymbol> bestPrefix() const { return seqTraceback (bestPrefixNode()); }
  vguard<InputSymbol> seqTraceback (const Node* node) const;
};

#endif /* CTC_INCLUDED */
