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

  typedef size_t SeqNodeIndex;
  struct SeqNode {
    const InputToken inTok;
    const SeqNodeIndex parent;
    list<SeqNodeIndex> child;
    SeqNode();
    SeqNode (SeqNodeIndex p, InputToken t);
  };

  typedef map<SeqNodeIndex,double> Cell;
  
  const EvaluatedMachine& machine;
  const vguard<OutputToken> output;
  const OutputIndex outLen;
  const StateIndex nStates;
  const size_t beamWidth;

  vguard<SeqNode> seqNodeStore;
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
  
  BeamSearchMatrix (const EvaluatedMachine& machine, const vguard<OutputSymbol>& outSym, size_t beamWidth);

  vguard<InputSymbol> doBeamSearch();
};

#endif /* BEAM_INCLUDED */
