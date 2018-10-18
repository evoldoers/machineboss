#include "beam.h"

BeamSearchMatrix::SeqNode::SeqNode() :
  inTok (0),
  parent (NULL)
{ }

BeamSearchMatrix::SeqNode::SeqNode (SeqNodePtr p, InputToken t) :
  inTok (t),
  parent (p)
{ }

BeamSearchMatrix::BeamSearchMatrix (const EvaluatedMachine& machine, const vguard<OutputSymbol>& outSym, size_t bw) :
  machine (machine),
  output (machine.outputTokenizer.tokenize (outSym)),
  outLen (output.size()),
  nStates (machine.nStates()),
  inToks (machine.inputTokenizer.tok2sym.size()),
  beamWidth (bw)
{ }

vguard<InputSymbol> BeamSearchMatrix::getSeq (SeqNodePtr node) const {
  list<InputToken> result;
  for (; node; node = node->parent)
    result.push_front (node->inTok);
  return machine.inputTokenizer.detokenize (vguard<InputToken> (result.begin(), result.end()));
}

vguard<InputSymbol> BeamSearchMatrix::doBeamSearch() {
  // for outPos = 0 to outLen
  //  for dest = 0 to nStates-1
  //   for type in (ins,del,match,null)
  //    for inTok in inputTokens(type)
  //     for src in incoming(dest,inTok,outTok)
  //      for (seq,prob) in src
  //       cell(seq+inTok,outPos,dest) += prob * trans(src,inTok,outTok,dest)
  //  keep only top beamWidth elements of cell

  for (OutputIndex outPos = 0; outPos <= outLen; ++outPos)
    for (StateIndex dest = 0; dest < nStates; ++dest) {
    }
  
  
  return vguard<InputSymbol>();
}
