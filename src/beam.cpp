#include "beam.h"

BeamSearchMatrix::SeqNode::SeqNode() :
  inTok (0),
  parent (0)
{ }

BeamSearchMatrix::SeqNode::SeqNode (SeqNodeIndex p, InputToken t) :
  inTok (t),
  parent (p)
{ }

BeamSearchMatrix::BeamSearchMatrix (const EvaluatedMachine& machine, const vguard<OutputSymbol>& outSym, size_t bw) :
  machine (machine),
  output (machine.outputTokenizer.tokenize (outSym)),
  outLen (output.size()),
  nStates (machine.nStates()),
  beamWidth (bw)
{ }

