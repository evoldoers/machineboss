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
  beamWidth (bw),
  cellStore (nCells())
{
  seqNodeStore.push_back (SeqNode());
  cell(0,0).logWeight[&seqNodeStore.front()] = 0;

  // for outPos = 0 to outLen
  //  for dest = 0 to nStates-1
  //   for type in (ins,del,match,null)
  //    for inTok in inputTokens(type)
  //     for src in incoming(dest,inTok,outTok)
  //      for (seq,prob) in src
  //       cell(seq+inTok,outPos,dest) += prob * trans(src,inTok,outTok,dest)
  //  keep only top beamWidth elements of cell

  ProgressLog(plogDP,5);
  plogDP.initProgress ("Performing beam-search (%lu cells)", nCells());
  for (OutputIndex outPos = 0; outPos <= outLen; ++outPos)
    for (StateIndex dest = 0; dest < nStates; ++dest) {
      plogDP.logProgress ((nStates * outPos + dest) / nCells(), "filled %lu cells", nStates * outPos + dest);
      Cell& destCell = cell (outPos, dest);
      const EvaluatedMachineState::InOutStateTransMap& inOutStateTransMap = machine.state[dest].incoming;
      for (const auto& tok_ostm: inOutStateTransMap) {
	const InputToken inTok = tok_ostm.first;
	if (inTok) {
	  const EvaluatedMachineState::OutStateTransMap& outStateTransMap = tok_ostm.second;
	  if (outPos > 0)
	    accumulate (destCell, outStateTransMap, inTok, output[outPos-1], outPos - 1);
	  accumulate (destCell, outStateTransMap, inTok, 0, outPos);
	}
      }
      if (inOutStateTransMap.count (0)) {
	const EvaluatedMachineState::OutStateTransMap& outStateTransMap = inOutStateTransMap.at(0);
	  if (outPos > 0)
	    accumulate (destCell, outStateTransMap, 0, output[outPos-1], outPos - 1);
	  accumulate (destCell, outStateTransMap, 0, 0, outPos);
      }
      if (destCell.logWeight.size() > beamWidth) {
	vguard<SeqNodePtr> seqs = extract_keys (destCell.logWeight);
	partial_sort (seqs.begin(), seqs.begin() + beamWidth, seqs.end(), destCell);
	Cell::LogWeightMap lw;
	for (size_t n = 0; n < beamWidth; ++n) {
	  const SeqNodePtr node = seqs[n];
	  lw[node] = destCell.logWeight.at (node);
	}
	destCell.logWeight.swap (lw);
      }
      if (LoggingThisAt(6)) {
	vguard<string> seqs;
	for (const auto& seq_lw: destCell.logWeight)
	  seqs.push_back (string(join(getSeq(seq_lw.first),"")) + "(" + to_string(seq_lw.second) + ")");
	clog << "Cell (" << outPos << "," << dest << "): " << to_string_join(seqs) << endl;
      }
    }
}

vguard<InputSymbol> BeamSearchMatrix::bestSeq() {
  const Cell& finalCell = cell (outLen, nStates - 1);
  pair<SeqNodePtr,LogWeight> best (NULL, -numeric_limits<double>::infinity());
  for (const auto& seq_lw: finalCell.logWeight)
    if (seq_lw.second > best.second)
      best = seq_lw;
  return getSeq (best.first);
}

vguard<InputSymbol> BeamSearchMatrix::getSeq (SeqNodePtr node) const {
  list<InputToken> result;
  for (; node && node->inTok; node = node->parent)
    result.push_front (node->inTok);
  return machine.inputTokenizer.detokenize (vguard<InputToken> (result.begin(), result.end()));
}
