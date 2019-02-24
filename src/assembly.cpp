#include "assembly.h"
#include "forward.h"

void CompactMachinePath::readJson (const json& pj) {
  trans.reserve (pj.size());
  for (auto& t: pj)
    trans.push_back (t.get<TransIndex>());
}

void CompactMachinePath::writeJson (ostream& out) const {
  out << "[" << to_string_join (trans, ",") << "]";
}

MachinePath CompactMachinePath::toMachinePath (const Machine& m) const {
  MachinePath mp;
  StateIndex s = m.startState();
  for (auto& ti: trans) {
    const MachineState& ms = m.state[s];
    const MachineTransition& mt = ms.getTransition (ti);
    mp.trans.push_back (mt);
    s = mt.dest;
  }
  return mp;
}

CompactMachinePath CompactMachinePath::fromMachinePath (const MachinePath& mp, const Machine& m) {
  CompactMachinePath cmp;
  cmp.trans.reserve (mp.trans.size());
  StateIndex s = m.startState();
  for (auto& t: mp.trans) {
    const MachineState& ms = m.state[s];
    cmp.trans.push_back (ms.findTransition (t));
    s = t.dest;
  }
  return cmp;
}

void CompactLocalMachinePath::readJson (const json& pj) {
  start = pj.at("start").get<InputIndex>();
  len = pj.at("len").get<InputIndex>();
  CompactMachinePath::readJson (pj.at("path"));
}

void CompactLocalMachinePath::writeJson (ostream& out) const {
  out << "{\"start\":" << start << ",\"len\":" << len << ",\"path\":";
  CompactMachinePath::writeJson (out);
  out << "}";
}

void Assembly::readJson (const json& pj) {
  generator.readJson (pj.at("generator"));
  error.readJson (pj.at("error"));
  generatorPath.readJson (pj.at("sequence"));
  auto alignments = pj.at("alignments");
  errorPaths.reserve (alignments.size());
  for (auto& a: alignments) {
    errorPaths.push_back (CompactLocalMachinePath());
    errorPaths.back().readJson (a);
  }
  validate();
  evaluateMachines();
}

void Assembly::writeJson (ostream& out) const {
  out << "{\"generator\":" << endl;
  generator.writeJson (out);
  out << ",\"error\":" << endl;
  error.writeJson (out);
  out << ",\"sequence\":";
  generatorPath.writeJson (out);
  out << endl << ",\"alignments\":[" << endl;
  for (size_t n = 0; n < errorPaths.size(); ++n) {
    errorPaths[n].writeJson (out);
    if (n + 1 < errorPaths.size())
      out << ",";
    out << endl;
  }
  out << "]}" << endl;
}

void Assembly::validate() const {
  Assert (generator.inputEmpty(), "Generator has nonempty input alphabet");
  Assert (generator.hasNullPaddingStates(), "Generator does not have separate start & end states");

  StateIndex gState = generator.startState();
  size_t ngTrans = 0;
  for (auto t: generatorPath.trans) {
    ++ngTrans;
    const MachineState gms = generator.state[gState];
    Assert (t < gms.trans.size(), "Generator path transition %u, source state %u: Transition %u out of range", ngTrans, gState, t);
    gState = gms.getTransition(t).dest;
  }
  Assert (gState == generator.endState(), "Generator path finishes in state %u, not the generator machine's end state which is %u", gState, generator.endState());

  const auto seq = sequence();
  size_t nPath = 0;
  for (auto errorPath: errorPaths) {
    ++nPath;
    Assert (errorPath.start <= seq.size(), "Alignment %u: start position %u out of range for sequence length %u", nPath, errorPath.start, seq.size());
    InputIndex ePos = errorPath.start;
    StateIndex eState = error.startState();
    size_t neTrans = 0;
    for (auto t: errorPath.trans) {
      ++neTrans;
      const MachineState ems = error.state[eState];
      Assert (t < ems.trans.size(), "Alignment %u, transition %u, source state %u: Transition %u out of range", nPath, neTrans, eState, t);
      const MachineTransition emt = ems.getTransition (t);
      if (!emt.inputEmpty()) {
	Assert (ePos < seq.size(), "Alignment %u, transition %u, source state %u: Alignment passed end of sequence", nPath, neTrans, eState);
	Assert (emt.in == seq[ePos], "Alignment %u, transition %u, source state %u: Transition input label (%s) does not match sequence symbol (%s)", nPath, neTrans, eState, emt.in.c_str(), seq[ePos].c_str());
	++ePos;
      }
      eState = emt.dest;
    }
    Assert (eState == error.endState(), "Alignment %u: Alignment path finishes in state %u, not the error machine's end state which is %u", nPath, eState, error.endState());
    Assert (ePos - errorPath.start == errorPath.len, "Alignment %u: Alignment input sequence length is %lu instead of the expected %lu", nPath, ePos - errorPath.start, errorPath.len);
  }
}

void Assembly::evaluateMachines() {
  evalGen = EvaluatedMachine (generator, generator.getParamDefs (true));
  evalErr = EvaluatedMachine (error, error.getParamDefs (true));
}

LogProb Assembly::logProb() const {
  LogProb lp = 0;
  StateIndex gState = generator.startState();
  for (auto t: generatorPath.trans) {
    lp += evalGen.state[gState].logTransWeight[t];
    gState = generator.state[gState].getTransition(t).dest;
  }

  for (auto errorPath: errorPaths) {
    StateIndex eState = error.startState();
    for (auto t: errorPath.trans) {
      lp += evalErr.state[eState].logTransWeight[t];
      eState = error.state[eState].getTransition(t).dest;
    }
  }

  return lp;
}

void Assembly::resampleAnnotation (mt19937& rng) {
  SeqPair seqPair;
  seqPair.output.name = "assembly";
  seqPair.output.seq = sequence();
  const ForwardMatrix forward (evalGen, seqPair);
  const MachinePath fwdTrace = forward.samplePath (generator, rng);
  generatorPath = CompactMachinePath::fromMachinePath (fwdTrace, generator);

  // Check we didn't change the sequence
  // Should be unnecessary, let's do it anyway
  const auto newSeq = sequence();
  Assert (equal (newSeq.begin(), newSeq.end(), seqPair.output.seq.begin()), "Sequence changed during annotation resampling move");
}

void Assembly::resampleAlignment (mt19937& rng, size_t maxAlignSlideWidth) {
  if (errorPaths.size()) {
    uniform_int_distribution<size_t> pAlignNum (0, errorPaths.size() - 1);
    const size_t nAlign = pAlignNum (rng);
    resampleIdentifiedAlignment (rng, nAlign, maxAlignSlideWidth);
  }
}

void Assembly::resampleIdentifiedAlignment (mt19937& rng, size_t nAlign, size_t maxAlignSlideWidth) {
  CompactLocalMachinePath& errorPath = errorPaths[nAlign];
  const MachinePath machineErrorPath = errorPath.toMachinePath (error);

  auto inSeq = machineErrorPath.inputSequence();
  auto outSeq = machineErrorPath.outputSequence();
  auto align = machineErrorPath.alignment();
  
  // Check that inSeq == localSeq
  // Should be unnecessary, let's do it anyway
  auto globalSeq = sequence();
  vguard<InputSymbol> localSeq (globalSeq.begin() + errorPath.start,
				globalSeq.begin() + errorPath.start + inSeq.size());
  Assert (localSeq.size() == inSeq.size() && equal (localSeq.begin(), localSeq.end(), inSeq.begin()), "Resampling alignment: error input sequence does not match generator output sequence");

  // OK, now create the seqPair and do the Forward traceback
  SeqPair seqPair;
  seqPair.input.name = "assembly";
  seqPair.input.seq = inSeq;
  seqPair.output.name = "read";
  seqPair.output.seq = outSeq;
  seqPair.alignment = align;
  
  const Envelope envelope (seqPair, maxAlignSlideWidth);

  const ForwardMatrix forward (evalErr, seqPair, envelope);
  const MachinePath fwdTrace = forward.samplePath (error, rng);

  // strictly speaking we should check the Forward likelihood in the other direction here,
  // and reject the move if it doesn't match. Instead...
  // Let's do a cast-assignment hack and exit.
  ((CompactMachinePath&)errorPath) = CompactMachinePath::fromMachinePath (fwdTrace, error);
}

size_t Assembly::nSequenceMoves (size_t maxResampledTransitions, const CompactMachinePath& gPath) {
  const size_t pathLen = gPath.trans.size();
  return (2*pathLen + 2 - maxResampledTransitions) * (maxResampledTransitions + 1) / 2;
}

Assembly::InputIndex Assembly::getOutputSeqLen (const MachinePath& mp, size_t transLen) {
  InputIndex len = 0;
  for (auto& t: mp.trans) {
    if (!transLen)
      break;
    --transLen;
    if (!t.outputEmpty())
      ++len;
  }
  return len;
}

vguard<OutputSymbol> Assembly::sequence() const {
  return getOutputSeq (generatorPath.toMachinePath (generator), 0, generatorPath.trans.size());
}

vguard<OutputSymbol> Assembly::getOutputSeq (const MachinePath& mp, size_t transStart, size_t transLen) {
  vguard<OutputSymbol> seq;
  seq.reserve (transLen);
  TransList::const_iterator iter = mp.trans.begin();
  advance (iter, transStart);
  for (size_t ti = 0; ti < transLen; ++ti) {
    if (!(*iter).outputEmpty())
      seq.push_back ((*iter).out);
    ++iter;
  }
  return seq;
}

StateIndex Assembly::getPathState (const MachinePath& mp, size_t nTrans) {
  StateIndex s = 0;
  for (TransList::const_iterator iter = mp.trans.begin(); iter != mp.trans.end() && nTrans > 0; ++iter, --nTrans)
    s = (*iter).dest;
  return s;
}

void Assembly::resampleSequence (mt19937& rng, size_t maxResampledTransitions) {
  // pick a section of the generator path to resample
  const size_t z12_old = nSequenceMoves (maxResampledTransitions, generatorPath);
  uniform_int_distribution<size_t> pGenTrans (0, z12_old - 1);
  const size_t rvGenTrans = pGenTrans (rng);
  size_t oldGenTransLen = 0;
  while (rvGenTrans > nSequenceMoves (oldGenTransLen, generatorPath))
    ++oldGenTransLen;
  const size_t oldGenTransStart = rvGenTrans - nSequenceMoves (oldGenTransLen - 1, generatorPath);
  const MachinePath oldGenPath = generatorPath.toMachinePath (generator);
  const InputIndex oldGenStart = getOutputSeqLen (oldGenPath, oldGenTransStart);
  const vguard<OutputSymbol> oldGenSeq = getOutputSeq (oldGenPath, oldGenTransStart, oldGenTransLen);
  const InputIndex oldGenLen = oldGenSeq.size();
  const StateIndex oldGenStartState = getPathState (oldGenPath, oldGenTransStart);
  const StateIndex oldGenEndState = getPathState (oldGenPath, oldGenTransStart + oldGenTransLen);

  // propose new path by stochastic Forward traceback, constraining end & start states

  // TODO: need to implement general traceback by matrix inversion
  // If t(i,j) is the transition matrix, we need S = \sum_{n=0}^\infty t^n = (1-t)^{-1}
  
  // loop through all alignments testing for overlap
  // propose new alignment by stochastic Forward traceback, constraining end & start states
}
