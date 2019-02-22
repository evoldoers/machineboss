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
  CompactMachinePath::readJson (pj.at("path"));
}

void CompactLocalMachinePath::writeJson (ostream& out) const {
  out << "{\"start\":" << start << ",\"path\":";
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
  }
}

void Assembly::evaluateMachines() {
  evalGen = EvaluatedMachine (generator, generator.getParamDefs (true));
  evalErr = EvaluatedMachine (error, error.getParamDefs (true));
}

vguard<OutputSymbol> Assembly::sequence() const {
  vguard<OutputSymbol> seq;
  seq.reserve (generatorPath.trans.size());
  StateIndex gState = generator.startState();
  for (auto t: generatorPath.trans) {
    const MachineTransition gmt = generator.state[gState].getTransition (t);
    if (!gmt.outputEmpty())
      seq.push_back (gmt.out);
    gState = gmt.dest;
  }
  return seq;
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
  const auto newSeq = sequence();
  Assert (equal (newSeq.begin(), newSeq.end(), seqPair.output.seq.begin()), "Sequence changed during annotation resampling move");
}
