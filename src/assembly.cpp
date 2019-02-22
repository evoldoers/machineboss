#include "assembly.h"

void CompactMachinePath::readJson (const json& pj) {
  trans.reserve (pj.size());
  for (auto& t: pj)
    trans.push_back (t.get<TransIndex>());
}

void CompactMachinePath::writeJson (ostream& out) const {
  out << "[" << to_string_join (trans, ",") << "]";
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

  vguard<InputSymbol> seq;
  seq.reserve (generatorPath.trans.size());
  StateIndex gState = generator.startState();
  size_t ngTrans = 0;
  for (auto t: generatorPath.trans) {
    ++ngTrans;
    const MachineState gms = generator.state[gState];
    Assert (t < gms.trans.size(), "Generator path transition %u, source state %u: Transition %u out of range", ngTrans, gState, t);
    const MachineTransition gmt = gms.getTransition (t);
    if (!gmt.outputEmpty())
      seq.push_back (gmt.out);
    gState = gmt.dest;
  }
  Assert (gState == generator.endState(), "Generator path finishes in state %u, not the generator machine's end state which is %u", gState, generator.endState());

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

