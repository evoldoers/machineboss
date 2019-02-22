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
  for (auto t: generatorPath.trans) {
    const MachineState gms = generator.state[gState];
    Assert (t < gms.trans.size(), "Generator transition out of range");
    const MachineTransition gmt = gms.getTransition (t);
    if (!gmt.outputEmpty())
      seq.push_back (gmt.out);
    gState = gmt.dest;
  }

  for (auto errorPath: errorPaths) {
    Assert (errorPath.start <= seq.size(), "Alignment start index out of range");
    InputIndex ePos = errorPath.start;
    StateIndex eState = error.startState();
    for (auto t: errorPath.trans) {
      const MachineState ems = error.state[eState];
      Assert (t < ems.trans.size(), "Error transition out of range");
      const MachineTransition emt = ems.getTransition (t);
      if (!emt.inputEmpty()) {
	Assert (ePos < seq.size(), "Alignment column out of range");
	Assert (emt.in == seq[ePos++], "Read/sequence mismatch");
      }
      eState = emt.dest;
    }
  }
}

