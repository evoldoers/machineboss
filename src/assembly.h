#ifndef ASSEMBLY_INCLUDED
#define ASSEMBLY_INCLUDED

#include "eval.h"
#include "seqpair.h"

struct CompactMachinePath {
  typedef EvaluatedMachineState::TransIndex TransIndex;
  vguard<TransIndex> trans;

  void readJson (const json&);
  void writeJson (ostream&) const;
};

struct CompactLocalMachinePath : CompactMachinePath {
  typedef Envelope::InputIndex InputIndex;
  InputIndex start;

  void readJson (const json&);
  void writeJson (ostream&) const;
};

struct Assembly {
  typedef CompactMachinePath::TransIndex TransIndex;
  typedef CompactLocalMachinePath::InputIndex InputIndex;

  Machine generator, error;
  CompactMachinePath generatorPath;
  vguard<CompactLocalMachinePath> errorPaths;

  void readJson (const json&);
  void writeJson (ostream&) const;

  void validate() const;
};

#endif /* ASSEMBLY_INCLUDED */
