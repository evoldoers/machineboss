#ifndef PHYLO_INTERSECT_INCLUDED
#define PHYLO_INTERSECT_INCLUDED

#include <string>
#include "machine.h"
#include "params.h"
#include "vguard.h"

namespace MachineBoss {

using namespace std;

// One node in a phylogenetic tree parsed from Newick. Trees may be of
// arbitrary topology: nodes can have any non-zero number of children
// (leaves have zero, internal nodes one or more).
struct PhyloNode {
  string name;
  bool hasBranchLength = false;
  double branchLength = 0;
  vguard<size_t> children;
  size_t parent = (size_t) -1;  // (size_t)-1 for the root
};

struct PhyloTree {
  vguard<PhyloNode> nodes;
  size_t root = 0;

  static PhyloTree parseNewick (const string& nwk);
  bool isRoot (size_t i) const { return nodes[i].parent == (size_t) -1; }
  bool isLeaf (size_t i) const { return nodes[i].children.empty(); }
};

// Build the phylogenetic intersection of branchTransducer T over the tree.
//
//   leaf v       -> wildEcho(T.outputAlphabet)
//   degree-1 v   -> compose(T_u, build(u))
//   degree-n v   -> intersect(intersect(... intersect(compose(T_{c1}, build(c1)),
//                                                    compose(T_{c2}, build(c2))) ...),
//                             compose(T_{cn}, build(cn)))
//                   where T_{ci} is a copy of T renamed so timeParam -> timeParam[<ci.name>]
//
// Polytomies (n > 2) fold-left into iterated intersection.
//
// If T has a parameter named timeParam, every non-root node must have a
// non-empty name unique within the tree. Otherwise no renaming is done and
// names are not required.
//
// branchLengthsOut: if non-null, populated with timeParam[<name>] -> branchLength
// for every non-root node that carries a branch length in the Newick.
Machine phyloIntersect (const Machine& branchTransducer,
                        const PhyloTree& tree,
                        const string& timeParam = "t",
                        ParamAssign* branchLengthsOut = NULL,
                        Machine::SilentCycleStrategy strategy = Machine::SumSilentCycles);

}  // end namespace

#endif /* PHYLO_INTERSECT_INCLUDED */
