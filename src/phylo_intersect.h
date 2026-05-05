#ifndef PHYLO_INTERSECT_INCLUDED
#define PHYLO_INTERSECT_INCLUDED

#include <string>
#include "machine.h"
#include "params.h"
#include "vguard.h"

namespace MachineBoss {

using namespace std;

// One node in a binary phylogenetic tree parsed from Newick.
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
//   internal v   -> intersect(compose(T_u, build(u)), compose(T_w, build(w)))
//                   where T_u, T_w are copies of T renamed so timeParam -> timeParam[<name>]
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
