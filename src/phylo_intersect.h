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
//
// leafClamps: if non-null, supplies an observed sequence (as a list of symbol
// strings, e.g. {"A","C","G"}) for one or more leaf nodes by name. For each
// clamped leaf, the leaf machine becomes a recognizer of that sequence
// instead of wildEcho — giving an output-empty machine whose Forward score
// equals the marginal likelihood of the observed leaves under the model.
// Useful as an independent verification target for the multidim Forward DP
// emitted by --codegen-rust.
//
// felsenstein: if true (default), call hoistSharedSubexpressions on the
// composed machine to factor shared sub-expressions in transition weights
// into named entries in funcs.defs (Felsenstein-style pruning). Set false
// to keep the older transition-exploded form, e.g. for cross-checking.
//
// skeleton: if true, perform the recursive intersect/compose on a unary-
// alphabet "skeleton" of T — every non-empty input/output symbol replaced
// with the placeholder "*" and every emit weight replaced with 1. The
// returned machine has the correct state-and-transition topology of the
// full phylo composition (each of the M_skel emit transitions is the
// stand-in for an entire |Σ|^k column-emission family of the full M),
// but does NOT carry the per-symbol substitution weights — it is intended
// as a fast structural pass for inspection or as the first half of a two-
// stage pipeline whose second stage (per-column symbol expansion via
// Felsenstein pruning) will be added later. Use --phylo-skeleton on the
// CLI. leafClamps in skeleton mode clamp leaf length only (each observed
// symbol is replaced by "*"); felsenstein is ignored when skeleton is
// true (no per-symbol weights to hoist).
Machine phyloIntersect (const Machine& branchTransducer,
                        const PhyloTree& tree,
                        const string& timeParam = "t",
                        ParamAssign* branchLengthsOut = NULL,
                        Machine::SilentCycleStrategy strategy = Machine::SumSilentCycles,
                        const map<string, vguard<string> >* leafClamps = NULL,
                        bool felsenstein = true,
                        bool skeleton = false);

}  // end namespace

#endif /* PHYLO_INTERSECT_INCLUDED */
