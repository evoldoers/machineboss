#ifndef FELSENSTEIN_INCLUDED
#define FELSENSTEIN_INCLUDED

#include "machine.h"

namespace MachineBoss {

// Hoist shared sub-expressions in a machine's transition weights into
// auto-named entries in `funcs.defs`. The result is functionally
// equivalent to `m` but typically has dramatically smaller transition
// weight expressions.
//
// This is the same idea as Felsenstein pruning in phylogenetics:
// gather the intermediate sum-products over child-state transitions
// once per unique sub-expression, store them as named intermediates,
// and have transitions reference them by name instead of repeating
// the same sub-expression at every transition that uses it.
//
// `minNodesToHoist`: only hoist sub-expressions whose tree size is at
// least this many internal nodes; tiny sub-expressions aren't worth
// hoisting (the def name is bigger than the inlined expression).
//
// `minRefsToHoist`: only hoist sub-expressions referenced at least this
// many times across all transition weights and existing defs.
Machine hoistSharedSubexpressions (const Machine& m,
                                   size_t minNodesToHoist = 3,
                                   size_t minRefsToHoist  = 2);

}  // namespace MachineBoss

#endif  // FELSENSTEIN_INCLUDED
