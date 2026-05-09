#include "felsenstein.h"

#include <iomanip>
#include <map>
#include <sstream>
#include <string>

#include "weight.h"
#include "logger.h"

namespace MachineBoss {

namespace {

using std::map;
using std::string;

// Recursive structural key for a WeightExpr. Identical structures produce
// identical keys regardless of pointer identity. Memoised by ExprPtr.
string structuralKey (WeightExpr w, map<WeightExpr, string>& memo) {
  if (!w) return "0";
  auto it = memo.find (w);
  if (it != memo.end()) return it->second;
  std::ostringstream o;
  switch (w->type) {
    case Null: o << "N"; break;
    case Int:  o << "I" << w->args.intValue; break;
    case Dbl:  o << "D" << std::setprecision(17) << w->args.doubleValue; break;
    case Param: o << "P:" << *w->args.param; break;
    case Mul: case Add: case Sub: case Div: case Pow: {
      char op = (w->type == Mul ? '*' :
                 w->type == Add ? '+' :
                 w->type == Sub ? '-' :
                 w->type == Div ? '/' : '^');
      o << op << '('
        << structuralKey (w->args.binary.l, memo) << ','
        << structuralKey (w->args.binary.r, memo) << ')';
      break;
    }
    case Log: case Exp: {
      char op = (w->type == Log ? 'l' : 'e');
      o << op << '(' << structuralKey (w->args.arg, memo) << ')';
      break;
    }
    default: o << "?";
  }
  memo[w] = o.str();
  return memo[w];
}

// Count node references in a WeightExpr tree. Counted by structural key,
// so identical sub-expressions reachable via different ExprPtrs add up.
void countByKey (WeightExpr w,
                 map<string, size_t>& counts,
                 map<WeightExpr, string>& memo) {
  if (!w) return;
  const string& k = structuralKey (w, memo);
  ++counts[k];
  switch (w->type) {
    case Mul: case Add: case Sub: case Div: case Pow:
      countByKey (w->args.binary.l, counts, memo);
      countByKey (w->args.binary.r, counts, memo);
      break;
    case Log: case Exp:
      countByKey (w->args.arg, counts, memo);
      break;
    default: break;
  }
}

// Tree size (number of internal nodes + leaves).
size_t exprSize (WeightExpr w, map<WeightExpr, size_t>& memo) {
  if (!w) return 0;
  auto it = memo.find (w);
  if (it != memo.end()) return it->second;
  size_t n = 1;
  switch (w->type) {
    case Mul: case Add: case Sub: case Div: case Pow:
      n += exprSize (w->args.binary.l, memo);
      n += exprSize (w->args.binary.r, memo);
      break;
    case Log: case Exp:
      n += exprSize (w->args.arg, memo);
      break;
    default: break;
  }
  memo[w] = n;
  return n;
}

// Rewrite a WeightExpr, replacing shared sub-expressions with Param refs
// to (auto-named) entries added to `defs`. Returns the rewritten expr.
//
// `allowToplevelDefRef`: if true, the top-level call may early-return as
// a Param ref to an existing def whose structure matches `w`. Set false
// when rewriting a def's OWN expression to prevent it from rewriting to
// a self-reference. Recursive (child) calls always use true.
WeightExpr rewrite (WeightExpr w,
                    const map<string, size_t>& counts,
                    ParamDefs& defs,
                    map<string, string>& keyToDef,
                    map<WeightExpr, WeightExpr>& rewriteMemo,
                    map<WeightExpr, string>& keyMemo,
                    map<WeightExpr, size_t>& sizeMemo,
                    size_t minNodes,
                    size_t minRefs,
                    size_t& nextIdx,
                    bool allowToplevelDefRef = true) {
  if (!w) return w;
  // The cache is only consulted (and populated) when allowToplevelDefRef
  // is true, because the same ExprPtr can rewrite differently depending
  // on whether it's the top of a def expression (which must NOT be
  // replaced by a Param ref to itself) versus a sub-expression of a
  // transition weight (which CAN be replaced).
  if (allowToplevelDefRef) {
    auto rit = rewriteMemo.find (w);
    if (rit != rewriteMemo.end()) return rit->second;
  }
  // If this exact expression's structural key has already been hoisted,
  // return the existing def reference (skip this lookup at the top of a
  // def's own rewrite to prevent the def from becoming a self-reference).
  const string& k = structuralKey (w, keyMemo);
  if (allowToplevelDefRef) {
    auto kit = keyToDef.find (k);
    if (kit != keyToDef.end()) {
      WeightExpr p = WeightAlgebra::param (kit->second);
      rewriteMemo[w] = p;
      return p;
    }
  }
  // Recurse on children, building the rewritten tree bottom-up.
  WeightExpr newW = w;
  switch (w->type) {
    case Null: case Int: case Dbl: case Param:
      newW = w;  // leaf
      break;
    case Mul: {
      WeightExpr l = rewrite (w->args.binary.l, counts, defs, keyToDef,
                              rewriteMemo, keyMemo, sizeMemo, minNodes, minRefs, nextIdx);
      WeightExpr r = rewrite (w->args.binary.r, counts, defs, keyToDef,
                              rewriteMemo, keyMemo, sizeMemo, minNodes, minRefs, nextIdx);
      newW = WeightAlgebra::multiply (l, r);
      break;
    }
    case Div: {
      WeightExpr l = rewrite (w->args.binary.l, counts, defs, keyToDef,
                              rewriteMemo, keyMemo, sizeMemo, minNodes, minRefs, nextIdx);
      WeightExpr r = rewrite (w->args.binary.r, counts, defs, keyToDef,
                              rewriteMemo, keyMemo, sizeMemo, minNodes, minRefs, nextIdx);
      newW = WeightAlgebra::divide (l, r);
      break;
    }
    case Add: {
      WeightExpr l = rewrite (w->args.binary.l, counts, defs, keyToDef,
                              rewriteMemo, keyMemo, sizeMemo, minNodes, minRefs, nextIdx);
      WeightExpr r = rewrite (w->args.binary.r, counts, defs, keyToDef,
                              rewriteMemo, keyMemo, sizeMemo, minNodes, minRefs, nextIdx);
      newW = WeightAlgebra::add (l, r);
      break;
    }
    case Sub: {
      WeightExpr l = rewrite (w->args.binary.l, counts, defs, keyToDef,
                              rewriteMemo, keyMemo, sizeMemo, minNodes, minRefs, nextIdx);
      WeightExpr r = rewrite (w->args.binary.r, counts, defs, keyToDef,
                              rewriteMemo, keyMemo, sizeMemo, minNodes, minRefs, nextIdx);
      newW = WeightAlgebra::subtract (l, r);
      break;
    }
    case Pow: {
      WeightExpr l = rewrite (w->args.binary.l, counts, defs, keyToDef,
                              rewriteMemo, keyMemo, sizeMemo, minNodes, minRefs, nextIdx);
      WeightExpr r = rewrite (w->args.binary.r, counts, defs, keyToDef,
                              rewriteMemo, keyMemo, sizeMemo, minNodes, minRefs, nextIdx);
      newW = WeightAlgebra::power (l, r);
      break;
    }
    case Log: {
      WeightExpr a = rewrite (w->args.arg, counts, defs, keyToDef,
                              rewriteMemo, keyMemo, sizeMemo, minNodes, minRefs, nextIdx);
      newW = WeightAlgebra::logOf (a);
      break;
    }
    case Exp: {
      WeightExpr a = rewrite (w->args.arg, counts, defs, keyToDef,
                              rewriteMemo, keyMemo, sizeMemo, minNodes, minRefs, nextIdx);
      newW = WeightAlgebra::expOf (a);
      break;
    }
    default: break;
  }
  // Hoist this sub-expression if it's shared and non-trivial.
  const bool isLeaf = (w->type == Null || w->type == Int ||
                       w->type == Dbl  || w->type == Param);
  if (!isLeaf) {
    auto cit = counts.find (k);
    const size_t refs = (cit == counts.end()) ? 0 : cit->second;
    const size_t sz = exprSize (w, sizeMemo);
    if (refs >= minRefs && sz >= minNodes) {
      std::ostringstream nm;
      nm << "_phy" << nextIdx++;
      const string defName = nm.str();
      defs[defName] = newW;
      keyToDef[k] = defName;
      newW = WeightAlgebra::param (defName);
    }
  }
  if (allowToplevelDefRef)
    rewriteMemo[w] = newW;
  return newW;
}

}  // anon

Machine hoistSharedSubexpressions (const Machine& m,
                                   size_t minNodesToHoist,
                                   size_t minRefsToHoist) {
  // Phase 1: count usage of each unique structural sub-expression across
  // all existing defs and transition weights.
  map<WeightExpr, string> keyMemo;
  map<string, size_t> counts;
  for (const auto& d : m.funcs.defs) countByKey (d.second, counts, keyMemo);
  for (const auto& s : m.state)
    for (const auto& t : s.trans)
      countByKey (t.weight, counts, keyMemo);

  // Phase 2: rewrite, hoisting any structural sub-expression that
  // appears `>= minRefsToHoist` times AND has tree size >= minNodesToHoist.
  Machine out = m;
  ParamDefs newDefs = m.funcs.defs;
  map<string, string> keyToDef;
  // Pre-populate keyToDef with existing defs so re-occurrences of those
  // structures resolve to the existing name (we don't need to invent a
  // new name for `pSame` if there's already a `pSame` def).
  for (const auto& d : m.funcs.defs) {
    string k = structuralKey (d.second, keyMemo);
    keyToDef[k] = d.first;
  }
  map<WeightExpr, WeightExpr> rewriteMemo;
  map<WeightExpr, size_t> sizeMemo;
  size_t nextIdx = 0;

  // Rewrite existing defs (they may share sub-expressions with each other).
  // Pass `allowToplevelDefRef=false` so the top of each def expression doesn't
  // dereference back to itself (the keyToDef pre-pop above mapped each def's
  // structural key to its own name).
  ParamDefs rewrittenDefs;
  for (const auto& d : m.funcs.defs)
    rewrittenDefs[d.first] = rewrite (d.second, counts, newDefs, keyToDef,
                                      rewriteMemo, keyMemo, sizeMemo,
                                      minNodesToHoist, minRefsToHoist, nextIdx,
                                      /*allowToplevelDefRef=*/false);
  // Replace original defs with rewritten ones (keeping any new _phy* defs
  // we added in `newDefs`).
  for (const auto& d : rewrittenDefs) newDefs[d.first] = d.second;

  // Rewrite transition weights.
  for (auto& s : out.state) {
    for (auto& t : s.trans) {
      t.weight = rewrite (t.weight, counts, newDefs, keyToDef,
                          rewriteMemo, keyMemo, sizeMemo,
                          minNodesToHoist, minRefsToHoist, nextIdx);
    }
  }

  out.funcs.defs = newDefs;
  LogThisAt(3,"Hoisted " << nextIdx << " shared sub-expressions into defs"
            << " (existing defs: " << m.funcs.defs.size()
            << ", final defs: " << out.funcs.defs.size() << ")" << std::endl);
  return out;
}

}  // namespace MachineBoss
