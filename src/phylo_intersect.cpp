#include <stdexcept>
#include <cctype>
#include <set>

#include "phylo_intersect.h"
#include "felsenstein.h"
#include "logger.h"
#include "util.h"

namespace MachineBoss {

// ----- Newick parser -----
// Grammar:
//   tree     := subtree ';'
//   subtree  := '(' subtree (',' subtree)+ ')' name? branch_length?
//             | name? branch_length?
//   name     := unquoted_chars | "'" quoted_chars "'"
//   branch_length := ':' number
//
// Whitespace is skipped between tokens. Square brackets [...] (Newick comments)
// are not supported.

namespace {

class NewickParser {
  const string& s;
  size_t pos = 0;
  PhyloTree tree;

  void skipWhitespace() {
    while (pos < s.size() && isspace((unsigned char) s[pos])) ++pos;
  }

  bool peek (char c) {
    skipWhitespace();
    return pos < s.size() && s[pos] == c;
  }

  bool consume (char c) {
    if (peek(c)) { ++pos; return true; }
    return false;
  }

  void expect (char c) {
    if (!consume(c))
      throw runtime_error (string("Newick parse error: expected '") + c + "' at position " + to_string(pos));
  }

  bool isNameChar (char c) const {
    return c != '(' && c != ')' && c != ',' && c != ':' && c != ';'
        && c != '\'' && c != '[' && c != ']' && !isspace((unsigned char) c);
  }

  string parseName() {
    skipWhitespace();
    string name;
    if (pos < s.size() && s[pos] == '\'') {
      ++pos;
      while (pos < s.size() && s[pos] != '\'') { name.push_back (s[pos]); ++pos; }
      if (pos == s.size()) throw runtime_error ("Newick parse error: unterminated quoted name");
      ++pos;
    } else {
      while (pos < s.size() && isNameChar (s[pos])) { name.push_back (s[pos]); ++pos; }
    }
    return name;
  }

  bool parseBranchLength (double& out) {
    skipWhitespace();
    if (!consume(':')) return false;
    skipWhitespace();
    size_t start = pos;
    if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) ++pos;
    while (pos < s.size() && (isdigit((unsigned char) s[pos]) || s[pos] == '.' || s[pos] == 'e' || s[pos] == 'E' || s[pos] == '+' || s[pos] == '-')) ++pos;
    if (pos == start) throw runtime_error ("Newick parse error: expected branch length number at position " + to_string(start));
    out = stod (s.substr (start, pos - start));
    return true;
  }

  size_t parseSubtree (size_t parent) {
    const size_t self = tree.nodes.size();
    tree.nodes.emplace_back();
    tree.nodes[self].parent = parent;
    skipWhitespace();
    if (peek('(')) {
      expect('(');
      const size_t firstChild = parseSubtree (self);
      tree.nodes[self].children.push_back (firstChild);
      while (peek(',')) {
        expect(',');
        const size_t next = parseSubtree (self);
        tree.nodes[self].children.push_back (next);
      }
      expect(')');
    }
    tree.nodes[self].name = parseName();
    double bl;
    if (parseBranchLength (bl)) {
      tree.nodes[self].hasBranchLength = true;
      tree.nodes[self].branchLength = bl;
    }
    return self;
  }

public:
  NewickParser (const string& s) : s(s) {}
  PhyloTree parse() {
    tree.root = parseSubtree ((size_t) -1);
    skipWhitespace();
    if (pos < s.size() && s[pos] == ';') ++pos;
    return tree;
  }
};

}  // anonymous

PhyloTree PhyloTree::parseNewick (const string& nwk) {
  NewickParser p (nwk);
  return p.parse();
}

// ----- Param renaming -----

namespace {

// Return a copy of m, renamed for use along the branch above a node with the
// given name. timeParam (free) and every key of m.funcs.defs are suffixed with
// "[" + nodeName + "]" so that two copies with different node names do not
// collide on the global names of derived quantities. Other free params (e.g.
// global rate constants) are left alone and remain shared across branches.
Machine renameForBranch (const Machine& m, const string& timeParam, const string& nodeName) {
  Machine out = m;
  const string suffix = "[" + nodeName + "]";

  // Build substitution map: timeParam -> timeParam[v]; each def key -> def_key[v].
  ParamDefs sub;
  sub[timeParam] = WeightAlgebra::param (timeParam + suffix);
  for (const auto& kv: m.funcs.defs)
    sub[kv.first] = WeightAlgebra::param (kv.first + suffix);

  // Substitute in every transition weight.
  for (auto& state: out.state)
    for (auto& t: state.trans)
      t.weight = WeightAlgebra::bind (t.weight, sub);

  // Substitute in every def value, then rename the keys.
  ParamDefs renamedDefs;
  for (const auto& kv: m.funcs.defs)
    renamedDefs[kv.first + suffix] = WeightAlgebra::bind (kv.second, sub);
  out.funcs.defs = renamedDefs;

  // Rename timeParam in constraint vectors. Def keys do not appear in cons
  // (which references only free params), so timeParam is the only rename here.
  auto renameVec = [&](vguard<string>& v) {
    for (auto& s: v) if (s == timeParam) s = timeParam + suffix;
  };
  renameVec (out.cons.prob);
  renameVec (out.cons.rate);
  for (auto& g: out.cons.norm) renameVec (g);

  return out;
}

bool machineHasParam (const Machine& m, const string& name) {
  return m.params().count (name) > 0;
}

// Replace every non-empty input/output symbol with `placeholder` and every
// emit weight with 1. Silent transitions are untouched. The result has the
// same state set and same per-state out-degree shape, but emit transitions
// of T that differed only in their alphabet symbols collapse into a single
// transition once the machine is round-tripped through a TransAccumulator
// (via Machine::compose, which is what phyloIntersect does next).
Machine skeletoniseBranch (const Machine& m, const string& placeholder) {
  Machine out = m;
  const WeightExpr one = WeightAlgebra::doubleConstant (1);
  for (auto& s: out.state)
    for (auto& t: s.trans) {
      if (!t.in.empty())  t.in  = placeholder;
      if (!t.out.empty()) t.out = placeholder;
      if (!t.in.empty() || !t.out.empty()) t.weight = one;
    }
  return out;
}

}  // anonymous

// ----- The phylogenetic intersection -----

namespace {

Machine buildSubtree (const PhyloTree& tree, size_t v,
                      const Machine& T, const string& timeParam,
                      bool renameTime,
                      ParamAssign* outParams,
                      Machine::SilentCycleStrategy strategy,
                      const map<string, vguard<string> >* leafClamps);

Machine branchTransducerForChild (const PhyloTree& tree, size_t parentV, size_t childV,
                                  const Machine& T, const string& timeParam,
                                  bool renameTime,
                                  ParamAssign* outParams,
                                  Machine::SilentCycleStrategy strategy,
                                  const map<string, vguard<string> >* leafClamps) {
  const PhyloNode& child = tree.nodes[childV];
  Machine Tcopy = T;
  if (renameTime) {
    Tcopy = renameForBranch (T, timeParam, child.name);
    if (outParams && child.hasBranchLength) {
      const string fullName = timeParam + "[" + child.name + "]";
      outParams->defs[fullName] = WeightAlgebra::doubleConstant (child.branchLength);
    }
  }
  Machine sub = buildSubtree (tree, childV, T, timeParam, renameTime, outParams, strategy, leafClamps);
  // pre-compose with the branch above the child
  return Machine::compose (Tcopy, sub, true, true, strategy);
}

Machine buildSubtree (const PhyloTree& tree, size_t v,
                      const Machine& T, const string& timeParam,
                      bool renameTime,
                      ParamAssign* outParams,
                      Machine::SilentCycleStrategy strategy,
                      const map<string, vguard<string> >* leafClamps) {
  const PhyloNode& node = tree.nodes[v];
  if (node.children.empty()) {
    // leaf: either an identity over T's output alphabet, or a recognizer
    // of an observed sequence if a clamp was supplied for this leaf.
    if (leafClamps) {
      auto it = leafClamps->find (node.name);
      if (it != leafClamps->end())
        return Machine::recognizer (it->second);
    }
    return Machine::wildEcho (T.outputAlphabet());
  }
  if (node.children.size() == 1) {
    // degree-1 internal node: descend through the single branch
    return branchTransducerForChild (tree, v, node.children[0], T, timeParam, renameTime, outParams, strategy, leafClamps);
  }
  // degree >= 2: fold-left intersect over the children's branch transducers
  Machine acc = branchTransducerForChild (tree, v, node.children[0], T, timeParam, renameTime, outParams, strategy, leafClamps);
  for (size_t i = 1; i < node.children.size(); ++i) {
    Machine sib = branchTransducerForChild (tree, v, node.children[i], T, timeParam, renameTime, outParams, strategy, leafClamps);
    acc = Machine::intersect (acc, sib, strategy);
  }
  return acc;
}

}  // anonymous

Machine phyloIntersect (const Machine& T,
                        const PhyloTree& tree,
                        const string& timeParam,
                        ParamAssign* branchLengthsOut,
                        Machine::SilentCycleStrategy strategy,
                        const map<string, vguard<string> >* leafClamps,
                        bool felsenstein,
                        bool skeleton) {
  if (tree.nodes.empty())
    throw runtime_error ("phylo intersection: empty tree");
  if (tree.nodes.size() == 1)
    throw runtime_error ("phylo intersection: tree has only one node (need at least one branch)");

  // In skeleton mode, replace T's emit alphabet with a single placeholder
  // and clamp leaf observations to the same placeholder (length-only). The
  // recursion below is unchanged. Felsenstein hoisting on a skeleton has
  // nothing to factor (weights are constant 1) so we skip it.
  Machine T_use = T;
  map<string, vguard<string> > skeletonClamps;
  const map<string, vguard<string> >* leafClamps_use = leafClamps;
  if (skeleton) {
    const string placeholder = "*";
    T_use = skeletoniseBranch (T, placeholder);
    if (leafClamps) {
      for (const auto& kv: *leafClamps)
        skeletonClamps[kv.first] = vguard<string> (kv.second.size(), placeholder);
      leafClamps_use = &skeletonClamps;
    }
    felsenstein = false;
  }

  const bool renameTime = machineHasParam (T_use, timeParam);
  if (renameTime) {
    set<string> seen;
    for (size_t i = 0; i < tree.nodes.size(); ++i)
      if (i != tree.root) {
        const string& nm = tree.nodes[i].name;
        if (nm.empty())
          throw runtime_error ("phylo intersection: branch transducer has parameter \"" + timeParam + "\" but a non-root node has no name");
        if (!seen.insert(nm).second)
          throw runtime_error ("phylo intersection: duplicate node name \"" + nm + "\"");
      }
  }

  if (leafClamps_use) {
    set<string> leafNames;
    for (size_t i = 0; i < tree.nodes.size(); ++i)
      if (tree.nodes[i].children.empty())
        leafNames.insert (tree.nodes[i].name);
    for (const auto& kv : *leafClamps_use)
      if (!leafNames.count (kv.first))
        throw runtime_error ("phylo intersection: --phylo-clamp leaf \"" + kv.first + "\" is not a leaf in the tree");
  }

  LogThisAt(3,"Phylo intersection: " << tree.nodes.size() << " nodes; "
            << (renameTime ? string("renaming param \"") + timeParam + "\" per branch" : string("no time-param renaming"))
            << (leafClamps_use ? string("; ") + to_string(leafClamps_use->size()) + " leaves clamped" : string())
            << (skeleton ? string("; skeleton mode") : string())
            << endl);

  Machine m = buildSubtree (tree, tree.root, T_use, timeParam, renameTime, branchLengthsOut, strategy, leafClamps_use);
  if (felsenstein)
    m = hoistSharedSubexpressions (m);
  return m;
}

}  // end namespace MachineBoss
