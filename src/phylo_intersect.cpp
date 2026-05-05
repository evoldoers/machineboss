#include <stdexcept>
#include <cctype>
#include <set>

#include "phylo_intersect.h"
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

}  // anonymous

// ----- The phylogenetic intersection -----

namespace {

Machine buildSubtree (const PhyloTree& tree, size_t v,
                      const Machine& T, const string& timeParam,
                      bool renameTime,
                      ParamAssign* outParams,
                      Machine::SilentCycleStrategy strategy);

Machine branchTransducerForChild (const PhyloTree& tree, size_t parentV, size_t childV,
                                  const Machine& T, const string& timeParam,
                                  bool renameTime,
                                  ParamAssign* outParams,
                                  Machine::SilentCycleStrategy strategy) {
  const PhyloNode& child = tree.nodes[childV];
  Machine Tcopy = T;
  if (renameTime) {
    Tcopy = renameForBranch (T, timeParam, child.name);
    if (outParams && child.hasBranchLength) {
      const string fullName = timeParam + "[" + child.name + "]";
      outParams->defs[fullName] = WeightAlgebra::doubleConstant (child.branchLength);
    }
  }
  Machine sub = buildSubtree (tree, childV, T, timeParam, renameTime, outParams, strategy);
  // pre-compose with the branch above the child
  return Machine::compose (Tcopy, sub, true, true, strategy);
}

Machine buildSubtree (const PhyloTree& tree, size_t v,
                      const Machine& T, const string& timeParam,
                      bool renameTime,
                      ParamAssign* outParams,
                      Machine::SilentCycleStrategy strategy) {
  const PhyloNode& node = tree.nodes[v];
  if (node.children.empty()) {
    // leaf: identity over T's output alphabet
    return Machine::wildEcho (T.outputAlphabet());
  }
  if (node.children.size() != 2)
    throw runtime_error ("phylo intersection: non-binary node \"" + node.name + "\" has " + to_string(node.children.size()) + " children (expected 2)");
  Machine left  = branchTransducerForChild (tree, v, node.children[0], T, timeParam, renameTime, outParams, strategy);
  Machine right = branchTransducerForChild (tree, v, node.children[1], T, timeParam, renameTime, outParams, strategy);
  return Machine::intersect (left, right, strategy);
}

}  // anonymous

Machine phyloIntersect (const Machine& T,
                        const PhyloTree& tree,
                        const string& timeParam,
                        ParamAssign* branchLengthsOut,
                        Machine::SilentCycleStrategy strategy) {
  if (tree.nodes.empty())
    throw runtime_error ("phylo intersection: empty tree");
  if (tree.nodes.size() == 1)
    throw runtime_error ("phylo intersection: tree has only one node (need at least one branch)");

  const bool renameTime = machineHasParam (T, timeParam);
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

  LogThisAt(3,"Phylo intersection: " << tree.nodes.size() << " nodes; "
            << (renameTime ? string("renaming param \"") + timeParam + "\" per branch" : string("no time-param renaming"))
            << endl);

  return buildSubtree (tree, tree.root, T, timeParam, renameTime, branchLengthsOut, strategy);
}

}  // end namespace MachineBoss
