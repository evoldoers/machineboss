#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <json.hpp>

#include "rust_codegen.h"
#include "machine.h"
#include "weight.h"
#include "logger.h"

namespace MachineBoss {

using nlohmann::json;

namespace {

// ---------------------------------------------------------------------------
// Parsed pair-token tree.
// Either a leaf (a string, possibly empty for silent) or an array of children.
// We always store the JSON representation so we can also detect "" at any
// internal node (which means the entire subtree is silent).
struct PTok {
  bool isLeaf;             // true => leaf string, false => non-empty array
  std::string leaf;        // when isLeaf
  std::vector<PTok> kids;  // when !isLeaf
};

PTok fromJson (const json& j) {
  PTok t;
  if (j.is_array()) {
    t.isLeaf = false;
    for (const auto& c : j) t.kids.push_back (fromJson (c));
  } else if (j.is_string()) {
    t.isLeaf = true;
    t.leaf = j.get<std::string>();
  } else if (j.is_number()) {
    t.isLeaf = true;
    // Tokens that happen to look like numbers (e.g. "0", "1") get parsed
    // as JSON numbers by --pair-json's recursive splice; treat them as the
    // string form for our purposes.
    t.leaf = j.dump();
  } else {
    throw std::runtime_error ("rust-codegen: unexpected JSON value in pair token: " + j.dump());
  }
  return t;
}

// Parse a token string into a PTok. For pair-token outputs (--pair-json),
// the string is JSON. For L=1 phylo (no intersection performed), output
// tokens are bare symbols like "A" with no JSON encoding — treat those as
// leaf-symbol tokens.
PTok parseTokenJson (const std::string& s) {
  if (s.empty()) {
    PTok t; t.isLeaf = true; t.leaf = ""; return t;
  }
  // Heuristic: only attempt JSON parse if the token looks like a JSON
  // value. Otherwise treat it as a bare symbol string. This keeps L=1
  // phylo machines (single-leaf trees, where no pair-token encoding is
  // performed) working.
  const char c = s[0];
  if (c == '[' || c == '"' || c == '-' || (c >= '0' && c <= '9')) {
    try { return fromJson (json::parse (s)); }
    catch (const json::exception&) { /* fall through to literal */ }
  }
  PTok t; t.isLeaf = true; t.leaf = s; return t;
}

// Merge two PTok shapes: at each position, the deeper structure wins.
// "" at an array position is allowed (means "all leaves under this subtree
// are silent in this token") and is replaced by the array shape from the
// other side.
PTok mergeShape (const PTok& a, const PTok& b) {
  if (!a.isLeaf && !b.isLeaf) {
    if (a.kids.size() != b.kids.size())
      throw std::runtime_error ("rust-codegen: inconsistent pair-token arity across transitions");
    PTok m; m.isLeaf = false;
    for (size_t i = 0; i < a.kids.size(); ++i)
      m.kids.push_back (mergeShape (a.kids[i], b.kids[i]));
    return m;
  }
  if (!a.isLeaf) {
    if (!(b.isLeaf && b.leaf.empty()))
      throw std::runtime_error ("rust-codegen: pair-token shape conflict (array vs non-empty leaf)");
    return a;
  }
  if (!b.isLeaf) {
    if (!(a.isLeaf && a.leaf.empty()))
      throw std::runtime_error ("rust-codegen: pair-token shape conflict (array vs non-empty leaf)");
    return b;
  }
  PTok m; m.isLeaf = true; m.leaf = "";  // shape only — symbol value forgotten
  return m;
}

// Walk a token alongside the canonical template. For each leaf position
// of the template, record the emitted symbol (or "" for ε) into out, in
// left-to-right traversal order.
void decodeAgainst (const PTok& tok, const PTok& tmpl, std::vector<std::string>& out) {
  if (tmpl.isLeaf) {
    if (tok.isLeaf) {
      out.push_back (tok.leaf);
    } else {
      throw std::runtime_error ("rust-codegen: token has array where template expects leaf");
    }
    return;
  }
  // template is an internal node
  if (tok.isLeaf) {
    if (!tok.leaf.empty())
      throw std::runtime_error ("rust-codegen: non-empty leaf where template expects array");
    // All leaves under this subtree are silent.
    std::function<void(const PTok&)> emitSilent = [&](const PTok& t) {
      if (t.isLeaf) out.push_back ("");
      else for (const auto& c : t.kids) emitSilent (c);
    };
    emitSilent (tmpl);
    return;
  }
  if (tok.kids.size() != tmpl.kids.size())
    throw std::runtime_error ("rust-codegen: token arity mismatch with template");
  for (size_t i = 0; i < tok.kids.size(); ++i)
    decodeAgainst (tok.kids[i], tmpl.kids[i], out);
}

size_t countLeaves (const PTok& t) {
  if (t.isLeaf) return 1;
  size_t n = 0;
  for (const auto& c : t.kids) n += countLeaves (c);
  return n;
}

// ---------------------------------------------------------------------------
// WeightExpr -> Rust expression converter. Returns a string of Rust f64 code
// referencing struct fields p.<name> for free parameters and local variables
// def_<name> for ParamFuncs defs (caller is responsible for emitting those
// in topological order).

std::string sanitize (const std::string& s) {
  // Replace anything not [A-Za-z0-9_] with _ to make a valid Rust ident.
  std::string r;
  r.reserve (s.size());
  for (char c : s) {
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')
      r.push_back (c);
    else
      r.push_back ('_');
  }
  if (r.empty() || (r[0] >= '0' && r[0] <= '9')) r = "p_" + r;
  return r;
}

// VM-bytecode emitter. The naive approach of inlining each weight expression
// as a Rust expression — even with CSE — produces hundreds of MB of source
// for a TKF92 quartet (after silent-cycle geomsum reductions). Rustc cannot
// compile that. We instead emit each unique sub-expression as one entry in
// a small, register-machine-style program represented as flat static arrays
// (OPCODES / ARG_A / ARG_B / CONSTS), and the runtime evaluator is a single
// straight-line loop. Compile time is then dominated by reading data tables,
// which scales linearly.
//
// Opcodes:
//   0 = PushConst, arg_a = index into CONSTS (no arg_b)
//   1 = PushParam, arg_a = index into a runtime params array
//   2 = Mul, arg_a/b = node indices
//   3 = Add
//   4 = Sub
//   5 = Div
//   6 = Pow
//   7 = Log, arg_a = node index (no arg_b)
//   8 = Exp, arg_a = node index
//
// CSE is performed across ALL defs and weights globally (one pass), so
// structurally identical sub-expressions share a node.
struct VmInstr { uint8_t opcode; uint32_t a; uint32_t b; };

class VmEmitter {
public:
  std::map<std::string, uint32_t> funcNode;   // def-name -> node index
  std::map<std::string, uint32_t> paramSlot;  // free-param-name -> param-array index
  std::map<WeightExpr, uint32_t> nodeOfExpr;  // pointer-keyed memo
  std::map<std::string, uint32_t> nodeByKey;  // structural-key memo
  std::map<double, uint32_t> nodeByConst;     // constant value -> node
  std::vector<double> consts;
  std::vector<VmInstr> instr;

  uint32_t emitConst (double v) {
    auto it = nodeByConst.find (v);
    if (it != nodeByConst.end()) return it->second;
    uint32_t cidx = (uint32_t) consts.size();
    consts.push_back (v);
    uint32_t nidx = (uint32_t) instr.size();
    instr.push_back ({ 0, cidx, 0 });
    nodeByConst[v] = nidx;
    return nidx;
  }

  uint32_t visit (WeightExpr w) {
    if (!w) return emitConst (1.0);
    auto pit = nodeOfExpr.find (w);
    if (pit != nodeOfExpr.end()) return pit->second;
    uint32_t result = (uint32_t) -1;
    switch (w->type) {
      case Null: result = emitConst (1.0); break;
      case Int:  result = emitConst ((double) w->args.intValue); break;
      case Dbl:  result = emitConst (w->args.doubleValue); break;
      case Param: {
        const std::string& name = *w->args.param;
        auto fit = funcNode.find (name);
        if (fit != funcNode.end()) { result = fit->second; break; }
        auto sit = paramSlot.find (name);
        if (sit != paramSlot.end()) {
          uint32_t nidx = (uint32_t) instr.size();
          instr.push_back ({ 1, sit->second, 0 });
          result = nidx;
          break;
        }
        throw std::runtime_error ("rust-codegen: unbound parameter \"" + name + "\"");
      }
      case Mul: case Add: case Sub: case Div: case Pow: {
        uint32_t l = visit (w->args.binary.l);
        uint32_t r = visit (w->args.binary.r);
        uint8_t op = (w->type == Mul ? 2 :
                      w->type == Add ? 3 :
                      w->type == Sub ? 4 :
                      w->type == Div ? 5 : 6);
        std::ostringstream k;
        k << (int)op << ":" << l << "," << r;
        const std::string key = k.str();
        auto kit = nodeByKey.find (key);
        if (kit != nodeByKey.end()) { result = kit->second; break; }
        uint32_t nidx = (uint32_t) instr.size();
        instr.push_back ({ op, l, r });
        nodeByKey[key] = nidx;
        result = nidx;
        break;
      }
      case Log: case Exp: {
        uint32_t a = visit (w->args.arg);
        uint8_t op = (w->type == Log ? 7 : 8);
        std::ostringstream k;
        k << (int)op << ":" << a;
        const std::string key = k.str();
        auto kit = nodeByKey.find (key);
        if (kit != nodeByKey.end()) { result = kit->second; break; }
        uint32_t nidx = (uint32_t) instr.size();
        instr.push_back ({ op, a, 0 });
        nodeByKey[key] = nidx;
        result = nidx;
        break;
      }
      default:
        throw std::runtime_error ("rust-codegen: unknown WeightExpr type");
    }
    nodeOfExpr[w] = result;
    return result;
  }
};

// Topologically sort def names so that each def comes after all defs it
// references. Free params are not in defs, so they don't constrain order.
std::vector<std::string> topoSortDefs (const ParamDefs& defs) {
  std::map<std::string, std::set<std::string> > depsOf;
  std::set<std::string> defNames;
  for (const auto& d : defs) defNames.insert (d.first);
  std::function<void(WeightExpr, std::set<std::string>&)> collect =
    [&](WeightExpr w, std::set<std::string>& acc) {
      if (!w) return;
      switch (w->type) {
        case Param: {
          const std::string& n = *w->args.param;
          if (defNames.count(n)) acc.insert(n);
          break;
        }
        case Mul: case Div: case Add: case Sub: case Pow:
          collect (w->args.binary.l, acc);
          collect (w->args.binary.r, acc);
          break;
        case Log: case Exp:
          collect (w->args.arg, acc);
          break;
        default: break;
      }
    };
  for (const auto& d : defs) {
    std::set<std::string> deps;
    collect (d.second, deps);
    deps.erase (d.first);
    depsOf[d.first] = deps;
  }
  std::vector<std::string> order;
  std::set<std::string> placed;
  std::function<void(const std::string&)> visit = [&](const std::string& n) {
    if (placed.count(n)) return;
    placed.insert(n);
    for (const auto& d : depsOf[n]) visit (d);
    order.push_back (n);
  };
  for (const auto& n : defNames) visit (n);
  return order;
}

// ---------------------------------------------------------------------------
// Bucket key: (src, dst, deltas, syms-at-emitting-positions).
struct BucketKey {
  StateIndex src, dst;
  std::vector<uint8_t> deltas;   // length L
  std::vector<int> symIdx;       // length L; -1 where delta=0
  bool operator< (const BucketKey& o) const {
    if (src != o.src) return src < o.src;
    if (dst != o.dst) return dst < o.dst;
    if (deltas != o.deltas) return deltas < o.deltas;
    return symIdx < o.symIdx;
  }
};

// ---------------------------------------------------------------------------
// Filesystem helpers.
void mkdirP (const std::string& dir) {
  ::mkdir (dir.c_str(), 0755);  // ignore EEXIST
  std::string sub = dir + "/src";
  ::mkdir (sub.c_str(), 0755);
}

}  // anonymous namespace

void compileRust (const Machine& m, const std::string& outputDir, bool emitViterbi) {
  // -------------------------- Validation & analysis ---------------------------
  if (!m.inputAlphabet().empty())
    throw std::runtime_error ("rust-codegen: machine has non-empty input alphabet; expected a generator");

  // Defensively advance-sort so silent transitions go from earlier to later states.
  Machine sorted = m.advanceSort();
  const auto& states = sorted.state;
  const size_t N = states.size();
  const StateIndex startState = 0;
  const StateIndex endState = (StateIndex)(N - 1);

  // Collect all distinct non-empty output tokens, parse them, build canonical template.
  PTok tmpl; bool haveTmpl = false;
  for (StateIndex s = 0; s < (StateIndex)N; ++s) {
    for (const auto& t : states[s].trans) {
      if (t.out.empty()) continue;
      PTok p = parseTokenJson (t.out);
      if (!haveTmpl) { tmpl = p; haveTmpl = true; }
      else tmpl = mergeShape (tmpl, p);
    }
  }
  if (!haveTmpl)
    throw std::runtime_error ("rust-codegen: machine has no emitting transitions");
  const size_t L = countLeaves (tmpl);
  if (L == 0)
    throw std::runtime_error ("rust-codegen: degenerate template with zero leaves");

  // Collect leaf alphabet (union over all per-leaf-position symbols, excluding "").
  std::set<std::string> alphSet;
  for (StateIndex s = 0; s < (StateIndex)N; ++s) {
    for (const auto& t : states[s].trans) {
      if (t.out.empty()) continue;
      std::vector<std::string> profile;
      decodeAgainst (parseTokenJson (t.out), tmpl, profile);
      for (const auto& sym : profile) if (!sym.empty()) alphSet.insert (sym);
    }
  }
  std::vector<std::string> alph (alphSet.begin(), alphSet.end());
  std::map<std::string,int> symIdx;
  for (size_t i = 0; i < alph.size(); ++i) symIdx[alph[i]] = (int)i;

  // Free parameters: walk only REACHABLE param names. Start from params
  // referenced in transition weights, then transitively descend into the
  // expressions of any referenced def. Any param that isn't reachable from
  // a transition (e.g. `t` referenced only by an unused root-side HKY85 def
  // that was pruned away by composition) is dropped — it shouldn't clutter
  // the generated Params struct.
  std::set<std::string> defNames;
  for (const auto& d : sorted.funcs.defs) defNames.insert (d.first);

  std::set<std::string> reachable;     // all reachable param names (def + free)
  std::function<void(WeightExpr)> collectParams = [&](WeightExpr w) {
    if (!w) return;
    switch (w->type) {
      case Param: reachable.insert (*w->args.param); break;
      case Mul: case Div: case Add: case Sub: case Pow:
        collectParams (w->args.binary.l);
        collectParams (w->args.binary.r);
        break;
      case Log: case Exp: collectParams (w->args.arg); break;
      default: break;
    }
  };
  for (StateIndex s = 0; s < (StateIndex)N; ++s)
    for (const auto& t : states[s].trans)
      collectParams (t.weight);
  // Iterate to fixed point: each newly-reached def name pulls in its
  // expression's parameter references.
  for (size_t prev = 0; prev != reachable.size(); ) {
    prev = reachable.size();
    std::vector<std::string> snap (reachable.begin(), reachable.end());
    for (const auto& nm : snap) {
      auto it = sorted.funcs.defs.find (nm);
      if (it != sorted.funcs.defs.end()) collectParams (it->second);
    }
  }
  std::set<std::string> freeParams;
  for (const auto& p : reachable) if (!defNames.count(p)) freeParams.insert (p);

  // Bucket transitions and accumulate weight expressions per bucket.
  // Also remember each bucket's original output token (all originals in
  // the same bucket share the same encoded out — they have identical
  // deltas+syms — so any one is fine; we use the first seen).
  std::map<BucketKey, std::pair<WeightExpr, std::string>> buckets;
  for (StateIndex s = 0; s < (StateIndex)N; ++s) {
    for (const auto& t : states[s].trans) {
      BucketKey k;
      k.src = s; k.dst = t.dest;
      k.deltas.assign (L, 0);
      k.symIdx.assign (L, -1);
      if (!t.out.empty()) {
        std::vector<std::string> profile;
        decodeAgainst (parseTokenJson (t.out), tmpl, profile);
        for (size_t i = 0; i < L; ++i) {
          if (!profile[i].empty()) {
            k.deltas[i] = 1;
            k.symIdx[i] = symIdx[profile[i]];
          }
        }
      }
      auto it = buckets.find (k);
      if (it == buckets.end()) buckets[k] = std::make_pair (t.weight, t.out);
      else it->second.first = WeightAlgebra::add (it->second.first, t.weight);
    }
  }

  // Split buckets into silent (all deltas == 0) and emitting.
  struct Entry {
    StateIndex src, dst;
    std::vector<uint8_t> deltas;
    std::vector<int> symIdx;
    size_t weightIndex;
    std::string outToken;  // raw encoded pair-token from the original transition
    WeightExpr summedWeight;  // for machine.json emission
  };
  std::vector<Entry> silent, emitting;
  std::vector<WeightExpr> weightsInOrder;
  for (auto& kv : buckets) {
    Entry e;
    e.src = kv.first.src; e.dst = kv.first.dst;
    e.deltas = kv.first.deltas; e.symIdx = kv.first.symIdx;
    e.weightIndex = weightsInOrder.size();
    e.outToken = kv.second.second;
    e.summedWeight = kv.second.first;
    weightsInOrder.push_back (kv.second.first);
    bool isSilent = true;
    for (uint8_t d : e.deltas) if (d) { isSilent = false; break; }
    if (isSilent) silent.push_back (e); else emitting.push_back (e);
  }
  // Topo order on silent: sort by dst (machine is advance-sorted, so dst-order
  // is a valid topological order for processing within a cell).
  std::sort (silent.begin(), silent.end(), [](const Entry& a, const Entry& b) {
    return a.dst < b.dst;
  });

  // ------------------------------- Emit code -------------------------------
  mkdirP (outputDir);

  // Cargo.toml
  {
    std::ofstream f (outputDir + "/Cargo.toml");
    f << "[package]\n"
      << "name = \"phylo_dp\"\n"
      << "version = \"0.1.0\"\n"
      << "edition = \"2021\"\n\n"
      << "[lib]\n"
      << "name = \"phylo_dp\"\n"
      << "path = \"src/lib.rs\"\n\n"
      << "[profile.release]\n"
      << "opt-level = 3\n"
      << "lto = true\n";
  }

  // src/lib.rs
  std::ofstream f (outputDir + "/src/lib.rs");
  f << std::setprecision(17);
  f << "// Auto-generated by Machine Boss --codegen-rust. Do not edit.\n";
  f << "// Multidimensional Forward (and Viterbi) DP for a phylogenetic\n";
  f << "// composition of a branch transducer on a fixed tree topology.\n\n";
  // non_snake_case: parameter field names follow the original (e.g.
  //   delRate, t_A_, pi_A) — they're not freely renameable.
  // clippy::needless_range_loop: NUM_LEAVES-bounded for-loops match the
  //   index variable to multiple parallel arrays; a range loop is the
  //   most readable form.
  f << "#![allow(non_snake_case, clippy::needless_range_loop)]\n\n";

  f << "pub const NUM_LEAVES: usize = " << L << ";\n";
  f << "pub const NUM_STATES: usize = " << N << ";\n";
  f << "pub const ALPHABET_SIZE: usize = " << alph.size() << ";\n";
  f << "pub const START_STATE: u32 = " << startState << ";\n";
  f << "pub const END_STATE: u32 = " << endState << ";\n\n";

  // Alphabet as a const array of &'static str.
  f << "pub const ALPHABET: [&str; " << alph.size() << "] = [";
  for (size_t i = 0; i < alph.size(); ++i) {
    if (i) f << ", ";
    f << "\"" << json(alph[i]).dump().substr(1, alph[i].size() + 2 - 2) << "\"";  // re-quote properly
  }
  f << "];\n\n";

  // Params struct: one f64 field per free parameter.
  std::map<std::string,std::string> paramRustName;
  f << "#[derive(Debug, Clone)]\n";
  f << "pub struct Params {\n";
  for (const auto& p : freeParams) {
    std::string r = sanitize (p);
    paramRustName[p] = r;
    f << "    pub " << r << ": f64,\n";
  }
  f << "}\n\n";

  // Default impl with zeros (for tests/quick start).
  f << "impl Default for Params {\n";
  f << "    fn default() -> Self {\n";
  f << "        Params {\n";
  for (const auto& p : freeParams) {
    f << "            " << paramRustName[p] << ": 0.0,\n";
  }
  f << "        }\n    }\n}\n\n";

  // compute_log_weights: emit a VM-bytecode program of unique sub-expressions
  // (one node per unique value, globally CSE'd), then a runtime evaluator.
  // This decouples the generated source size from model complexity — large
  // models (e.g. TKF92 quartet) end up with O(unique-nodes) data tables
  // rather than O(weight-expression-bytes) source code. Rustc compiles
  // huge static arrays linearly, so this scales.
  {
    VmEmitter vm;
    auto topoOrder = topoSortDefs (sorted.funcs.defs);
    // Drop unreachable defs from the topological order: their nodes would
    // never be referenced by a transition's weight, but visiting them
    // unconditionally would still emit instructions (and bloat the VM
    // tables). Keep only defs that appear in the reachable set.
    {
      std::vector<std::string> filtered;
      for (const auto& n : topoOrder)
        if (reachable.count (n)) filtered.push_back (n);
      topoOrder.swap (filtered);
    }

    // Free params get sequential slots into a runtime params array.
    std::vector<std::string> paramSlotOrder;
    for (const auto& p : freeParams) {
      vm.paramSlot[p] = (uint32_t) paramSlotOrder.size();
      paramSlotOrder.push_back (p);
    }

    // Visit defs in topological order, populating funcNode so subsequent
    // references to the def name resolve to the def's already-built node.
    for (const auto& n : topoOrder) {
      uint32_t nidx = vm.visit (sorted.funcs.defs.at(n));
      vm.funcNode[n] = nidx;
    }

    // Visit each weight expression; collect the resulting node index per weight.
    const size_t W = weightsInOrder.size();
    std::vector<uint32_t> weightNode (W);
    for (size_t i = 0; i < W; ++i)
      weightNode[i] = vm.visit (weightsInOrder[i]);

    // Emit data tables.
    f << "// VM tables — see compute_log_weights for the evaluator.\n";
    f << "const VM_NUM_NODES: usize = " << vm.instr.size() << ";\n";
    f << "const VM_OPCODES: [u8; VM_NUM_NODES] = [";
    for (size_t i = 0; i < vm.instr.size(); ++i) {
      if (i % 32 == 0) f << "\n    ";
      f << (int) vm.instr[i].opcode << ",";
    }
    f << "\n];\n\n";

    f << "const VM_ARG_A: [u32; VM_NUM_NODES] = [";
    for (size_t i = 0; i < vm.instr.size(); ++i) {
      if (i % 16 == 0) f << "\n    ";
      f << vm.instr[i].a << ",";
    }
    f << "\n];\n\n";

    f << "const VM_ARG_B: [u32; VM_NUM_NODES] = [";
    for (size_t i = 0; i < vm.instr.size(); ++i) {
      if (i % 16 == 0) f << "\n    ";
      f << vm.instr[i].b << ",";
    }
    f << "\n];\n\n";

    f << "const VM_CONSTS: [f64; " << vm.consts.size() << "] = [";
    for (size_t i = 0; i < vm.consts.size(); ++i) {
      if (i % 4 == 0) f << "\n    ";
      f << std::setprecision(17) << vm.consts[i] << "_f64,";
    }
    f << "\n];\n\n";

    // For each weight, the node whose value (after .ln()) becomes the log-weight.
    f << "const WEIGHT_NODE: [u32; " << W << "] = [";
    for (size_t i = 0; i < W; ++i) {
      if (i % 16 == 0) f << "\n    ";
      f << weightNode[i] << ",";
    }
    f << "\n];\n\n";

    // The evaluator. Pre-flatten Params into a stack array so VM_OP=1
    // reads can be a single array index. Then walk nodes in topological
    // order (which is just sequential because nodes were emitted in
    // post-order), filling vals[i] from vals[VM_ARG_A[i]] and vals[VM_ARG_B[i]].
    f << "/// Compute the log-weight of every transition bucket from the\n";
    f << "/// supplied parameters. Callers making many `forward` / `viterbi`\n";
    f << "/// calls with the same `Params` (e.g. benchmark sweeps) can call\n";
    f << "/// this once and reuse the result via `forward_with_log_weights` /\n";
    f << "/// `viterbi_with_log_weights`.\n";
    f << "pub fn precompute_log_weights(p: &Params) -> Vec<f64> {\n";
    f << "    let pvals: [f64; " << paramSlotOrder.size() << "] = [\n";
    for (const auto& pn : paramSlotOrder)
      f << "        p." << paramRustName[pn] << ",\n";
    f << "    ];\n";
    f << "    let mut vals = vec![0.0_f64; VM_NUM_NODES];\n";
    f << "    for i in 0..VM_NUM_NODES {\n";
    f << "        let a = unsafe { *VM_ARG_A.get_unchecked(i) } as usize;\n";
    f << "        let b = unsafe { *VM_ARG_B.get_unchecked(i) } as usize;\n";
    f << "        vals[i] = unsafe { match *VM_OPCODES.get_unchecked(i) {\n";
    f << "            0 => *VM_CONSTS.get_unchecked(a),\n";
    f << "            1 => *pvals.get_unchecked(a),\n";
    f << "            2 => vals.get_unchecked(a) * vals.get_unchecked(b),\n";
    f << "            3 => vals.get_unchecked(a) + vals.get_unchecked(b),\n";
    f << "            4 => vals.get_unchecked(a) - vals.get_unchecked(b),\n";
    f << "            5 => vals.get_unchecked(a) / vals.get_unchecked(b),\n";
    f << "            6 => vals.get_unchecked(a).powf(*vals.get_unchecked(b)),\n";
    f << "            7 => vals.get_unchecked(a).ln(),\n";
    f << "            8 => vals.get_unchecked(a).exp(),\n";
    f << "            _ => core::hint::unreachable_unchecked(),\n";
    f << "        }};\n";
    f << "    }\n";
    f << "    let mut out = vec![0.0_f64; " << W << "];\n";
    f << "    for i in 0.." << W << " {\n";
    f << "        out[i] = unsafe { vals.get_unchecked(*WEIGHT_NODE.get_unchecked(i) as usize) }.ln();\n";
    f << "    }\n";
    f << "    out\n";
    f << "}\n\n";
  }

  // ---- Bucket-index assignment ----
  //
  // Every bucket (silent or emitting) gets a unique global index in the
  // expected_counts vector emitted by forward_backward_counts:
  //   silent[i]               ->  i
  //   emitting (shard d, j)   ->  silent.size() + EMIT_SHARD_BUCKET_OFFSET[d] + j
  // Constants below let runtime code translate (shard, idx) -> global bucket.
  f << "pub const NUM_SILENT: usize = " << silent.size() << ";\n";
  f << "pub const NUM_EMIT: usize = " << emitting.size() << ";\n";
  f << "pub const NUM_BUCKETS: usize = " << (silent.size() + emitting.size()) << ";\n\n";

  // Emit constant tables for silent and emitting transitions.
  // Layout:
  //   SILENT_TRANSITIONS:     [(src, dst, weight_idx)] sorted by dst ASCENDING
  //                           (forward silent closure: process in topo dst-order
  //                           so each dst's inbounds are visited before reads).
  //   SILENT_TRANSITIONS_BWD: [(src, dst, weight_idx)] sorted by src DESCENDING
  //                           (backward silent closure: process in reverse src-
  //                           order so each src's outbounds are finalised first).
  f << "const SILENT_TRANSITIONS: &[(u32, u32, u32)] = &[\n";
  for (const auto& e : silent)
    f << "    (" << e.src << ", " << e.dst << ", " << e.weightIndex << "),\n";
  f << "];\n\n";

  std::vector<Entry> silentBwd = silent;
  std::sort (silentBwd.begin(), silentBwd.end(), [](const Entry& a, const Entry& b) {
    return a.src > b.src;  // descending
  });
  f << "const SILENT_TRANSITIONS_BWD: &[(u32, u32, u32)] = &[\n";
  for (const auto& e : silentBwd)
    f << "    (" << e.src << ", " << e.dst << ", " << e.weightIndex << "),\n";
  f << "];\n\n";

  // Shard emitting transitions by their delta vector. There are at most
  // 2^L distinct delta vectors; many are unused. The DP inner loop checks
  // delta-vector feasibility ONCE per shard, hoisting it out of the
  // per-transition loop.
  std::map<std::vector<uint8_t>, std::vector<size_t>> emitByDelta;
  for (size_t i = 0; i < emitting.size(); ++i)
    emitByDelta[emitting[i].deltas].push_back (i);

  std::vector<std::vector<uint8_t>> deltaVecs;
  deltaVecs.reserve (emitByDelta.size());
  for (const auto& kv : emitByDelta) deltaVecs.push_back (kv.first);

  f << "pub const NUM_DELTA: usize = " << deltaVecs.size() << ";\n";
  f << "pub const DELTA_VEC: [[u8; " << L << "]; NUM_DELTA] = [";
  for (size_t d = 0; d < deltaVecs.size(); ++d) {
    f << (d ? ", " : "\n    ") << "[";
    for (size_t k = 0; k < L; ++k) f << (k ? ", " : "") << (int)deltaVecs[d][k];
    f << "]";
  }
  f << "\n];\n\n";

  // Per-shard transition table: (src, dst, syms[L], weight_idx). Symbols
  // are emitted as i32 with -1 in non-emitting positions; the inner loop
  // only inspects positions where the delta vector says to.
  for (size_t d = 0; d < deltaVecs.size(); ++d) {
    const auto& bucket = emitByDelta.at (deltaVecs[d]);
    f << "const EMIT_SHARD_" << d << ": &[(u32, u32, [i32; " << L << "], u32)] = &[\n";
    for (size_t idx : bucket) {
      const Entry& e = emitting[idx];
      f << "    (" << e.src << ", " << e.dst << ", [";
      for (size_t k = 0; k < L; ++k) f << (k ? ", " : "") << e.symIdx[k];
      f << "], " << e.weightIndex << "),\n";
    }
    f << "];\n";
  }
  f << "const EMIT_SHARDS: [&[(u32, u32, [i32; " << L << "], u32)]; NUM_DELTA] = [";
  for (size_t d = 0; d < deltaVecs.size(); ++d)
    f << (d ? ", " : "\n    ") << "EMIT_SHARD_" << d;
  f << "\n];\n\n";

  // Cumulative offsets for global bucket indexing of emitting transitions.
  // EMIT_SHARD_BUCKET_OFFSET[d] = number of emitting transitions in shards 0..d-1
  // (relative to NUM_SILENT — add NUM_SILENT for the global bucket index).
  // Length NUM_DELTA + 1 so EMIT_SHARD_BUCKET_OFFSET[NUM_DELTA] == NUM_EMIT.
  f << "const EMIT_SHARD_BUCKET_OFFSET: [usize; NUM_DELTA + 1] = [";
  size_t accum = 0;
  for (size_t d = 0; d < deltaVecs.size(); ++d) {
    f << (d ? ", " : "\n    ") << accum;
    accum += emitByDelta.at (deltaVecs[d]).size();
  }
  f << ", " << accum << "\n];\n\n";

  // ----- DP runtime helpers -----
  f << R"RUST(
// log_sum_exp(a, b) = log(exp(a) + exp(b)). When the gap between hi and lo
// exceeds 36 nats, the contribution log1p(exp(d)) ≈ exp(d) ≈ 2.3e-16 is
// smaller than the ULP of any cell value `hi` realistically encountered
// in this DP (log-probabilities are deeply negative — typically below -5,
// where ulp(hi) ≥ 2^-49 ≈ 1.78e-15 > exp(-36)), so `hi + log1p(exp(d))`
// rounds back to `hi`. We skip the two transcendental calls in that case.
//
// (Note: this is NOT a universal "below f64 epsilon" claim — for very
// small |hi|, e.g. hi = 0, the contribution exp(-36) IS representable and
// `0 + exp(-36) != 0`. But the DP only reaches that regime in pathological
// cases where the entire likelihood is ~1, which doesn't occur for the
// phylo-composed generators this codegen targets.)
const LSE_CUTOFF: f64 = -36.0;

#[inline(always)]
fn lse(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY { return b; }
    if b == f64::NEG_INFINITY { return a; }
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    let d = lo - hi;
    if d <= LSE_CUTOFF { return hi; }
    hi + d.exp().ln_1p()
}

#[inline(always)]
fn max2(a: f64, b: f64) -> f64 {
    if a >= b { a } else { b }
}

/// A populated Forward or Backward DP matrix.
///
/// `data[state * total + cell]` is the log-probability of being in `state`
/// at multi-cell `idx` (where `cell = sum_k idx[k] * strides[k]`).
/// `log_likelihood` is f[end_state, full] for a Forward matrix or
/// b[start_state, 0] for a Backward matrix; the two should agree to
/// floating-point noise on the same input.
pub struct DPMatrix {
    pub data: Vec<f64>,
    pub strides: [usize; NUM_LEAVES],
    pub lens: [usize; NUM_LEAVES],
    pub total: usize,
    pub log_likelihood: f64,
}

impl DPMatrix {
    /// Linear cell offset for a multi-index. `idx[k]` runs 0..=lens[k].
    #[inline(always)]
    pub fn cell(&self, idx: [usize; NUM_LEAVES]) -> usize {
        let mut c = 0usize;
        for k in 0..NUM_LEAVES { c += idx[k] * self.strides[k]; }
        c
    }
    /// Log-prob of being in `state` at multi-index `idx`.
    #[inline(always)]
    pub fn at(&self, state: u32, idx: [usize; NUM_LEAVES]) -> f64 {
        self.data[(state as usize) * self.total + self.cell(idx)]
    }
}

/// Result of forward_backward_counts: total log-likelihood, and the
/// expected number of times each transition bucket fires (one entry per
/// global bucket index in [0..NUM_BUCKETS): silents first, then emitting
/// in shard order).
pub struct FBResult {
    pub log_likelihood: f64,
    pub expected_counts: Vec<f64>,
}

#[inline]
fn make_strides(leaves: [&[u32]; NUM_LEAVES]) -> ([usize; NUM_LEAVES], [usize; NUM_LEAVES], usize) {
    let mut lens: [usize; NUM_LEAVES] = [0; NUM_LEAVES];
    for k in 0..NUM_LEAVES { lens[k] = leaves[k].len(); }
    let mut total: usize = 1;
    for k in 0..NUM_LEAVES { total *= lens[k] + 1; }
    let mut strides: [usize; NUM_LEAVES] = [1; NUM_LEAVES];
    if NUM_LEAVES > 0 {
        for k in (0..NUM_LEAVES-1).rev() { strides[k] = strides[k+1] * (lens[k+1] + 1); }
    }
    (strides, lens, total)
}

)RUST";

  // ----- Forward fill body (shared by forward_matrix_with_log_weights
  // and viterbi_with_log_weights, parameterised over the reduce op).
  // Iterates cells in lex order; for each cell processes emitting
  // transitions (sharded by delta vector) then silent transitions in
  // topological dst-state order.
  auto emitForwardFillBody = [&](std::ofstream& f, const char* reduce) {
    f << "    let (strides, lens, total) = make_strides(leaves);\n";
    f << "    let mut g = vec![f64::NEG_INFINITY; NUM_STATES * total];\n";
    f << "    g[(START_STATE as usize) * total + 0] = 0.0;\n";
    f << "    // Silent closure at cell 0.\n";
    f << "    for &(src, dst, widx) in SILENT_TRANSITIONS.iter() {\n";
    f << "        unsafe {\n";
    f << "            let sv = *g.get_unchecked((src as usize) * total + 0);\n";
    f << "            if sv != f64::NEG_INFINITY {\n";
    f << "                let dst_off = (dst as usize) * total + 0;\n";
    f << "                let dv = *g.get_unchecked(dst_off);\n";
    f << "                *g.get_unchecked_mut(dst_off) = " << reduce << "(dv, sv + *lw.get_unchecked(widx as usize));\n";
    f << "            }\n";
    f << "        }\n";
    f << "    }\n";
    f << "    let mut idx: [usize; NUM_LEAVES] = [0; NUM_LEAVES];\n";
    f << "    loop {\n";
    f << "        let mut advanced = false;\n";
    f << "        let mut k = NUM_LEAVES;\n";
    f << "        while k > 0 {\n";
    f << "            k -= 1;\n";
    f << "            if idx[k] < lens[k] { idx[k] += 1; for j in (k+1)..NUM_LEAVES { idx[j] = 0; } advanced = true; break; }\n";
    f << "        }\n";
    f << "        if !advanced { break; }\n";
    f << "        let cell: usize = ";
    for (size_t i = 0; i < L; ++i) f << (i ? " + " : "") << "idx[" << i << "] * strides[" << i << "]";
    f << ";\n";
    f << "        // Emitting transitions into this cell, sharded by delta vector.\n";
    f << "        for d_idx in 0..NUM_DELTA {\n";
    f << "            let dvec = unsafe { *DELTA_VEC.get_unchecked(d_idx) };\n";
    f << "            let mut prev = cell;\n";
    f << "            let mut feasible = true;\n";
    f << "            for k in 0..NUM_LEAVES {\n";
    f << "                if dvec[k] == 1 {\n";
    f << "                    if idx[k] == 0 { feasible = false; break; }\n";
    f << "                    prev -= strides[k];\n";
    f << "                }\n";
    f << "            }\n";
    f << "            if !feasible { continue; }\n";
    f << "            let observed: [i32; NUM_LEAVES] = unsafe {[\n";
    for (size_t i = 0; i < L; ++i)
      f << "                if dvec[" << i << "] == 1 { *leaves[" << i << "].get_unchecked(idx[" << i << "] - 1) as i32 } else { -1 },\n";
    f << "            ]};\n";
    f << "            for &(src, dst, syms, widx) in unsafe { EMIT_SHARDS.get_unchecked(d_idx) }.iter() {\n";
    f << "                let mut sym_ok = true;\n";
    f << "                for k in 0..NUM_LEAVES {\n";
    f << "                    if dvec[k] == 1 && observed[k] != syms[k] { sym_ok = false; break; }\n";
    f << "                }\n";
    f << "                if sym_ok {\n";
    f << "                    unsafe {\n";
    f << "                        let sv = *g.get_unchecked((src as usize) * total + prev);\n";
    f << "                        if sv != f64::NEG_INFINITY {\n";
    f << "                            let dst_off = (dst as usize) * total + cell;\n";
    f << "                            let dv = *g.get_unchecked(dst_off);\n";
    f << "                            *g.get_unchecked_mut(dst_off) = " << reduce << "(dv, sv + *lw.get_unchecked(widx as usize));\n";
    f << "                        }\n";
    f << "                    }\n";
    f << "                }\n";
    f << "            }\n";
    f << "        }\n";
    f << "        for &(src, dst, widx) in SILENT_TRANSITIONS.iter() {\n";
    f << "            unsafe {\n";
    f << "                let sv = *g.get_unchecked((src as usize) * total + cell);\n";
    f << "                if sv != f64::NEG_INFINITY {\n";
    f << "                    let dst_off = (dst as usize) * total + cell;\n";
    f << "                    let dv = *g.get_unchecked(dst_off);\n";
    f << "                    *g.get_unchecked_mut(dst_off) = " << reduce << "(dv, sv + *lw.get_unchecked(widx as usize));\n";
    f << "                }\n";
    f << "            }\n";
    f << "        }\n";
    f << "    }\n";
  };

  // ----- Backward fill body. Iterates cells in REVERSE lex order; for
  // each cell processes silent transitions in REVERSE src-state order
  // (so b[dst] at this cell is finalised before b[src] reads it), then
  // pulls in contributions from emitting transitions out to LATER cells.
  auto emitBackwardFillBody = [&](std::ofstream& f) {
    f << "    let (strides, lens, total) = make_strides(leaves);\n";
    f << "    let mut g = vec![f64::NEG_INFINITY; NUM_STATES * total];\n";
    f << "    let final_cell: usize = total - 1;\n";
    f << "    g[(END_STATE as usize) * total + final_cell] = 0.0;\n";
    f << "    // Silent closure at the final cell, in reverse src-order.\n";
    f << "    for &(src, dst, widx) in SILENT_TRANSITIONS_BWD.iter() {\n";
    f << "        unsafe {\n";
    f << "            let dv = *g.get_unchecked((dst as usize) * total + final_cell);\n";
    f << "            if dv != f64::NEG_INFINITY {\n";
    f << "                let src_off = (src as usize) * total + final_cell;\n";
    f << "                let sv = *g.get_unchecked(src_off);\n";
    f << "                *g.get_unchecked_mut(src_off) = lse(sv, dv + *lw.get_unchecked(widx as usize));\n";
    f << "            }\n";
    f << "        }\n";
    f << "    }\n";
    // Iterate cells in reverse lex order (start at lens, descend).
    f << "    let mut idx: [usize; NUM_LEAVES] = lens;\n";
    f << "    loop {\n";
    f << "        // descend idx (returns false if we just processed origin)\n";
    f << "        let mut decremented = false;\n";
    f << "        let mut k = NUM_LEAVES;\n";
    f << "        while k > 0 {\n";
    f << "            k -= 1;\n";
    f << "            if idx[k] > 0 { idx[k] -= 1; for j in (k+1)..NUM_LEAVES { idx[j] = lens[j]; } decremented = true; break; }\n";
    f << "        }\n";
    f << "        if !decremented { break; }\n";
    f << "        let cell: usize = ";
    for (size_t i = 0; i < L; ++i) f << (i ? " + " : "") << "idx[" << i << "] * strides[" << i << "]";
    f << ";\n";
    f << "        // Emitting transitions out of this cell to later cell idx+delta.\n";
    f << "        for d_idx in 0..NUM_DELTA {\n";
    f << "            let dvec = unsafe { *DELTA_VEC.get_unchecked(d_idx) };\n";
    f << "            let mut next = cell;\n";
    f << "            let mut feasible = true;\n";
    f << "            for k in 0..NUM_LEAVES {\n";
    f << "                if dvec[k] == 1 {\n";
    f << "                    if idx[k] == lens[k] { feasible = false; break; }\n";
    f << "                    next += strides[k];\n";
    f << "                }\n";
    f << "            }\n";
    f << "            if !feasible { continue; }\n";
    f << "            // For backward, the consumed observation is at position idx[k] (0-based) — the next char.\n";
    f << "            let observed: [i32; NUM_LEAVES] = unsafe {[\n";
    for (size_t i = 0; i < L; ++i)
      f << "                if dvec[" << i << "] == 1 { *leaves[" << i << "].get_unchecked(idx[" << i << "]) as i32 } else { -1 },\n";
    f << "            ]};\n";
    f << "            for &(src, dst, syms, widx) in unsafe { EMIT_SHARDS.get_unchecked(d_idx) }.iter() {\n";
    f << "                let mut sym_ok = true;\n";
    f << "                for k in 0..NUM_LEAVES {\n";
    f << "                    if dvec[k] == 1 && observed[k] != syms[k] { sym_ok = false; break; }\n";
    f << "                }\n";
    f << "                if sym_ok {\n";
    f << "                    unsafe {\n";
    f << "                        let dv = *g.get_unchecked((dst as usize) * total + next);\n";
    f << "                        if dv != f64::NEG_INFINITY {\n";
    f << "                            let src_off = (src as usize) * total + cell;\n";
    f << "                            let sv = *g.get_unchecked(src_off);\n";
    f << "                            *g.get_unchecked_mut(src_off) = lse(sv, dv + *lw.get_unchecked(widx as usize));\n";
    f << "                        }\n";
    f << "                    }\n";
    f << "                }\n";
    f << "            }\n";
    f << "        }\n";
    f << "        // Silent closure at this cell, in reverse src-order.\n";
    f << "        for &(src, dst, widx) in SILENT_TRANSITIONS_BWD.iter() {\n";
    f << "            unsafe {\n";
    f << "                let dv = *g.get_unchecked((dst as usize) * total + cell);\n";
    f << "                if dv != f64::NEG_INFINITY {\n";
    f << "                    let src_off = (src as usize) * total + cell;\n";
    f << "                    let sv = *g.get_unchecked(src_off);\n";
    f << "                    *g.get_unchecked_mut(src_off) = lse(sv, dv + *lw.get_unchecked(widx as usize));\n";
    f << "                }\n";
    f << "            }\n";
    f << "        }\n";
    f << "    }\n";
  };

  // ----- Forward matrix + scalar wrappers -----
  f << "/// Forward DP matrix with precomputed log-weights.\n";
  f << "pub fn forward_matrix_with_log_weights(lw: &[f64], leaves: [&[u32]; NUM_LEAVES]) -> DPMatrix {\n";
  emitForwardFillBody (f, "lse");
  f << "    let log_likelihood = g[(END_STATE as usize) * total + (total - 1)];\n";
  f << "    DPMatrix { data: g, strides, lens, total, log_likelihood }\n";
  f << "}\n\n";

  f << "/// Forward DP matrix.\n";
  f << "#[inline]\n";
  f << "pub fn forward_matrix(p: &Params, leaves: [&[u32]; NUM_LEAVES]) -> DPMatrix {\n";
  f << "    forward_matrix_with_log_weights(&precompute_log_weights(p), leaves)\n";
  f << "}\n\n";

  f << "/// Forward log-likelihood with precomputed log-weights.\n";
  f << "#[inline]\n";
  f << "pub fn forward_with_log_weights(lw: &[f64], leaves: [&[u32]; NUM_LEAVES]) -> f64 {\n";
  f << "    forward_matrix_with_log_weights(lw, leaves).log_likelihood\n";
  f << "}\n\n";

  f << "/// Forward log-likelihood (computes log-weights internally).\n";
  f << "#[inline]\n";
  f << "pub fn forward(p: &Params, leaves: [&[u32]; NUM_LEAVES]) -> f64 {\n";
  f << "    forward_with_log_weights(&precompute_log_weights(p), leaves)\n";
  f << "}\n\n";

  // ----- Backward matrix + scalar wrappers -----
  f << "/// Backward DP matrix with precomputed log-weights.\n";
  f << "pub fn backward_matrix_with_log_weights(lw: &[f64], leaves: [&[u32]; NUM_LEAVES]) -> DPMatrix {\n";
  emitBackwardFillBody (f);
  f << "    let log_likelihood = g[(START_STATE as usize) * total + 0];\n";
  f << "    DPMatrix { data: g, strides, lens, total, log_likelihood }\n";
  f << "}\n\n";

  f << "/// Backward DP matrix.\n";
  f << "#[inline]\n";
  f << "pub fn backward_matrix(p: &Params, leaves: [&[u32]; NUM_LEAVES]) -> DPMatrix {\n";
  f << "    backward_matrix_with_log_weights(&precompute_log_weights(p), leaves)\n";
  f << "}\n\n";

  f << "/// Backward log-likelihood with precomputed log-weights.\n";
  f << "/// Equals `forward_with_log_weights(lw, leaves)` to floating-point noise.\n";
  f << "#[inline]\n";
  f << "pub fn backward_with_log_weights(lw: &[f64], leaves: [&[u32]; NUM_LEAVES]) -> f64 {\n";
  f << "    backward_matrix_with_log_weights(lw, leaves).log_likelihood\n";
  f << "}\n\n";

  f << "/// Backward log-likelihood (computes log-weights internally).\n";
  f << "#[inline]\n";
  f << "pub fn backward(p: &Params, leaves: [&[u32]; NUM_LEAVES]) -> f64 {\n";
  f << "    backward_with_log_weights(&precompute_log_weights(p), leaves)\n";
  f << "}\n\n";

  // ----- Viterbi (uses forward fill body with max2 reducer) -----
  if (emitViterbi) {
    f << "/// Viterbi log-likelihood with precomputed log-weights.\n";
    f << "pub fn viterbi_with_log_weights(lw: &[f64], leaves: [&[u32]; NUM_LEAVES]) -> f64 {\n";
    emitForwardFillBody (f, "max2");
    f << "    g[(END_STATE as usize) * total + (total - 1)]\n";
    f << "}\n\n";

    f << "/// Viterbi log-likelihood (computes log-weights internally).\n";
    f << "#[inline]\n";
    f << "pub fn viterbi(p: &Params, leaves: [&[u32]; NUM_LEAVES]) -> f64 {\n";
    f << "    viterbi_with_log_weights(&precompute_log_weights(p), leaves)\n";
    f << "}\n\n";
  }

  // ----- Posterior helpers -----
  f << R"RUST(
/// Log-posterior of being in `state` at multi-index `idx`, given Forward
/// matrix `f` and Backward matrix `b` (both run on the same params and
/// leaves). Uses `f.log_likelihood` as the partition function.
#[inline]
pub fn state_log_posterior(
    f: &DPMatrix, b: &DPMatrix,
    state: u32, idx: [usize; NUM_LEAVES],
) -> f64 {
    f.at(state, idx) + b.at(state, idx) - f.log_likelihood
}

/// Log-posterior of the silent transition at `SILENT_TRANSITIONS[bucket]`
/// firing at multi-index `idx`. (`bucket` is the per-silent index, not
/// the global bucket index — same as the array offset in
/// SILENT_TRANSITIONS.) Returns -inf if either endpoint is unreachable.
pub fn silent_transition_log_posterior(
    f: &DPMatrix, b: &DPMatrix, lw: &[f64],
    bucket: usize, idx: [usize; NUM_LEAVES],
) -> f64 {
    let (src, dst, widx) = SILENT_TRANSITIONS[bucket];
    let cell = f.cell(idx);
    let fv = f.data[(src as usize) * f.total + cell];
    let bv = b.data[(dst as usize) * b.total + cell];
    if fv == f64::NEG_INFINITY || bv == f64::NEG_INFINITY { return f64::NEG_INFINITY; }
    fv + lw[widx as usize] + bv - f.log_likelihood
}

/// Log-posterior of an emitting bucket firing at multi-index `idx`.
/// `shard` indexes into EMIT_SHARDS, `i` is the offset within that shard.
/// "Firing at idx" means: state `src` at cell idx-delta transitions via
/// this bucket to state `dst` at cell idx, consuming observed[k][idx[k]-1]
/// for each k where deltas[k]==1. Returns -inf if infeasible (predecessor
/// out of range or symbols don't match).
pub fn emitting_transition_log_posterior(
    f: &DPMatrix, b: &DPMatrix, leaves: [&[u32]; NUM_LEAVES], lw: &[f64],
    shard: usize, i: usize, idx: [usize; NUM_LEAVES],
) -> f64 {
    let dvec = DELTA_VEC[shard];
    let mut prev = idx;
    for k in 0..NUM_LEAVES {
        if dvec[k] == 1 {
            if prev[k] == 0 { return f64::NEG_INFINITY; }
            prev[k] -= 1;
        }
    }
    let (src, dst, syms, widx) = EMIT_SHARDS[shard][i];
    for k in 0..NUM_LEAVES {
        if dvec[k] == 1 && leaves[k][idx[k] - 1] as i32 != syms[k] {
            return f64::NEG_INFINITY;
        }
    }
    let fv = f.at(src, prev);
    let bv = b.at(dst, idx);
    if fv == f64::NEG_INFINITY || bv == f64::NEG_INFINITY { return f64::NEG_INFINITY; }
    fv + lw[widx as usize] + bv - f.log_likelihood
}

/// Global bucket index for an emitting bucket: NUM_SILENT + offset.
#[inline]
pub fn emit_bucket_index(shard: usize, i: usize) -> usize {
    NUM_SILENT + EMIT_SHARD_BUCKET_OFFSET[shard] + i
}

)RUST";

  // ----- Forward-Backward expected counts -----
  f << R"RUST(
/// Forward-Backward expected transition counts: for each bucket b,
///   E[count(b)] = exp( lse_over_cells( f[src,prev] + lw[b] + b[dst,cell] ) − Z )
/// where Z = forward log-likelihood. Returned vector has length NUM_BUCKETS,
/// indexed in [0..NUM_SILENT) for silent buckets and [NUM_SILENT..NUM_BUCKETS)
/// for emitting buckets in shard order.
pub fn forward_backward_counts_with_log_weights(lw: &[f64], leaves: [&[u32]; NUM_LEAVES]) -> FBResult {
    let f = forward_matrix_with_log_weights(lw, leaves);
    let bm = backward_matrix_with_log_weights(lw, leaves);
    let log_z = f.log_likelihood;
    let total = f.total;
    let strides = f.strides;
    let lens = f.lens;
    let mut log_count = vec![f64::NEG_INFINITY; NUM_BUCKETS];

    // We sweep cells in lex order. At each cell:
    //  - Silent buckets (src,dst,widx): contribution at this cell is
    //    f[src,cell] + lw[widx] + b[dst,cell].
    //  - Emitting buckets at shard d: contribution at this cell is
    //    f[src, cell - delta] + lw[widx] + b[dst, cell], when feasible
    //    (idx[k] >= 1 for emitting positions, and observed match).
    //
    // We accumulate via lse into log_count[bucket].
    let mut idx: [usize; NUM_LEAVES] = [0; NUM_LEAVES];
    loop {
        let cell: usize = {
            let mut c = 0usize;
            for k in 0..NUM_LEAVES { c += idx[k] * strides[k]; }
            c
        };
        // Silent contributions at this cell.
        for (bi, &(src, dst, widx)) in SILENT_TRANSITIONS.iter().enumerate() {
            let fv = f.data[(src as usize) * total + cell];
            let bv = bm.data[(dst as usize) * total + cell];
            if fv != f64::NEG_INFINITY && bv != f64::NEG_INFINITY {
                let term = fv + lw[widx as usize] + bv;
                log_count[bi] = lse(log_count[bi], term);
            }
        }
        // Emitting contributions, sharded by delta vector.
        for d_idx in 0..NUM_DELTA {
            let dvec = DELTA_VEC[d_idx];
            let mut prev = cell;
            let mut feasible = true;
            for k in 0..NUM_LEAVES {
                if dvec[k] == 1 {
                    if idx[k] == 0 { feasible = false; break; }
                    prev -= strides[k];
                }
            }
            if !feasible { continue; }
            let mut observed: [i32; NUM_LEAVES] = [-1; NUM_LEAVES];
            for k in 0..NUM_LEAVES {
                if dvec[k] == 1 { observed[k] = leaves[k][idx[k] - 1] as i32; }
            }
            let bucket_base = NUM_SILENT + EMIT_SHARD_BUCKET_OFFSET[d_idx];
            for (i, &(src, dst, syms, widx)) in EMIT_SHARDS[d_idx].iter().enumerate() {
                let mut sym_ok = true;
                for k in 0..NUM_LEAVES {
                    if dvec[k] == 1 && observed[k] != syms[k] { sym_ok = false; break; }
                }
                if !sym_ok { continue; }
                let fv = f.data[(src as usize) * total + prev];
                let bv = bm.data[(dst as usize) * total + cell];
                if fv != f64::NEG_INFINITY && bv != f64::NEG_INFINITY {
                    let term = fv + lw[widx as usize] + bv;
                    let bidx = bucket_base + i;
                    log_count[bidx] = lse(log_count[bidx], term);
                }
            }
        }
        // advance idx
        let mut advanced = false;
        let mut k = NUM_LEAVES;
        while k > 0 {
            k -= 1;
            if idx[k] < lens[k] { idx[k] += 1; for j in (k+1)..NUM_LEAVES { idx[j] = 0; } advanced = true; break; }
        }
        if !advanced { break; }
    }

    let mut expected_counts = vec![0.0_f64; NUM_BUCKETS];
    for i in 0..NUM_BUCKETS {
        if log_count[i] == f64::NEG_INFINITY { expected_counts[i] = 0.0; }
        else { expected_counts[i] = (log_count[i] - log_z).exp(); }
    }
    FBResult { log_likelihood: log_z, expected_counts }
}

/// Forward-Backward expected transition counts (computes log-weights internally).
#[inline]
pub fn forward_backward_counts(p: &Params, leaves: [&[u32]; NUM_LEAVES]) -> FBResult {
    forward_backward_counts_with_log_weights(&precompute_log_weights(p), leaves)
}

)RUST";

  // ----- machine.json template + counts_to_machine_json helper -----
  // The codegen also writes machine.json next to lib.rs; the helper below
  // weaves expected_counts into it via `__C<idx>__` sentinel substitution.
  f << "/// Bucketed-machine JSON template (the same shape as the standard\n";
  f << "/// Machine Boss JSON). Each transition has an `expected_count`\n";
  f << "/// field whose value is a `__C<idx>__` placeholder which the\n";
  f << "/// `counts_to_machine_json` helper substitutes with the value\n";
  f << "/// from `FBResult::expected_counts[idx]`.\n";
  f << "pub const MACHINE_JSON: &str = include_str!(\"../machine.json\");\n\n";

  f << R"RUST(
impl FBResult {
    /// Render the codegen's bucketed machine as JSON, with the
    /// `expected_count` field on each transition filled in from
    /// `self.expected_counts`. Output is appended to `out`.
    pub fn to_machine_json(&self, out: &mut String) {
        let template = MACHINE_JSON;
        out.reserve(template.len() + self.expected_counts.len() * 24);
        let mut last = 0usize;
        let bytes = template.as_bytes();
        let mut i = 0usize;
        while i + 4 < bytes.len() {
            // Match marker pattern "__C<digits>__".
            if bytes[i] == b'_' && bytes[i+1] == b'_' && bytes[i+2] == b'C' {
                let nstart = i + 3;
                let mut nend = nstart;
                while nend < bytes.len() && bytes[nend].is_ascii_digit() { nend += 1; }
                if nend > nstart && nend + 1 < bytes.len() && bytes[nend] == b'_' && bytes[nend+1] == b'_' {
                    let n: usize = std::str::from_utf8(&bytes[nstart..nend]).unwrap().parse().unwrap();
                    out.push_str(&template[last..i]);
                    use std::fmt::Write;
                    write!(out, "{}", self.expected_counts[n]).ok();
                    last = nend + 2;
                    i = last;
                    continue;
                }
            }
            i += 1;
        }
        out.push_str(&template[last..]);
    }
}

)RUST";

  // ---- Write machine.json next to lib.rs ----
  //
  // This is the bucketed-composed machine in standard Machine Boss JSON
  // shape. Each transition has an `expected_count` field whose value is
  // a `__C<global_bucket_idx>__` placeholder that gets substituted by
  // FBResult::to_machine_json at runtime.
  //
  // Bucket index assignment (matches NUM_BUCKETS layout above):
  //   silents in declaration order:    0..NUM_SILENT
  //   emittings in shard order:        NUM_SILENT..NUM_BUCKETS
  {
    // Group buckets by source state for emission as state[].trans[].
    std::vector<std::vector<std::pair<size_t, bool>>> byState (N);  // (entry_idx, isEmitting)
    for (size_t i = 0; i < silent.size(); ++i)
      byState[silent[i].src].push_back ({i, false});
    // Build flat emit list in shard order to match the global bucket index.
    std::vector<size_t> emitFlat;
    for (size_t d = 0; d < deltaVecs.size(); ++d)
      for (size_t idx : emitByDelta.at (deltaVecs[d]))
        emitFlat.push_back (idx);
    for (size_t i = 0; i < emitFlat.size(); ++i)
      byState[emitting[emitFlat[i]].src].push_back ({i, true});

    std::ofstream mj (outputDir + "/machine.json");
    mj << std::setprecision(17);
    mj << "{\"state\":\n [";
    for (StateIndex s = 0; s < (StateIndex)N; ++s) {
      if (s) mj << ",\n  ";
      mj << "{\"n\":" << s;
      if (!byState[s].empty()) {
        mj << ",\n   \"trans\":[";
        bool first = true;
        for (const auto& kv : byState[s]) {
          if (!first) mj << ",\n            ";
          first = false;
          const Entry& e = kv.second ? emitting[emitFlat[kv.first]] : silent[kv.first];
          const size_t globalIdx = kv.second
              ? (silent.size() + kv.first)
              : kv.first;
          mj << "{\"to\":" << e.dst;
          if (!e.outToken.empty())
            mj << ",\"out\":" << json(e.outToken).dump();
          mj << ",\"weight\":" << WeightAlgebra::toJsonString (e.summedWeight);
          mj << ",\"expected_count\":__C" << globalIdx << "__";
          mj << "}";
        }
        mj << "]";
      }
      mj << "}";
    }
    mj << "\n ]";
    // Defs and cons (carried over from the composed machine, useful for
    // anyone wanting to evaluate the symbolic weight expressions).
    if (!sorted.funcs.defs.empty()) {
      mj << ",\n \"defs\":\n  ";
      WeightAlgebra::toJsonStream (mj, sorted.funcs.defs);
    }
    if (!sorted.cons.empty()) {
      mj << ",\n \"cons\":\n  ";
      sorted.cons.writeJson (mj);
    }
    mj << "\n}\n";
  }

  LogThisAt(2,"Wrote Rust crate to " << outputDir
            << " (states=" << N << ", leaves=" << L
            << ", alphabet=" << alph.size()
            << ", silent=" << silent.size() << ", emitting=" << emitting.size()
            << ", weights=" << weightsInOrder.size()
            << ", buckets=" << (silent.size() + emitting.size()) << ")" << std::endl);
}

// ---- Skeleton bake mode (Step-2 path; Increment 1: stub crate) ----------

namespace {

// Emit a string as a Rust raw string literal `r#"..."#`. JSON content
// almost always contains `"`, so we use at least one hash; we walk up if
// the content itself contains `"<run-of-hashes>` to avoid premature close.
std::string rustRawStringLit (const std::string& s) {
  size_t hashes = 1;
  for (size_t i = 0; i + 1 < s.size(); ++i) {
    if (s[i] == '"' && s[i+1] == '#') {
      size_t run = 1;
      size_t j = i + 2;
      while (j < s.size() && s[j] == '#') { ++run; ++j; }
      if (run >= hashes) hashes = run + 1;
    }
  }
  std::string sep (hashes, '#');
  return "r" + sep + "\"" + s + "\"" + sep;
}

}  // anonymous

void compileRustSkeleton (const Machine& M_skel, const Machine& T,
                          const std::string& tree_newick,
                          const std::string& time_param,
                          const std::string& outputDir) {
  mkdirP (outputDir);  // creates outputDir/ and outputDir/src/

  // Serialise M_skel and T as JSON strings to embed.
  std::ostringstream skelStream;
  M_skel.writeJson (skelStream);
  std::ostringstream tStream;
  T.writeJson (tStream);

  // Cargo.toml — minimal stub crate that depends on serde_json so the bakes
  // are usable. Crate name borrowed from existing convention.
  {
    std::ofstream f (outputDir + "/Cargo.toml");
    f << "[package]\n"
         "name = \"phylo_skeleton\"\n"
         "version = \"0.1.0\"\n"
         "edition = \"2021\"\n"
         "\n"
         "[lib]\n"
         "name = \"phylo_skeleton\"\n"
         "path = \"src/lib.rs\"\n"
         "\n"
         "[dependencies]\n"
         "serde_json = \"1\"\n"
         "\n"
         "[profile.release]\n"
         "opt-level = 3\n"
         "lto = true\n";
  }

  // src/weight_algebra.rs — Rust port of weight.h's WeightExpr evaluator.
  // Parses Machine Boss JSON weight expressions and evaluates them against
  // a Params map, with recursive resolution of `defs` references.
  {
    std::ofstream f (outputDir + "/src/weight_algebra.rs");
    f << R"RUST(//! Weight algebra evaluator. Rust port of `src/weight.cpp` in the
//! Machine Boss C++ codebase. Parses the JSON weight-expression form
//! used in machine.json and evaluates it against a Params map.
//!
//! JSON encoding (mirrors `WeightAlgebra::fromJson`):
//!   - JSON number              → constant
//!   - JSON string `"name"`     → parameter; resolved via `defs` first,
//!                                then `params` (for free parameters)
//!   - `{"log": expr}`          → ln(expr)
//!   - `{"exp": expr}`          → e^expr
//!   - `{"not": expr}`          → 1 - expr
//!   - `{"geomsum": expr}`      → 1 / (1 - expr)
//!   - `{"*": [a, b]}`          → a * b
//!   - `{"/": [a, b]}`          → a / b
//!   - `{"+": [a, b]}`          → a + b
//!   - `{"-": [a, b]}`          → a - b
//!   - `{"pow": [a, b]}`        → a.powf(b)   (output-side only in C++)
//!
//! `defs` are themselves WeightExprs that may transitively reference other
//! defs / params; we cycle-protect with a small `visiting` stack.

use std::collections::HashMap;
use serde_json::Value;

pub type Params = HashMap<String, f64>;
pub type Defs = HashMap<String, Value>;

/// Parse the `defs` section of a Machine Boss machine JSON into a Defs
/// map. Returns an empty map if `machine_json` has no defs.
pub fn parse_defs(machine_json: &Value) -> Defs {
    let mut defs = Defs::new();
    if let Some(d) = machine_json.get("defs").and_then(|d| d.as_object()) {
        for (k, v) in d.iter() {
            defs.insert(k.clone(), v.clone());
        }
    }
    defs
}

/// Evaluate a WeightExpr against the given parameter assignments and defs.
///
/// Panics on undefined names or cyclic def references.
pub fn evaluate(expr: &Value, params: &Params, defs: &Defs) -> f64 {
    let mut visiting: Vec<String> = Vec::new();
    eval_inner(expr, params, defs, &mut visiting)
}

/// Structural literal-1 check.  Mirror of `WeightAlgebra::isOne`.
pub fn is_one(w: &Value) -> bool {
    match w {
        Value::Bool(true) => true,
        Value::Number(n) => n.as_f64() == Some(1.0),
        _ => false,
    }
}

/// Structural literal-0 check.  Mirror of `WeightAlgebra::isZero`.
pub fn is_zero(w: &Value) -> bool {
    match w {
        Value::Null => true,
        Value::Bool(false) => true,
        Value::Number(n) => n.as_f64() == Some(0.0),
        _ => false,
    }
}

fn as_number(w: &Value) -> Option<f64> {
    match w {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Symbolic multiplication of two WeightExprs.  Mirror of
/// `WeightAlgebra::multiply`: applies the same identity/zero/numeric
/// short-circuits, otherwise returns `{"*": [l, r]}`.
pub fn multiply(l: &Value, r: &Value) -> Value {
    if is_one(l) { return r.clone(); }
    if is_one(r) { return l.clone(); }
    if is_zero(l) || is_zero(r) { return Value::from(0.0_f64); }
    if let (Some(a), Some(b)) = (as_number(l), as_number(r)) {
        return Value::from(a * b);
    }
    serde_json::json!({"*": [l.clone(), r.clone()]})
}

/// Symbolic addition.  Mirror of `WeightAlgebra::add` with the same
/// 0+x → x, num+num → num short-circuits.
pub fn add(l: &Value, r: &Value) -> Value {
    if is_zero(l) { return r.clone(); }
    if is_zero(r) { return l.clone(); }
    if let (Some(a), Some(b)) = (as_number(l), as_number(r)) {
        return Value::from(a + b);
    }
    serde_json::json!({"+": [l.clone(), r.clone()]})
}

/// Symbolic 1/(1-p).  Mirror of `WeightAlgebra::geometricSum`.  C++
/// emits `{"geomsum": p}` with no folding.
pub fn geometric_sum(p: &Value) -> Value {
    serde_json::json!({"geomsum": p.clone()})
}

fn eval_inner(expr: &Value, params: &Params, defs: &Defs, visiting: &mut Vec<String>) -> f64 {
    match expr {
        Value::Number(n) => n.as_f64().expect("WeightExpr number not f64-convertible"),
        Value::Bool(b) => if *b { 1.0 } else { 0.0 },
        Value::String(s) => {
            if visiting.iter().any(|v| v == s) {
                panic!("WeightExpr cyclic def reference: {}", s);
            }
            if let Some(d) = defs.get(s) {
                visiting.push(s.clone());
                let v = eval_inner(d, params, defs, visiting);
                visiting.pop();
                v
            } else if let Some(p) = params.get(s) {
                *p
            } else {
                panic!("WeightExpr unknown name: {}", s);
            }
        }
        Value::Object(map) => {
            // Single-keyed object; first key is the operator.
            let (op, arg) = map.iter().next()
                .expect("WeightExpr object has no opcode");
            match op.as_str() {
                "log" => eval_inner(arg, params, defs, visiting).ln(),
                "exp" => eval_inner(arg, params, defs, visiting).exp(),
                "not" => 1.0 - eval_inner(arg, params, defs, visiting),
                "geomsum" => 1.0 / (1.0 - eval_inner(arg, params, defs, visiting)),
                "*" | "/" | "+" | "-" | "pow" => {
                    let arr = arg.as_array().expect("binary op arg not an array");
                    assert_eq!(arr.len(), 2, "binary op arg has wrong arity: {}", op);
                    let l = eval_inner(&arr[0], params, defs, visiting);
                    let r = eval_inner(&arr[1], params, defs, visiting);
                    match op.as_str() {
                        "*" => l * r,
                        "/" => l / r,
                        "+" => l + r,
                        "-" => l - r,
                        "pow" => l.powf(r),
                        _ => unreachable!(),
                    }
                }
                _ => panic!("WeightExpr unknown opcode: {}", op),
            }
        }
        _ => panic!("WeightExpr unsupported JSON type: {:?}", expr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn empty_defs() -> Defs { Defs::new() }
    fn empty_params() -> Params { Params::new() }

    #[test]
    fn constants() {
        assert_eq!(evaluate(&json!(0.5), &empty_params(), &empty_defs()), 0.5);
        assert_eq!(evaluate(&json!(3),   &empty_params(), &empty_defs()), 3.0);
        assert_eq!(evaluate(&json!(true),  &empty_params(), &empty_defs()), 1.0);
        assert_eq!(evaluate(&json!(false), &empty_params(), &empty_defs()), 0.0);
    }

    #[test]
    fn param_lookup() {
        let mut p = Params::new();
        p.insert("x".into(), 2.5);
        assert_eq!(evaluate(&json!("x"), &p, &empty_defs()), 2.5);
    }

    #[test]
    fn arithmetic() {
        assert_eq!(evaluate(&json!({"+": [1.0, 2.0]}), &empty_params(), &empty_defs()), 3.0);
        assert_eq!(evaluate(&json!({"-": [5.0, 2.0]}), &empty_params(), &empty_defs()), 3.0);
        assert_eq!(evaluate(&json!({"*": [3.0, 4.0]}), &empty_params(), &empty_defs()), 12.0);
        assert_eq!(evaluate(&json!({"/": [10.0, 4.0]}), &empty_params(), &empty_defs()), 2.5);
        assert_eq!(evaluate(&json!({"pow": [2.0, 3.0]}), &empty_params(), &empty_defs()), 8.0);
    }

    #[test]
    fn unary_ops() {
        // ln(e) ≈ 1
        let v = evaluate(&json!({"log": std::f64::consts::E}), &empty_params(), &empty_defs());
        assert!((v - 1.0).abs() < 1e-12, "log(e) = {}", v);
        // exp(0) = 1
        assert_eq!(evaluate(&json!({"exp": 0.0}), &empty_params(), &empty_defs()), 1.0);
        // not(0.3) = 0.7
        let v = evaluate(&json!({"not": 0.3}), &empty_params(), &empty_defs());
        assert!((v - 0.7).abs() < 1e-12);
        // geomsum(0.5) = 1/(1-0.5) = 2
        let v = evaluate(&json!({"geomsum": 0.5}), &empty_params(), &empty_defs());
        assert!((v - 2.0).abs() < 1e-12);
    }

    #[test]
    fn def_resolution() {
        // pHalf = 0.5; pSquared = pHalf * pHalf
        let mut defs = Defs::new();
        defs.insert("pHalf".into(),    json!(0.5));
        defs.insert("pSquared".into(), json!({"*": ["pHalf", "pHalf"]}));
        let v = evaluate(&json!("pSquared"), &empty_params(), &defs);
        assert_eq!(v, 0.25);
    }

    #[test]
    fn def_with_param() {
        // pNoSub = exp(-mu * t)
        let mut defs = Defs::new();
        defs.insert("pNoSub".into(),
            json!({"exp": {"-": [0.0, {"*": ["mu", "t"]}]}}));
        let mut p = Params::new();
        p.insert("mu".into(), 2.0);
        p.insert("t".into(),  0.5);
        let v = evaluate(&json!("pNoSub"), &p, &defs);
        assert!((v - (-1.0_f64).exp()).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "cyclic")]
    fn cyclic_def_panics() {
        let mut defs = Defs::new();
        defs.insert("a".into(), json!("b"));
        defs.insert("b".into(), json!("a"));
        evaluate(&json!("a"), &empty_params(), &defs);
    }

    #[test]
    #[should_panic(expected = "unknown name")]
    fn unknown_name_panics() {
        evaluate(&json!("never_defined"), &empty_params(), &empty_defs());
    }

    /// Smoke test: parse the canonical TKF91 branch transducer's defs and
    /// evaluate `pSame` against concrete params; verify against the closed
    /// form pSame = pNoSub + pSub * pi where pi = 0.25 for JC. (This test
    /// only runs when the bake is present, so it lives in tests/ not here.)
    #[test]
    fn def_chain_via_arithmetic() {
        // pSub = 1 - pNoSub; with pNoSub = 0.5, pSub = 0.5.
        let mut defs = Defs::new();
        defs.insert("pNoSub".into(), json!(0.5));
        defs.insert("pSub".into(), json!({"not": "pNoSub"}));
        defs.insert("pSame".into(),
            json!({"+": ["pNoSub", {"*": ["pSub", 0.25]}]}));
        let v = evaluate(&json!("pSame"), &empty_params(), &defs);
        assert!((v - 0.625).abs() < 1e-12);  // 0.5 + 0.5 * 0.25
    }
}
)RUST";
  }

  // src/machine.rs — Machine struct + JSON ingest + rename_for_branch.
  // Mirrors src/machine.h's Machine + src/phylo_intersect.cpp's
  // renameForBranch.
  {
    std::ofstream f (outputDir + "/src/machine.rs");
    f << R"RUST(//! Machine (WFST) struct + JSON ingest + per-branch parameter renaming.
//!
//! Mirrors `src/machine.h`'s `Machine` and `src/phylo_intersect.cpp`'s
//! `renameForBranch`. JSON shape (subset of the Machine Boss schema):
//!
//! ```text
//! { "state": [ { "id": <Value>, "trans": [ { "to": <usize|String>,
//!                                            "in": <String?>,
//!                                            "out": <String?>,
//!                                            "weight": <WeightExpr?> },
//!                                          ... ] }, ... ],
//!   "defs": { "name": <WeightExpr>, ... },
//!   "cons": { "rate": [...], "prob": [...], "norm": [[...], ...] } }
//! ```

use std::collections::HashMap;
use serde_json::{Value, Map};

use crate::weight_algebra;

#[derive(Debug, Clone)]
pub struct Transition {
    pub to: usize,
    pub in_sym: String,
    pub out_sym: String,
    /// JSON WeightExpr; evaluate via `weight_algebra::evaluate`.
    pub weight: Value,
}

#[derive(Debug, Clone)]
pub struct State {
    /// Original `id` field (may be a string, array, or nested object).
    pub id: Value,
    pub trans: Vec<Transition>,
}

#[derive(Debug, Clone, Default)]
pub struct Constraints {
    pub rate: Vec<String>,
    pub prob: Vec<String>,
    pub norm: Vec<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct Machine {
    pub state: Vec<State>,
    pub defs: HashMap<String, Value>,
    pub cons: Constraints,
}

impl Machine {
    /// Parse a Machine from its JSON representation.
    pub fn from_json(v: &Value) -> Self {
        let states_arr = v.get("state")
            .and_then(|s| s.as_array())
            .expect("Machine JSON must have a `state` array");

        // Build a name -> index map first; transition `to` may reference state
        // names (string/array) rather than indices.
        let mut name_to_idx: HashMap<String, usize> = HashMap::new();
        for (i, s) in states_arr.iter().enumerate() {
            // Prefer explicit `n` (numeric) over `id` for index resolution.
            if let Some(n) = s.get("n").and_then(|x| x.as_u64()) {
                if (n as usize) != i {
                    // honour explicit n; but for our purposes we just use i.
                }
            }
            // Stringify `id` as a hashable name (Value's Display happens to be
            // the JSON serialisation, which is what the C++ code does too).
            if let Some(id) = s.get("id") {
                name_to_idx.insert(id.to_string(), i);
                // Also accept the bare string form for plain string ids.
                if let Some(s_str) = id.as_str() {
                    name_to_idx.insert(s_str.to_string(), i);
                }
            }
            name_to_idx.insert(format!("{}", i), i);
        }

        let mut state = Vec::with_capacity(states_arr.len());
        for s in states_arr.iter() {
            let id = s.get("id").cloned().unwrap_or(Value::Null);
            let trans_arr = s.get("trans").and_then(|t| t.as_array());
            let mut trans = Vec::new();
            if let Some(arr) = trans_arr {
                for t in arr.iter() {
                    let to = resolve_to(&t.get("to").expect("trans missing `to`"),
                                        &name_to_idx);
                    let in_sym = t.get("in").and_then(|x| x.as_str()).unwrap_or("").to_string();
                    let out_sym = t.get("out").and_then(|x| x.as_str()).unwrap_or("").to_string();
                    let weight = t.get("weight").cloned().unwrap_or(Value::from(1.0_f64));
                    trans.push(Transition { to, in_sym, out_sym, weight });
                }
            }
            state.push(State { id, trans });
        }

        let defs = if let Some(d) = v.get("defs").and_then(|d| d.as_object()) {
            d.iter().map(|(k, val)| (k.clone(), val.clone())).collect()
        } else {
            HashMap::new()
        };

        let cons = parse_cons(v.get("cons"));

        Machine { state, defs, cons }
    }

    /// Number of states.
    pub fn n_states(&self) -> usize { self.state.len() }

    /// Whether this machine references a free param named `name` anywhere
    /// (transition weights or def values, transitively via def chains).
    pub fn has_param(&self, name: &str) -> bool {
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for s in &self.state {
            for t in &s.trans {
                if walk_for_name(&t.weight, name, &self.defs, &mut seen) {
                    return true;
                }
            }
        }
        for v in self.defs.values() {
            if walk_for_name(v, name, &self.defs, &mut seen) { return true; }
        }
        false
    }
}

fn resolve_to(to: &Value, name_to_idx: &HashMap<String, usize>) -> usize {
    if let Some(n) = to.as_u64() { return n as usize; }
    if let Some(s) = to.as_str() {
        if let Some(i) = name_to_idx.get(s) { return *i; }
    }
    let key = to.to_string();
    *name_to_idx.get(&key)
        .unwrap_or_else(|| panic!("Cannot resolve `to` reference: {}", key))
}

fn parse_cons(v: Option<&Value>) -> Constraints {
    let mut c = Constraints::default();
    let Some(obj) = v.and_then(|v| v.as_object()) else { return c; };
    if let Some(arr) = obj.get("rate").and_then(|a| a.as_array()) {
        c.rate = arr.iter().filter_map(|x| x.as_str().map(String::from)).collect();
    }
    if let Some(arr) = obj.get("prob").and_then(|a| a.as_array()) {
        c.prob = arr.iter().filter_map(|x| x.as_str().map(String::from)).collect();
    }
    if let Some(arr) = obj.get("norm").and_then(|a| a.as_array()) {
        for g in arr.iter() {
            if let Some(group) = g.as_array() {
                let names: Vec<String> = group.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect();
                c.norm.push(names);
            }
        }
    }
    c
}

fn walk_for_name<'a>(v: &'a Value, target: &str, defs: &'a HashMap<String, Value>,
                     seen: &mut std::collections::HashSet<&'a str>) -> bool {
    match v {
        Value::String(s) => {
            if s == target { return true; }
            if seen.contains(s.as_str()) { return false; }
            if let Some(d) = defs.get(s) {
                seen.insert(s.as_str());
                return walk_for_name(d, target, defs, seen);
            }
            false
        }
        Value::Object(map) => {
            for val in map.values() {
                if walk_for_name(val, target, defs, seen) { return true; }
            }
            false
        }
        Value::Array(arr) => arr.iter().any(|x| walk_for_name(x, target, defs, seen)),
        _ => false,
    }
}

/// Substitute parameter names according to `subst` recursively in a
/// WeightExpr JSON value. Only string-valued names are substituted; numbers,
/// booleans, and structural ops are preserved verbatim.
pub fn bind(expr: &Value, subst: &HashMap<String, String>) -> Value {
    match expr {
        Value::String(s) => {
            if let Some(replacement) = subst.get(s) {
                Value::String(replacement.clone())
            } else {
                Value::String(s.clone())
            }
        }
        Value::Object(map) => {
            let mut out = Map::new();
            for (k, v) in map.iter() {
                out.insert(k.clone(), bind(v, subst));
            }
            Value::Object(out)
        }
        Value::Array(arr) => {
            Value::Array(arr.iter().map(|x| bind(x, subst)).collect())
        }
        other => other.clone(),
    }
}

/// Mirror of `phylo_intersect.cpp::renameForBranch`. Returns a copy of `m`
/// where (a) every transition-weight reference to `time_param` or to a
/// def-key is replaced with the suffixed name `<orig>[<node_name>]`, (b)
/// every def's value is similarly substituted and its key suffixed, and
/// (c) `cons.rate` / `cons.prob` / `cons.norm` entries equal to
/// `time_param` are suffixed.
pub fn rename_for_branch(m: &Machine, time_param: &str, node_name: &str) -> Machine {
    let suffix = format!("[{}]", node_name);
    let mut subst: HashMap<String, String> = HashMap::new();
    subst.insert(time_param.to_string(), format!("{}{}", time_param, suffix));
    for k in m.defs.keys() {
        subst.insert(k.clone(), format!("{}{}", k, suffix));
    }

    // Rewrite every transition weight.
    let mut new_state = Vec::with_capacity(m.state.len());
    for s in &m.state {
        let mut nt = Vec::with_capacity(s.trans.len());
        for t in &s.trans {
            nt.push(Transition {
                to: t.to,
                in_sym: t.in_sym.clone(),
                out_sym: t.out_sym.clone(),
                weight: bind(&t.weight, &subst),
            });
        }
        new_state.push(State { id: s.id.clone(), trans: nt });
    }

    // Rewrite defs (rebind values, suffix keys).
    let mut new_defs: HashMap<String, Value> = HashMap::new();
    for (k, v) in m.defs.iter() {
        new_defs.insert(format!("{}{}", k, suffix), bind(v, &subst));
    }

    // Rewrite cons. Only entries equal to time_param are suffixed; def keys
    // do not appear in cons.
    let mut new_cons = Constraints::default();
    let rename_vec = |v: &[String]| -> Vec<String> {
        v.iter().map(|s| if s == time_param { format!("{}{}", time_param, suffix) } else { s.clone() }).collect()
    };
    new_cons.rate = rename_vec(&m.cons.rate);
    new_cons.prob = rename_vec(&m.cons.prob);
    new_cons.norm = m.cons.norm.iter().map(|g| rename_vec(g)).collect();

    Machine { state: new_state, defs: new_defs, cons: new_cons }
}

/// Convenience: evaluate a transition's weight against a Params map and the
/// machine's defs. (Wraps `weight_algebra::evaluate`.)
pub fn eval_weight(m: &Machine, weight: &Value, params: &weight_algebra::Params) -> f64 {
    weight_algebra::evaluate(weight, params, &m.defs)
}

// ---- Pair-token encoder (--pair-json / JsonMode mirror) -----------------
//
// `Machine::encodePairToken` in JsonMode emits a two-element JSON array
// `[<a>, <b>]`, where each side is either an embedded JSON value (if the
// side string already parses as JSON, so nested intersections produce
// nested arrays) or a JSON string. The Rust port always uses this mode
// because (a) the codegen pipeline requires it (`--rust` requires
// `--pair-json` in the legacy boss flow), and (b) it round-trips through
// `forward::parse_token`.

fn encode_pair_side_json(s: &str) -> Value {
    if s.is_empty() { return Value::String(String::new()); }
    if let Ok(v) = serde_json::from_str::<Value>(s) {
        return v;
    }
    Value::String(s.to_string())
}

/// Mirror of `Machine::encodePairToken` in `--pair-json` mode.
pub fn encode_pair_token(a: &str, b: &str) -> String {
    if a.is_empty() && b.is_empty() { return String::new(); }
    let arr = Value::Array(vec![encode_pair_side_json(a), encode_pair_side_json(b)]);
    serde_json::to_string(&arr).expect("encode_pair_token: serialize")
}

// ---- Transducer composition ---------------------------------------------

/// Symbolic accumulator for transitions sharing the same (in, out, dest)
/// triple. Mirrors `TransAccumulator` in machine.cpp: when the same triple
/// is accumulated multiple times, the weights are added (via WeightAlgebra
/// add) — this is the natural Felsenstein-style sum-over-intermediate-symbol
/// that emerges from the compose inner loop.
#[derive(Default)]
struct TransAccumulator {
    by_key: std::collections::BTreeMap<(String, String, usize), Value>,
}

impl TransAccumulator {
    fn clear(&mut self) { self.by_key.clear(); }
    fn accumulate(&mut self, in_sym: &str, out_sym: &str, dest: usize, weight: &Value) {
        let key = (in_sym.to_string(), out_sym.to_string(), dest);
        match self.by_key.remove(&key) {
            Some(existing) => {
                let combined = weight_algebra::add(&existing, weight);
                self.by_key.insert(key, combined);
            }
            None => {
                self.by_key.insert(key, weight.clone());
            }
        }
    }
    fn into_transitions(self) -> Vec<Transition> {
        self.by_key.into_iter().map(|((in_sym, out_sym, to), weight)| Transition {
            to, in_sym, out_sym, weight
        }).collect()
    }
}

/// Mirror of `Machine::compose(first, second, ...)` from src/machine.cpp.
///
/// Cross-product state space (i, j); accessibility DFS from (0, 0) prunes
/// unreachable states; each kept state's outgoing transitions are computed
/// per the three classical compose cases and accumulated through a
/// `TransAccumulator` (so duplicates with the same (in, out, dest) sum
/// their weights — this is what produces the implicit Felsenstein sum).
///
/// The C++ post-processing chain
/// `.ergodicMachine().advanceSort().processCycles().ergodicMachine()` is
/// approximated here by a single `ergodic_machine()` call. `advance_sort`
/// and `process_cycles` ports are pending (Increment 4d/e); for inputs
/// without silent cycles this short pipeline produces a topologically valid
/// result.
pub fn compose(first: &Machine, second: &Machine) -> Machine {
    let second_w = if second.is_waiting_machine() {
        second.clone()
    } else {
        second.waiting_machine()
    };
    debug_assert!(second_w.is_waiting_machine());

    let i_states = first.state.len();
    let j_states = second_w.state.len();
    let n_pairs = i_states * j_states;
    let comp = |i: usize, j: usize| -> usize { i * j_states + j };

    // Accessibility DFS from (0, 0).
    let mut keep = vec![false; n_pairs];
    let mut to_visit: Vec<usize> = Vec::new();
    keep[0] = true;
    to_visit.push(0);
    while let Some(c) = to_visit.pop() {
        let i = c / j_states;
        let j = c % j_states;
        let msi = &first.state[i];
        let msj = &second_w.state[j];
        let mut dest_buf: Vec<usize> = Vec::new();
        if msj.waits() || msj.terminates() {
            for it in &msi.trans {
                if it.output_empty() {
                    dest_buf.push(comp(it.to, j));
                } else {
                    for jt in &msj.trans {
                        if it.out_sym == jt.in_sym {
                            dest_buf.push(comp(it.to, jt.to));
                        }
                    }
                }
            }
        } else {
            for jt in &msj.trans {
                dest_buf.push(comp(i, jt.to));
            }
        }
        for &d in &dest_buf {
            if !keep[d] {
                keep[d] = true;
                to_visit.push(d);
            }
        }
    }
    if !keep[n_pairs - 1] {
        // End state inaccessible; return an empty machine.
        return Machine { state: Vec::new(), defs: HashMap::new(),
                         cons: Constraints::default() };
    }

    // Build kept-state list (sorted) and comp→kept index map.
    let mut kept_states: Vec<usize> = (0..n_pairs).filter(|&c| keep[c]).collect();
    kept_states.sort();
    let mut comp_to_kept: Vec<usize> = vec![0; n_pairs];
    for (k, &c) in kept_states.iter().enumerate() {
        comp_to_kept[c] = k;
    }

    // Initialise composite states (state names + empty transition lists).
    let mut comp_state: Vec<State> = Vec::with_capacity(kept_states.len());
    for &c in &kept_states {
        let i = c / j_states;
        let j = c % j_states;
        // Composite state name: ordered 2-array (matches the post-fix C++).
        let name = Value::Array(vec![first.state[i].id.clone(),
                                     second_w.state[j].id.clone()]);
        comp_state.push(State { id: name, trans: Vec::new() });
    }

    // Compute transitions per kept state via TransAccumulator.
    let mut ta = TransAccumulator::default();
    for k in 0..kept_states.len() {
        let c = kept_states[k];
        let i = c / j_states;
        let j = c % j_states;
        let msi = &first.state[i];
        let msj = &second_w.state[j];
        ta.clear();
        if msj.waits() || msj.terminates() {
            for it in &msi.trans {
                if it.output_empty() {
                    let d = comp(it.to, j);
                    if keep[d] {
                        ta.accumulate(&it.in_sym, "", comp_to_kept[d], &it.weight);
                    }
                } else {
                    for jt in &msj.trans {
                        if it.out_sym == jt.in_sym {
                            let d = comp(it.to, jt.to);
                            if keep[d] {
                                let w = weight_algebra::multiply(&it.weight, &jt.weight);
                                ta.accumulate(&it.in_sym, &jt.out_sym, comp_to_kept[d], &w);
                            }
                        }
                    }
                }
            }
        } else {
            for jt in &msj.trans {
                let d = comp(i, jt.to);
                if keep[d] {
                    ta.accumulate("", &jt.out_sym, comp_to_kept[d], &jt.weight);
                }
            }
        }
        comp_state[k].trans = std::mem::take(&mut ta).into_transitions();
    }

    // Merge defs from both inputs (later writers win on key collision).
    let mut comp_defs = first.defs.clone();
    for (k, v) in &second_w.defs {
        comp_defs.insert(k.clone(), v.clone());
    }

    let raw = Machine {
        state: comp_state,
        defs: comp_defs,
        cons: first.cons.clone(),  // approximate: cons-merging is an open issue
    };
    // Mirror of C++ Machine::compose post-processing chain:
    //   compMachine.ergodicMachine().advanceSort().processCycles().ergodicMachine()
    raw.ergodic_machine()
        .advance_sort()
        .process_cycles()
        .ergodic_machine()
}

// ---- Transducer intersection --------------------------------------------

/// Whether any state has a transition with non-empty output. Used to
/// decide between dual-output (pair-token) and asymmetric-merge intersect
/// semantics.
fn has_nonempty_output(m: &Machine) -> bool {
    m.state.iter().any(|s| s.trans.iter().any(|t| !t.out_sym.is_empty()))
}

/// Mirror of `Machine::intersect(first, second, ...)` from src/machine.cpp.
///
/// Cross-product state space (i, j); transitions per state per the three
/// classical intersect cases (sync on INPUT, not output as in compose).
/// When both inputs have non-empty output alphabets, transitions emit pair
/// tokens encoded via `encode_pair_token`; otherwise the asymmetric-merge
/// fallback preserves whichever output is non-empty.
///
/// Unlike compose, the C++ intersect does NOT pre-filter the cross product
/// for accessibility — it builds all i*j states first, then relies on
/// `ergodic_machine` to prune. This port matches that ordering.
pub fn intersect(first: &Machine, second: &Machine) -> Machine {
    let second_w = if second.is_waiting_machine() {
        second.clone()
    } else {
        second.waiting_machine()
    };
    debug_assert!(second_w.is_waiting_machine());

    let dual_output = has_nonempty_output(first) && has_nonempty_output(&second_w);
    let pair_out = |a: &str, b: &str| -> String {
        if a.is_empty() && b.is_empty() { return String::new(); }
        if !dual_output {
            return if a.is_empty() { b.to_string() } else { a.to_string() };
        }
        encode_pair_token(a, b)
    };

    let i_states = first.state.len();
    let j_states = second_w.state.len();
    let n_pairs = i_states * j_states;
    let inter_state = |i: usize, j: usize| -> usize { i * j_states + j };

    let mut inter_state_vec: Vec<State> = Vec::with_capacity(n_pairs);
    for i in 0..i_states {
        for j in 0..j_states {
            let id = Value::Array(vec![first.state[i].id.clone(),
                                       second_w.state[j].id.clone()]);
            inter_state_vec.push(State { id, trans: Vec::new() });
        }
    }

    for i in 0..i_states {
        for j in 0..j_states {
            let msi = &first.state[i];
            let msj = &second_w.state[j];
            let trans_buf: &mut Vec<Transition> = &mut inter_state_vec[inter_state(i, j)].trans;
            if msj.waits() || msj.terminates() {
                for it in &msi.trans {
                    if it.input_empty() {
                        // Silent advance of first only; second stays at j.
                        let out_sym = if dual_output && !it.out_sym.is_empty() {
                            pair_out(&it.out_sym, "")
                        } else {
                            it.out_sym.clone()
                        };
                        trans_buf.push(Transition {
                            to: inter_state(it.to, j),
                            in_sym: it.in_sym.clone(),
                            out_sym,
                            weight: it.weight.clone(),
                        });
                    } else {
                        for jt in &msj.trans {
                            if it.in_sym == jt.in_sym {
                                trans_buf.push(Transition {
                                    to: inter_state(it.to, jt.to),
                                    in_sym: it.in_sym.clone(),
                                    out_sym: pair_out(&it.out_sym, &jt.out_sym),
                                    weight: weight_algebra::multiply(&it.weight, &jt.weight),
                                });
                            }
                        }
                    }
                }
            } else {
                // Second advances silently; first stays at i.
                for jt in &msj.trans {
                    let out_sym = if dual_output && !jt.out_sym.is_empty() {
                        pair_out("", &jt.out_sym)
                    } else {
                        jt.out_sym.clone()
                    };
                    trans_buf.push(Transition {
                        to: inter_state(i, jt.to),
                        in_sym: String::new(),
                        out_sym,
                        weight: jt.weight.clone(),
                    });
                }
            }
        }
    }

    let mut inter_defs = first.defs.clone();
    for (k, v) in &second_w.defs {
        inter_defs.insert(k.clone(), v.clone());
    }

    let raw = Machine {
        state: inter_state_vec,
        defs: inter_defs,
        cons: first.cons.clone(),
    };
    // Mirror of C++ Machine::intersect post-processing chain:
    //   interMachine.ergodicMachine().advanceSort().processCycles().ergodicMachine()
    raw.ergodic_machine()
        .advance_sort()
        .process_cycles()
        .ergodic_machine()
}

// ---- WFST state predicates and waitingMachine port ----------------------
//
// Port of MachineState::{terminates,exits_with_input,exits_without_input,
// waits,continues} and Machine::isWaitingMachine / Machine::waitingMachine
// from src/machine.cpp.

impl State {
    /// `trans.is_empty()` — no outgoing transitions.
    pub fn terminates(&self) -> bool { self.trans.is_empty() }
    /// At least one outgoing transition consumes an input symbol.
    pub fn exits_with_input(&self) -> bool {
        self.trans.iter().any(|t| !t.in_sym.is_empty())
    }
    /// At least one outgoing transition is silent on input.
    pub fn exits_without_input(&self) -> bool {
        self.trans.iter().any(|t| t.in_sym.is_empty())
    }
    /// Every outgoing transition consumes an input symbol (or none exist).
    pub fn waits(&self) -> bool { !self.exits_without_input() }
    /// Has outgoing transitions, none of which consume input.
    pub fn continues(&self) -> bool {
        !self.exits_with_input() && !self.terminates()
    }
}

/// `Machine::WAIT_TAG` constant from `src/machine.h` (`#define MachineWaitTag "wait"`).
pub const WAIT_TAG: &str = "wait";

/// `MachineCatLeftTag` / `MachineCatRightTag` from `src/machine.h`. Used by
/// `concatenate` to wrap left-side / right-side state names.
pub const MACHINE_CAT_LEFT_TAG:  &str = "concat-l";
pub const MACHINE_CAT_RIGHT_TAG: &str = "concat-r";

/// Mirror of `Transition::isSilent` from src/machine.h: both `in` and `out`
/// are empty.
impl Transition {
    pub fn is_silent(&self) -> bool {
        self.in_sym.is_empty() && self.out_sym.is_empty()
    }
    /// `in` is empty (silent on input).
    pub fn input_empty(&self) -> bool { self.in_sym.is_empty() }
    /// `out` is empty (silent on output).
    pub fn output_empty(&self) -> bool { self.out_sym.is_empty() }
}

/// Mirror of `WeightAlgebra::isOne`: structural check, only the literal
/// constants 1 (int / double / bool true). General algebraic
/// simplification is intentionally NOT performed here, matching the C++.
pub fn weight_is_one(w: &Value) -> bool {
    match w {
        Value::Bool(true) => true,
        Value::Number(n) => n.as_f64() == Some(1.0),
        _ => false,
    }
}

impl Machine {
    /// Every state either waits, continues, or terminates — i.e. no state
    /// has BOTH input-consuming and silent-input outgoing transitions.
    pub fn is_waiting_machine(&self) -> bool {
        self.state.iter().all(|s| s.waits() || s.continues())
    }

    /// Set of state indices that are both reachable from state 0 and can
    /// reach the end state (last state by index). Mirror of
    /// `Machine::accessibleStates`.
    pub fn accessible_states(&self) -> std::collections::BTreeSet<usize> {
        let n = self.state.len();
        if n == 0 { return Default::default(); }
        let start: usize = 0;
        let end: usize = n - 1;

        // Forward BFS from start.
        let mut from_start = vec![false; n];
        let mut q: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        from_start[start] = true;
        q.push_back(start);
        while let Some(c) = q.pop_front() {
            for t in &self.state[c].trans {
                if !from_start[t.to] {
                    from_start[t.to] = true;
                    q.push_back(t.to);
                }
            }
        }

        // Reverse BFS to end.
        let mut sources: Vec<Vec<usize>> = vec![Vec::new(); n];
        for s in 0..n {
            for t in &self.state[s].trans {
                sources[t.to].push(s);
            }
        }
        let mut to_end = vec![false; n];
        to_end[end] = true;
        q.clear();
        q.push_back(end);
        while let Some(c) = q.pop_front() {
            for &src in &sources[c] {
                if !to_end[src] {
                    to_end[src] = true;
                    q.push_back(src);
                }
            }
        }

        let mut acc = std::collections::BTreeSet::new();
        for s in 0..n {
            if from_start[s] && to_end[s] { acc.insert(s); }
        }
        acc
    }

    /// All states accessible AND end state is accessible. Mirror of
    /// `Machine::isErgodicMachine`.
    pub fn is_ergodic_machine(&self) -> bool {
        let acc = self.accessible_states();
        acc.len() == self.state.len() && self.state.len() > 0 && acc.contains(&(self.state.len() - 1))
    }

    /// Mirror of `Machine::ergodicMachine`: trim inaccessible states and
    /// collapse silent-weight-1 chains where the intermediate state has
    /// exactly one outgoing transition (silent, weight=1).
    pub fn ergodic_machine(&self) -> Self {
        if self.is_ergodic_machine() {
            return self.clone();
        }
        let n = self.state.len();
        let acc = self.accessible_states();
        let mut keep = vec![false; n];
        for &s in acc.iter() { keep[s] = true; }

        if !keep.last().copied().unwrap_or(false) {
            // End state inaccessible → return an empty/zero machine.
            // We approximate by returning a single-state empty machine.
            return Machine {
                state: Vec::new(),
                defs: HashMap::new(),
                cons: Constraints::default(),
            };
        }

        // null-equivalence: for each kept state s, follow chain forward
        // while state has exactly one outgoing trans that is silent and
        // weight-1. The final reached state is the equivalence target.
        let mut null_equiv: HashMap<usize, usize> = HashMap::new();
        for s in 0..n {
            if !keep[s] { continue; }
            let mut d = s;
            loop {
                let st = &self.state[d];
                if st.trans.len() == 1 && st.trans[0].is_silent() && weight_is_one(&st.trans[0].weight) {
                    d = st.trans[0].to;
                } else { break; }
            }
            if d != s { null_equiv.insert(s, d); }
        }

        // Build old2new mapping: kept-and-not-null-equiv states get fresh
        // indices in order; null-equiv states get the index of their target.
        let mut old2new = vec![0usize; n];
        let mut ns: usize = 0;
        for s in 0..n {
            if keep[s] && !null_equiv.contains_key(&s) {
                old2new[s] = ns;
                ns += 1;
            }
        }
        for s in 0..n {
            if keep[s] && null_equiv.contains_key(&s) {
                old2new[s] = old2new[null_equiv[&s]];
            }
        }

        if ns == 0 {
            return Machine {
                state: Vec::new(), defs: self.defs.clone(),
                cons: self.cons.clone(),
            };
        }

        // Emit kept-and-not-null-equiv states in order, with transitions
        // remapped via old2new (skipping any whose dest is not kept).
        let mut new_state: Vec<State> = Vec::with_capacity(ns);
        for s in 0..n {
            if keep[s] && !null_equiv.contains_key(&s) {
                let mut nt: Vec<Transition> = Vec::new();
                for t in &self.state[s].trans {
                    if keep[t.to] {
                        nt.push(Transition {
                            to: old2new[t.to],
                            in_sym: t.in_sym.clone(),
                            out_sym: t.out_sym.clone(),
                            weight: t.weight.clone(),
                        });
                    }
                }
                new_state.push(State { id: self.state[s].id.clone(), trans: nt });
            }
        }

        let out = Machine {
            state: new_state,
            defs: self.defs.clone(),
            cons: self.cons.clone(),
        };
        debug_assert!(out.is_ergodic_machine() || out.state.is_empty(),
                      "ergodic_machine failed to produce ergodic output");
        out
    }

    /// Mirror of `Machine::waitingMachine` (with the default
    /// `waitTag="wait"`, `continueTag=NULL`). Splits each "neither waits
    /// nor continues" state s into a continue-state (keeping s's silent
    /// transitions, plus a fresh silent edge to its paired wait-state)
    /// and a wait-state (taking s's input-consuming transitions, named
    /// `{"wait": <original-name>}`). The output state order interleaves
    /// each pair (c, w) at the original state's position, matching the
    /// C++ implementation's `for (StateIndex s : new2old)` reassembly.
    pub fn waiting_machine(&self) -> Self {
        if self.is_waiting_machine() {
            return self.clone();
        }
        let n0 = self.state.len();
        // `new_state` is a working buffer indexed by "newState index": entries
        // 0..n0 mirror the original states (replaced with continue-states for
        // splits); entries n0.. are wait-states appended on demand.
        let mut new_state: Vec<State> = self.state.clone();
        // `old2new[newstate_idx] = iteration_idx`: the position of that
        // newState entry in the final reassembled machine, in iteration order.
        let mut old2new: Vec<usize> = vec![0; n0];
        // `new2old[iter_idx] = newstate_idx`: lookup for reassembly.
        let mut new2old: Vec<usize> = Vec::with_capacity(n0);

        for s in 0..n0 {
            old2new[s] = new2old.len();
            new2old.push(s);
            let ms = &self.state[s];
            if !ms.waits() && !ms.continues() {
                let mut c_trans: Vec<Transition> = Vec::new();
                let mut w_trans: Vec<Transition> = Vec::new();
                for t in &ms.trans {
                    if t.in_sym.is_empty() {
                        c_trans.push(t.clone());
                    } else {
                        w_trans.push(t.clone());
                    }
                }
                // Wait-state goes at the end of `new_state` — capture its
                // index BEFORE the push.
                let wait_ns_idx = new_state.len();
                c_trans.push(Transition {
                    to: wait_ns_idx,
                    in_sym: String::new(),
                    out_sym: String::new(),
                    weight: Value::from(1.0_f64),
                });
                let mut wait_name = Map::new();
                wait_name.insert(WAIT_TAG.into(), ms.id.clone());
                let w_state = State {
                    id: Value::Object(wait_name),
                    trans: w_trans,
                };
                let c_state = State { id: ms.id.clone(), trans: c_trans };
                // The wait-state's iteration position is the next slot in
                // new2old; record it in old2new so wait_ns_idx maps there.
                old2new.push(new2old.len());
                new2old.push(wait_ns_idx);
                new_state[s] = c_state;
                new_state.push(w_state);
            }
        }
        // Reassemble in iteration order; remap each transition's `to` from
        // newState-index space to iteration-order (output) space.
        let mut wm_state: Vec<State> = Vec::with_capacity(new2old.len());
        for &ns_idx in new2old.iter() {
            let mut ms = new_state[ns_idx].clone();
            for t in ms.trans.iter_mut() {
                t.to = old2new[t.to];
            }
            wm_state.push(ms);
        }
        let out = Machine {
            state: wm_state,
            defs: self.defs.clone(),
            cons: self.cons.clone(),
        };
        debug_assert!(out.is_waiting_machine(),
                      "waiting_machine failed to produce a waiting machine");
        out
    }

    // ---- Predicate / counting helpers (Machine::nXxxBackTransitions in C++) ----

    /// Count of transitions s → t with t ≤ s, ignoring start state. Mirror
    /// of `Machine::nBackTransitions`.
    pub fn n_back_transitions(&self) -> usize {
        let mut n = 0;
        for s in 1..self.state.len() {
            for t in &self.state[s].trans {
                if t.to <= s { n += 1; }
            }
        }
        n
    }

    /// Count of silent (in.is_empty() && out.is_empty()) transitions
    /// s → t with t ≤ s. Mirror of `Machine::nSilentBackTransitions`.
    pub fn n_silent_back_transitions(&self) -> usize {
        let mut n = 0;
        for s in 1..self.state.len() {
            for t in &self.state[s].trans {
                if t.is_silent() && t.to <= s { n += 1; }
            }
        }
        n
    }

    /// Count of out-empty transitions s → t with t ≤ s. Mirror of
    /// `Machine::nEmptyOutputBackTransitions`.
    pub fn n_empty_output_back_transitions(&self) -> usize {
        let mut n = 0;
        for s in 1..self.state.len() {
            for t in &self.state[s].trans {
                if t.output_empty() && t.to <= s { n += 1; }
            }
        }
        n
    }

    /// True iff there are no silent backward (s → t with t ≤ s)
    /// transitions. Mirror of `Machine::isAdvancingMachine`.
    pub fn is_advancing_machine(&self) -> bool {
        self.n_silent_back_transitions() == 0
    }

    // ---- Trim silent back-transitions (DropSilentCycles strategy) ----

    /// Mirror of `Machine::dropSilentBackTransitions`: returns a copy with
    /// every silent t with t.to ≤ s removed (so no silent back-transitions
    /// remain). Used by `processCycles(DropSilentCycles)`.
    pub fn drop_silent_back_transitions(&self) -> Self {
        if self.is_advancing_machine() { return self.clone(); }
        let mut new_state = Vec::with_capacity(self.state.len());
        for (s, ms) in self.state.iter().enumerate() {
            let mut nt: Vec<Transition> = Vec::new();
            for t in &ms.trans {
                if !(t.is_silent() && t.to <= s) {
                    nt.push(t.clone());
                }
            }
            new_state.push(State { id: ms.id.clone(), trans: nt });
        }
        Machine { state: new_state, defs: self.defs.clone(), cons: self.cons.clone() }
    }

    // ---- advance_sort: state reordering to minimize backward edges ----

    /// Mirror of `Machine::advanceSort` with the default (silent
    /// back-transition) cost function. Reorders states to minimize the
    /// count of silent backward transitions; reverts to the original
    /// order if reordering doesn't help. Falls back to `padWithNullStates`
    /// + retry if the un-padded sort fails to eliminate all silent backs
    /// (matches C++).
    pub fn advance_sort(&self) -> Self {
        let count_back: fn(&Machine) -> usize = |m| m.n_silent_back_transitions();
        let must_advance: fn(&Transition) -> bool = |t| t.is_silent();
        self.advance_sort_inner(&count_back, &must_advance)
    }

    /// Generic version with custom counter / predicate (mirrors the
    /// templated C++ signature). The closures are passed by reference so
    /// they can be re-used in the recursive padding fallback.
    fn advance_sort_inner(
        &self,
        count_back: &dyn Fn(&Machine) -> usize,
        must_advance: &dyn Fn(&Transition) -> bool,
    ) -> Self {
        let n_silent_before = count_back(self);
        if n_silent_before == 0 {
            return self.clone();
        }

        let n = self.state.len();
        let start = 0usize;
        let end = if n > 0 { n - 1 } else { 0 };

        // Silent forward / backward incidence (excluding self-loops, edges
        // to start, and edges to end — matches C++).
        let mut sil_outgoing: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut sil_incoming: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut n_sil_in: Vec<i32> = vec![0; n];
        let mut n_sil_out: Vec<i32> = vec![0; n];
        if n >= 2 {
            for s in 1..(n - 1) {
                for t in &self.state[s].trans {
                    if must_advance(t) && t.to != s && t.to != end && t.to != start {
                        sil_outgoing[s].push(t.to);
                        sil_incoming[t.to].push(s);
                        n_sil_out[s] += 1;
                        n_sil_in[t.to] += 1;
                    }
                }
            }
        }

        // (n_in, n_in - n_out, idx) lexicographic comparator — the same
        // primary/secondary/tertiary keys C++ uses for its set.
        fn cmp_key(s: usize, n_in: &[i32], n_out: &[i32]) -> (i32, i32, usize) {
            (n_in[s], n_in[s] - n_out[s], s)
        }

        let mut order: Vec<usize> = Vec::with_capacity(n);
        let mut queue: Vec<usize> = Vec::new();

        // C++ remove/insert dance: only re-add a neighbour if it was
        // already in the queue — once it's been "added to order" it must
        // never re-enter.
        let do_add = |s: usize,
                      order: &mut Vec<usize>,
                      queue: &mut Vec<usize>,
                      n_sil_in: &mut [i32],
                      n_sil_out: &mut [i32],
                      sil_outgoing: &[Vec<usize>],
                      sil_incoming: &[Vec<usize>]| {
            order.push(s);
            for &next in &sil_outgoing[s] {
                let was_in = queue.iter().position(|&x| x == next);
                if let Some(idx) = was_in { queue.remove(idx); }
                n_sil_in[next] -= 1;
                if was_in.is_some() { queue.push(next); }
            }
            for &prev in &sil_incoming[s] {
                let was_in = queue.iter().position(|&x| x == prev);
                if let Some(idx) = was_in { queue.remove(idx); }
                n_sil_out[prev] -= 1;
                if was_in.is_some() { queue.push(prev); }
            }
            queue.sort_by_key(|&x| cmp_key(x, n_sil_in, n_sil_out));
        };

        do_add(start, &mut order, &mut queue,
               &mut n_sil_in, &mut n_sil_out, &sil_outgoing, &sil_incoming);

        if n > 1 {
            // Seed queue with all middle states.
            for s in 1..(n - 1) { queue.push(s); }
            queue.sort();
            queue.dedup();
            queue.sort_by_key(|&x| cmp_key(x, &n_sil_in, &n_sil_out));
            while !queue.is_empty() {
                let next = queue.remove(0);
                do_add(next, &mut order, &mut queue,
                       &mut n_sil_in, &mut n_sil_out, &sil_outgoing, &sil_incoming);
            }
            do_add(end, &mut order, &mut queue,
                   &mut n_sil_in, &mut n_sil_out, &sil_outgoing, &sil_incoming);
        }

        // Build old2new and check whether the order changed.
        let mut old2new = vec![0usize; n];
        let mut changed = false;
        for (new_idx, &old_idx) in order.iter().enumerate() {
            if old_idx != new_idx { changed = true; }
            old2new[old_idx] = new_idx;
        }

        let result = if !changed {
            self.clone()
        } else {
            let mut new_state: Vec<State> = Vec::with_capacity(n);
            for &s in &order {
                let mut ms = self.state[s].clone();
                for t in ms.trans.iter_mut() {
                    t.to = old2new[t.to];
                }
                new_state.push(ms);
            }
            Machine { state: new_state, defs: self.defs.clone(), cons: self.cons.clone() }
        };

        // NB: `n_silent_after` is the count on the *sorted* result (the
        // sort attempt's output), NOT on the restored-original `result`.
        // C++ deliberately keeps the post-sort number around as the
        // comparison baseline for the padding fallback below, so a
        // padded-and-unsorted machine that drops back to the pre-sort
        // count (e.g. 7 < 8 here) is preferred over the un-padded
        // failed-sort attempt. Updating this to `n_silent_before` would
        // make the comparison `n_silent_dummy < n_silent_before`, which
        // diverges from C++ when reordering increases the count.
        let n_silent_after = count_back(&result);
        let mut result = result;
        if n_silent_after >= n_silent_before {
            result = self.clone();
        }

        // C++ fallback: if the post-sort count is still > 0 and the
        // machine isn't already null-padded, try again with dummy null
        // start/end states. We accept the padded result iff its silent-
        // back count is strictly less than `n_silent_after` — the
        // post-sort count, NOT n_silent_before. This is the C++
        // `if (nSilentBackDummy < nSilentBackAfter)` check verbatim.
        if n_silent_after > 0 && !self.has_null_padding_states() {
            let with_dummy = self.pad_with_null_states();
            debug_assert!(with_dummy.has_null_padding_states(),
                          "pad_with_null_states failed to produce padded machine");
            let sorted_with_dummy = with_dummy.advance_sort_inner(count_back, must_advance);
            let n_silent_dummy = count_back(&sorted_with_dummy);
            if n_silent_dummy < n_silent_after {
                return sorted_with_dummy;
            }
        }

        result
    }

    /// Mirror of `Machine::padWithNullStates`: returns a copy with a null
    /// state prepended (if state 0 doesn't already look like one) and a
    /// null state appended (if the result still doesn't satisfy
    /// `has_null_padding_states`). Used by `advance_sort` as a fallback.
    pub fn pad_with_null_states(&self) -> Self {
        let n = self.state.len();
        let has_null_start = n > 0
            && self.state[0].trans.len() == 1
            && self.state[0].trans[0].is_silent();
        // C++ also rejects `hasNullStart` if any state has a transition
        // back to the start state; mirror that.
        let has_null_start = has_null_start && {
            let start = 0usize;
            !self.state.iter().flat_map(|s| s.trans.iter()).any(|t| t.to == start)
        };
        let dummy = Machine::null_machine();
        let mut result = if has_null_start {
            self.clone()
        } else {
            Machine::concatenate(&dummy, self,
                                 MACHINE_CAT_LEFT_TAG, MACHINE_CAT_RIGHT_TAG)
        };
        if !result.has_null_padding_states() {
            result = Machine::concatenate(&result, &dummy,
                                          MACHINE_CAT_LEFT_TAG, MACHINE_CAT_RIGHT_TAG);
        }
        result
    }

    // ---- process_cycles + advancing_machine (SumSilentCycles strategy) ----

    /// Mirror of `Machine::processCycles(SumSilentCycles)` — the only
    /// strategy used by compose / intersect's default post-processing.
    pub fn process_cycles(&self) -> Self {
        self.advancing_machine()
    }

    /// Mirror of `Machine::advancingMachine`: eliminate every silent
    /// backward transition by symbolic forward-substitution. Silent
    /// self-loops with weight w become a `geomsum(w) = 1/(1-w)` factor on
    /// the state's other outgoing edges. Silent back-edges to a strictly
    /// earlier state are folded into the destination's already-resolved
    /// outgoing transitions. The cost: weights become symbolic
    /// `WeightAlgebra` JSON expressions whose evaluation yields the
    /// summed-over-all-silent-paths weight.
    pub fn advancing_machine(&self) -> Self {
        if self.is_advancing_machine() { return self.clone(); }

        let n = self.state.len();
        let mut fwd_trans: std::collections::BTreeMap<(usize, usize), Vec<Transition>>
            = std::collections::BTreeMap::new();
        let mut n_elim = 0usize;
        let mut new_state: Vec<State> = Vec::with_capacity(n);

        for s in 0..n {
            // Recursive call to populate fwd_trans[(s, s)].
            update_fwd_trans(self, &mut fwd_trans, &mut n_elim, s, s);

            let mut ta = TransAccumulator::default();
            for t in fwd_trans.get(&(s, s)).cloned().unwrap_or_default() {
                ta.accumulate(&t.in_sym, &t.out_sym, t.to, &t.weight);
            }
            let et = std::mem::take(&mut ta).into_transitions();

            // Factor out self-loops: any silent self-loop at s contributes
            // a geomsum(w) factor to all other outgoing weights.
            let mut exit_self: Value = Value::from(1.0_f64);
            let mut surviving: Vec<Transition> = Vec::new();
            for t in et {
                if t.is_silent() && t.to == s {
                    exit_self = weight_algebra::geometric_sum(&t.weight);
                } else {
                    surviving.push(t);
                }
            }
            if !weight_algebra::is_one(&exit_self) {
                for t in surviving.iter_mut() {
                    t.weight = weight_algebra::multiply(&exit_self, &t.weight);
                }
            }
            // Record post-self-loop result back into fwd_trans for use by
            // later updateFwdTrans calls (mirrors C++ `fwdTrans[s][s] = ams.trans;`).
            fwd_trans.insert((s, s), surviving.clone());
            new_state.push(State { id: self.state[s].id.clone(), trans: surviving });
        }

        let am = Machine { state: new_state, defs: self.defs.clone(), cons: self.cons.clone() };
        debug_assert!(am.is_advancing_machine(),
                      "advancing_machine failed to eliminate silent back transitions");
        am
    }

    // ---- Null-padding helpers (used by C++ advance_sort fallback) ----

    /// 1-state machine with no transitions. Mirror of `Machine::null`.
    pub fn null_machine() -> Self {
        Machine {
            state: vec![State { id: Value::Null, trans: Vec::new() }],
            defs: HashMap::new(),
            cons: Constraints::default(),
        }
    }

    /// Mirror of `Machine::concatenate` (default left/right tags). Both
    /// inputs must have ≥ 1 state.
    pub fn concatenate(left: &Machine, right: &Machine, left_tag: &str, right_tag: &str) -> Self {
        assert!(!left.state.is_empty() && !right.state.is_empty(),
                "concatenate: both machines must have ≥ 1 state");

        let mut state: Vec<State> = Vec::with_capacity(left.state.len() + right.state.len());
        // Left states with name wrapped as ["leftTag", original-name].
        for ms in &left.state {
            let id = if ms.id.is_null() { ms.id.clone() }
                     else { Value::Array(vec![Value::String(left_tag.to_string()), ms.id.clone()]) };
            state.push(State { id, trans: ms.trans.clone() });
        }
        // Right states with name wrapped + transition `to` shifted by left.size.
        let shift = left.state.len();
        for ms in &right.state {
            let id = if ms.id.is_null() { ms.id.clone() }
                     else { Value::Array(vec![Value::String(right_tag.to_string()), ms.id.clone()]) };
            let mut nt = ms.trans.clone();
            for t in nt.iter_mut() { t.to += shift; }
            state.push(State { id, trans: nt });
        }
        // Bridge: silent w=1 from left's end to right's start.
        let left_end = left.state.len() - 1;
        let right_start = shift; // right's state 0 lives at index shift
        state[left_end].trans.push(Transition {
            to: right_start,
            in_sym: String::new(),
            out_sym: String::new(),
            weight: Value::from(1.0_f64),
        });

        // Merge defs (later writers win on collision, matching `import`).
        let mut defs = left.defs.clone();
        for (k, v) in &right.defs { defs.insert(k.clone(), v.clone()); }
        Machine { state, defs, cons: left.cons.clone() }
    }

    /// Mirror of `Machine::hasNullPaddingStates`: state 0 has exactly one
    /// outgoing edge (silent) and the end state has exactly one incoming
    /// edge from any state, also silent — and no transitions to start.
    pub fn has_null_padding_states(&self) -> bool {
        let n = self.state.len();
        if n == 0 { return false; }
        let s0 = &self.state[0];
        if !(s0.trans.len() == 1 && s0.trans[0].is_silent()) { return false; }
        let end = n - 1;
        if !self.state[end].trans.is_empty() { return false; }
        let mut null_to_end = 0;
        for ms in &self.state {
            for t in &ms.trans {
                if t.to == 0 { return false; }
                if t.to == end {
                    if !t.is_silent() { return false; }
                    null_to_end += 1;
                }
            }
        }
        null_to_end == 1
    }
}

// ---- updateFwdTrans helper (free function — mirrors C++) ----------------

/// Mirror of `updateFwdTrans` in machine.cpp. Populates
/// `fwd_trans[(i, new_min)]` with the set of effective transitions from
/// state `i` after eliminating all silent back-transitions to states <
/// `new_min`. Recursive in `new_min` (decrementing) and in `j` for each
/// silent transition i → j with j < new_min.
fn update_fwd_trans(
    machine: &Machine,
    fwd_trans: &mut std::collections::BTreeMap<(usize, usize), Vec<Transition>>,
    n_elim: &mut usize,
    i: usize,
    new_min: usize,
) {
    if fwd_trans.contains_key(&(i, new_min)) { return; }

    let old_trans: Vec<Transition> = if new_min > i {
        update_fwd_trans(machine, fwd_trans, n_elim, i, new_min - 1);
        fwd_trans.get(&(i, new_min - 1)).cloned().unwrap_or_default()
    } else if new_min == i {
        machine.state[new_min].trans.clone()
    } else {
        // new_min < i: should not happen at top-level entry; treat as empty.
        Vec::new()
    };

    let mut new_fwd: Vec<Transition> = Vec::new();
    for t_ij in &old_trans {
        if !t_ij.is_silent() {
            new_fwd.push(t_ij.clone());
        } else {
            let j = t_ij.to;
            if j >= new_min {
                new_fwd.push(t_ij.clone());
            } else {
                if i != j {
                    update_fwd_trans(machine, fwd_trans, n_elim, j, new_min);
                }
                let inner: Vec<Transition> = if i == j {
                    old_trans.clone()
                } else {
                    fwd_trans.get(&(j, new_min)).cloned().unwrap_or_default()
                };
                for t_jk in &inner {
                    let w = weight_algebra::multiply(&t_ij.weight, &t_jk.weight);
                    new_fwd.push(Transition {
                        to: t_jk.to,
                        in_sym: t_jk.in_sym.clone(),
                        out_sym: t_jk.out_sym.clone(),
                        weight: w,
                    });
                }
                if i > j { *n_elim += 1; }
            }
        }
    }
    fwd_trans.insert((i, new_min), new_fwd);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tiny_machine_json() -> Value {
        json!({
            "state": [
                { "id": "begin",
                  "trans": [
                      { "to": 1, "weight": "pSame" },
                      { "to": 2, "in": "x", "out": "y", "weight": {"*": ["pDiff", "time"]} }
                  ]
                },
                { "id": "mid",
                  "trans": [ { "to": 2 } ]
                },
                { "id": "end" }
            ],
            "defs": {
                "pSame": {"+": ["pNoSub", {"*": ["pSub", 0.25]}]},
                "pNoSub": {"exp": {"*": [-1.0, "time"]}},
                "pSub": {"not": "pNoSub"},
                "pDiff": {"*": ["pSub", 0.25]}
            },
            "cons": {
                "rate": ["time", "delRate"],
                "prob": [],
                "norm": [["pi_A", "pi_C", "pi_G", "pi_T"]]
            }
        })
    }

    #[test]
    fn parse_basic() {
        let m = Machine::from_json(&tiny_machine_json());
        assert_eq!(m.n_states(), 3);
        assert_eq!(m.state[0].trans.len(), 2);
        assert_eq!(m.state[0].trans[0].to, 1);
        assert_eq!(m.state[0].trans[1].in_sym, "x");
        assert_eq!(m.state[0].trans[1].out_sym, "y");
        assert_eq!(m.defs.len(), 4);
        assert_eq!(m.cons.rate, vec!["time".to_string(), "delRate".to_string()]);
    }

    #[test]
    fn has_param_walks_defs() {
        let m = Machine::from_json(&tiny_machine_json());
        assert!(m.has_param("time"));     // referenced by pNoSub
        assert!(!m.has_param("absent"));
    }

    #[test]
    fn rename_for_branch_suffixes_keys_and_refs() {
        let m = Machine::from_json(&tiny_machine_json());
        let m_a = rename_for_branch(&m, "time", "A");

        // Defs are renamed: pSame -> pSame[A], etc.
        assert!(m_a.defs.contains_key("pSame[A]"));
        assert!(m_a.defs.contains_key("pNoSub[A]"));
        assert!(!m_a.defs.contains_key("pSame"));

        // Inside pSame[A], references to pNoSub / pSub are suffixed.
        let psame = m_a.defs.get("pSame[A]").unwrap();
        let psame_str = psame.to_string();
        assert!(psame_str.contains("pNoSub[A]"), "pSame[A] body: {}", psame_str);
        assert!(psame_str.contains("pSub[A]"),   "pSame[A] body: {}", psame_str);

        // pNoSub[A] body references time[A] (since `time` was the time-param).
        let pnosub = m_a.defs.get("pNoSub[A]").unwrap().to_string();
        assert!(pnosub.contains("time[A]"), "pNoSub[A] body: {}", pnosub);

        // Transition weights have references substituted: state[0].trans[0]
        // had weight "pSame" -> "pSame[A]"; trans[1] had {"*": ["pDiff", "time"]}
        // -> {"*": ["pDiff[A]", "time[A]"]}.
        let w0 = m_a.state[0].trans[0].weight.to_string();
        assert_eq!(w0, "\"pSame[A]\"");
        let w1 = m_a.state[0].trans[1].weight.to_string();
        assert!(w1.contains("pDiff[A]"));
        assert!(w1.contains("time[A]"));

        // cons: time -> time[A]; delRate stays; norm group stays.
        assert_eq!(m_a.cons.rate, vec!["time[A]".to_string(), "delRate".to_string()]);
        assert_eq!(m_a.cons.norm[0],
                   vec!["pi_A".to_string(), "pi_C".to_string(),
                        "pi_G".to_string(), "pi_T".to_string()]);
    }

    #[test]
    fn predicates() {
        let m = Machine::from_json(&tiny_machine_json());
        // state[0] = "begin" has both silent (to mid) and consuming (to end with in/out)
        assert!(!m.state[0].waits());
        assert!(!m.state[0].continues());
        assert!(!m.state[0].terminates());
        // state[1] = "mid" has only silent transition
        assert!(!m.state[1].waits());
        assert!(m.state[1].continues());
        // state[2] = "end" has no transitions
        assert!(m.state[2].terminates());
        assert!(m.state[2].waits()); // vacuously
    }

    #[test]
    fn waiting_machine_round_trips_already_waiting() {
        // Pre-build a machine that's already waiting: every state either
        // waits or continues.
        let m = Machine::from_json(&json!({
            "state": [
                { "id": "S", "trans": [ {"to": 1, "in": "x", "out": "y"} ] },
                { "id": "E" }
            ]
        }));
        assert!(m.is_waiting_machine());
        let wm = m.waiting_machine();
        assert_eq!(wm.n_states(), m.n_states());
    }

    #[test]
    fn waiting_machine_splits_mixed_state() {
        // Single-state-mixed: state 0 has both silent (to 2) and consuming (to 1).
        let m = Machine::from_json(&json!({
            "state": [
                { "id": "S", "trans": [
                    {"to": 1, "in": "x", "out": "y", "weight": "wMatch"},
                    {"to": 2, "weight": "wSilent"}
                ] },
                { "id": "M", "trans": [ {"to": 2} ] },
                { "id": "E" }
            ]
        }));
        assert!(!m.is_waiting_machine());
        let wm = m.waiting_machine();
        assert!(wm.is_waiting_machine());
        // S split into c (still named "S", silent transitions) and w
        // (named {"wait": "S"}, consuming transitions).
        assert!(wm.n_states() > m.n_states());
        // c got an extra silent edge to w.
        // Check w's id: object with single "wait" key.
        let w_idx = wm.state.iter().position(|s| {
            if let Value::Object(o) = &s.id {
                o.contains_key(WAIT_TAG)
            } else { false }
        }).expect("expected at least one wait-state");
        // w should hold the consuming transition with in:'x'.
        assert!(wm.state[w_idx].trans.iter().any(|t| t.in_sym == "x"));
    }

    #[test]
    fn weight_is_one_basics() {
        assert!(weight_is_one(&json!(1.0)));
        assert!(weight_is_one(&json!(1)));
        assert!(weight_is_one(&json!(true)));
        assert!(!weight_is_one(&json!(0.0)));
        assert!(!weight_is_one(&json!(0.5)));
        assert!(!weight_is_one(&json!("p")));
        assert!(!weight_is_one(&json!({"+": [1, 0]}))); // not structurally simplified
    }

    #[test]
    fn accessible_states_filters_unreachable() {
        // 4 states: 0 -> 1 -> 3, plus orphan state 2 (unreachable from 0).
        let m = Machine::from_json(&json!({
            "state": [
                { "id": "S",  "trans": [{"to": 1}] },
                { "id": "M",  "trans": [{"to": 3}] },
                { "id": "X",  "trans": [{"to": 3}] },
                { "id": "E" }
            ]
        }));
        let acc = m.accessible_states();
        assert_eq!(acc, [0, 1, 3].iter().copied().collect());
        assert!(!m.is_ergodic_machine()); // state 2 inaccessible
    }

    #[test]
    fn ergodic_machine_short_circuits_when_ergodic() {
        // C++ short-circuits: if the input is already ergodic, the chain-
        // collapse pass does NOT fire. So a trivially-collapsible-looking
        // input is returned unchanged.
        let m = Machine::from_json(&json!({
            "state": [
                { "id": "S", "trans": [{"to": 1}] },
                { "id": "A", "trans": [{"to": 2}] },
                { "id": "E" }
            ]
        }));
        assert!(m.is_ergodic_machine());
        let em = m.ergodic_machine();
        assert_eq!(em.n_states(), m.n_states());
    }

    #[test]
    fn ergodic_machine_trims_and_collapses_chain() {
        // 0 -> 1 -> 3 silent-weight-1 chain, plus orphan state 2 (unreachable).
        // Non-ergodic → trim path runs → both states 0,1 chain-collapse to 3.
        // Result: 1 state.
        let m = Machine::from_json(&json!({
            "state": [
                { "id": "S",  "trans": [{"to": 1}] },
                { "id": "M",  "trans": [{"to": 3}] },
                { "id": "X",  "trans": [{"to": 3}] },
                { "id": "E" }
            ]
        }));
        assert!(!m.is_ergodic_machine());
        let em = m.ergodic_machine();
        assert_eq!(em.n_states(), 1);
    }

    #[test]
    fn ergodic_machine_trims_but_chain_blocked_by_branching() {
        // 0 -> 1 (silent w=1) and 0 -> 3 (silent w=p) both reach end via
        // different paths. State 0 has TWO outgoing transitions, so it
        // doesn't satisfy the chain-collapse condition (trans.len()==1).
        // Plus orphan state 2. Expected: trim drops 2; states 0, 1, 3 kept;
        // chain at state 1 collapses (1→3) so 1 maps to 3's slot. Result:
        // 2 states (0 and 3).
        let m = Machine::from_json(&json!({
            "state": [
                { "id": "S", "trans": [{"to": 1}, {"to": 3, "weight": "p"}] },
                { "id": "A", "trans": [{"to": 3}] },
                { "id": "X", "trans": [{"to": 3}] },
                { "id": "E" }
            ]
        }));
        assert!(!m.is_ergodic_machine());
        let em = m.ergodic_machine();
        // S kept (no chain collapse because 2 outgoing); A null-equiv to E;
        // X dropped (unreachable); E kept. → 2 states.
        assert_eq!(em.n_states(), 2);
        assert!(em.is_ergodic_machine());
        // Both of S's outgoing transitions remap to E (slot 1):
        // - {"to": 1} (originally to A) → A null-equivs to E → slot 1
        // - {"to": 3, "weight": "p"} (originally to E) → slot 1
        for t in &em.state[0].trans {
            assert_eq!(t.to, 1);
        }
    }

    #[test]
    fn waiting_machine_on_baked_t_yields_waiting() {
        // Real TKF91 branch transducer: states like "begin" have both silent
        // (begin→insert/wait) AND no consuming transitions; "match" is the
        // only state with consuming transitions and they all consume. So T
        // should already be a waiting machine. Verify the transformation
        // is a no-op.
        let m = Machine::from_json(&tiny_machine_json());
        let wm = m.waiting_machine();
        assert!(wm.is_waiting_machine());
        // For tiny_machine_json (which has the mixed state[0]), splits happen.
        assert_ne!(wm.n_states(), m.n_states());
    }

    #[test]
    fn compose_simple_two_state_pair() {
        // Two trivial single-state-with-self-loop machines.
        let m1 = Machine::from_json(&json!({
            "state": [
                {"id": "S", "trans": [{"to": 1, "in": "x", "out": "y", "weight": "p"}]},
                {"id": "E"}
            ]
        }));
        let m2 = Machine::from_json(&json!({
            "state": [
                {"id": "S", "trans": [{"to": 1, "in": "y", "out": "z", "weight": "q"}]},
                {"id": "E"}
            ]
        }));
        let c = compose(&m1, &m2);
        // Cross-product reachable from (0,0) with end-state accessible:
        // (0,0) -> (1,1) via y-symbol sync. So 2 states total.
        assert!(c.n_states() >= 2);
        // First state's name is the array [m1.state[0].id, m2.state[0].id].
        if let Value::Array(name) = &c.state[0].id {
            assert_eq!(name.len(), 2);
            assert_eq!(name[0], json!("S"));
            assert_eq!(name[1], json!("S"));
        } else {
            panic!("expected array state name, got {:?}", c.state[0].id);
        }
    }

    #[test]
    fn compose_collapses_degenerate_transitions() {
        // m1 has TWO match transitions S→E for the same (in, out).
        // m2 trivially echoes. Compose should collapse the duplicates by
        // summing their weights.
        let m1 = Machine::from_json(&json!({
            "state": [
                {"id": "S", "trans": [
                    {"to": 1, "in": "x", "out": "y", "weight": "p"},
                    {"to": 1, "in": "x", "out": "y", "weight": "q"}
                ]},
                {"id": "E"}
            ]
        }));
        let m2 = Machine::from_json(&json!({
            "state": [
                {"id": "S", "trans": [{"to": 1, "in": "y", "out": "y", "weight": 1.0}]},
                {"id": "E"}
            ]
        }));
        let c = compose(&m1, &m2);
        // The two duplicate transitions in m1 collapse into one in c with
        // weight = p + q (after multiply with m2's weight 1).
        let s0 = &c.state[0];
        assert_eq!(s0.trans.len(), 1, "expected collapsed; got {:?}", s0.trans);
        // Weight should be {"+": [p, q]} or simplified equivalent.
        let w = &s0.trans[0].weight;
        let s = w.to_string();
        assert!(s.contains("p") && s.contains("q"), "weight string: {}", s);
    }

    #[test]
    fn compose_evaluates_consistently_against_eval() {
        // m1 = identity-on-x with weight p. m2 = identity-on-x with weight q.
        // After compose: weight on the combined transition should evaluate
        // to p * q.
        let m1 = Machine::from_json(&json!({
            "state": [
                {"id": "S", "trans": [{"to": 1, "in": "x", "out": "x", "weight": "p"}]},
                {"id": "E"}
            ]
        }));
        let m2 = Machine::from_json(&json!({
            "state": [
                {"id": "S", "trans": [{"to": 1, "in": "x", "out": "x", "weight": "q"}]},
                {"id": "E"}
            ]
        }));
        let c = compose(&m1, &m2);
        let mut p = weight_algebra::Params::new();
        p.insert("p".into(), 0.4);
        p.insert("q".into(), 0.5);
        // Find the single emit transition.
        let emit = c.state.iter()
            .flat_map(|s| s.trans.iter())
            .find(|t| !t.in_sym.is_empty())
            .expect("an emit transition should exist");
        let v = weight_algebra::evaluate(&emit.weight, &p, &c.defs);
        assert!((v - 0.4 * 0.5).abs() < 1e-15, "v={}", v);
    }

    #[test]
    fn pair_token_basic() {
        // JsonMode: empty pair → empty string; otherwise a 2-element JSON
        // array. Sides that already parse as JSON are spliced in (so nested
        // intersections produce nested arrays).
        assert_eq!(encode_pair_token("", ""), "");
        assert_eq!(encode_pair_token("A", "B"), "[\"A\",\"B\"]");
        // A side that already parses as JSON (a 2-element array) is
        // spliced in as a value, not embedded as a string.
        assert_eq!(encode_pair_token("[\"A\",\"X\"]", "B"), "[[\"A\",\"X\"],\"B\"]");
        // Empty side → embedded as JSON empty string ("").
        assert_eq!(encode_pair_token("", "B"), "[\"\",\"B\"]");
    }

    #[test]
    fn intersect_simple_two_machines() {
        // Two trivial transducers, both consume input "x" and emit "y" / "z".
        // After intersect (sync on INPUT "x"), produce one transition with
        // input "x" and output pair-token "y,z".
        let m1 = Machine::from_json(&json!({
            "state": [
                {"id": "S", "trans": [{"to": 1, "in": "x", "out": "y", "weight": "p"}]},
                {"id": "E"}
            ]
        }));
        let m2 = Machine::from_json(&json!({
            "state": [
                {"id": "S", "trans": [{"to": 1, "in": "x", "out": "z", "weight": "q"}]},
                {"id": "E"}
            ]
        }));
        let i = intersect(&m1, &m2);
        // Find the emit transition in the result.
        let emit = i.state.iter()
            .flat_map(|s| s.trans.iter())
            .find(|t| !t.in_sym.is_empty())
            .expect("expected an emit transition");
        assert_eq!(emit.in_sym, "x");
        // JsonMode pair-token encoding: ["y","z"].
        assert_eq!(emit.out_sym, "[\"y\",\"z\"]");
    }

    #[test]
    fn intersect_evaluates_consistently() {
        // Composed weight should be p * q after intersect.
        let m1 = Machine::from_json(&json!({
            "state": [
                {"id": "S", "trans": [{"to": 1, "in": "x", "out": "y", "weight": "p"}]},
                {"id": "E"}
            ]
        }));
        let m2 = Machine::from_json(&json!({
            "state": [
                {"id": "S", "trans": [{"to": 1, "in": "x", "out": "z", "weight": "q"}]},
                {"id": "E"}
            ]
        }));
        let i = intersect(&m1, &m2);
        let mut p = weight_algebra::Params::new();
        p.insert("p".into(), 0.4);
        p.insert("q".into(), 0.5);
        let emit = i.state.iter()
            .flat_map(|s| s.trans.iter())
            .find(|t| !t.in_sym.is_empty())
            .expect("expected an emit transition");
        let v = weight_algebra::evaluate(&emit.weight, &p, &i.defs);
        assert!((v - 0.4 * 0.5).abs() < 1e-15, "v={}", v);
    }

    #[test]
    fn rename_for_branch_evaluates_consistently() {
        // Evaluating pSame on T at (time=0.5, ...) should equal evaluating
        // pSame[A] on T_renamed at (time[A]=0.5, ...).
        let m   = Machine::from_json(&tiny_machine_json());
        let m_a = rename_for_branch(&m, "time", "A");

        let mut p0 = weight_algebra::Params::new();
        p0.insert("time".into(), 0.5);
        let v0 = weight_algebra::evaluate(&Value::String("pSame".into()), &p0, &m.defs);

        let mut p1 = weight_algebra::Params::new();
        p1.insert("time[A]".into(), 0.5);
        let v1 = weight_algebra::evaluate(&Value::String("pSame[A]".into()), &p1, &m_a.defs);

        assert!((v0 - v1).abs() < 1e-15, "v0={} v1={}", v0, v1);
    }

    // ---- advance_sort + process_cycles ----

    #[test]
    fn n_silent_back_transitions_counts_correctly() {
        // S(0) silent → A(1), A silent → A (self-loop), A in:x → E(2). The
        // self-loop is a silent back-transition (dest=1 ≤ src=1); plus the
        // start-state's outgoing edge does NOT count (loop starts at s=1).
        let m = Machine::from_json(&json!({
            "state": [
                {"id": "S", "trans": [{"to": 1}]},
                {"id": "A", "trans": [
                    {"to": 1, "weight": 0.5},
                    {"to": 2, "in": "x", "out": "y"}
                ]},
                {"id": "E"}
            ]
        }));
        assert_eq!(m.n_silent_back_transitions(), 1);
        assert!(!m.is_advancing_machine());
    }

    #[test]
    fn drop_silent_back_transitions_strips_self_loop() {
        let m = Machine::from_json(&json!({
            "state": [
                {"id": "S", "trans": [{"to": 1}]},
                {"id": "A", "trans": [
                    {"to": 1, "weight": 0.5},
                    {"to": 2, "in": "x", "out": "y"}
                ]},
                {"id": "E"}
            ]
        }));
        let dropped = m.drop_silent_back_transitions();
        assert!(dropped.is_advancing_machine());
        assert_eq!(dropped.state[1].trans.len(), 1);
        assert_eq!(dropped.state[1].trans[0].to, 2);
    }

    #[test]
    fn advance_sort_noop_when_already_advancing() {
        // Linear chain — already topologically sorted.
        let m = Machine::from_json(&json!({
            "state": [
                {"id": "S", "trans": [{"to": 1}]},
                {"id": "A", "trans": [{"to": 2, "in": "x", "out": "y"}]},
                {"id": "E"}
            ]
        }));
        assert!(m.is_advancing_machine());
        let sorted = m.advance_sort();
        // No change — same n_states and same transitions.
        assert_eq!(sorted.n_states(), m.n_states());
    }

    #[test]
    fn advance_sort_reorders_to_remove_back_edge() {
        // 4-state machine: S=0, B=1, A=2, E=3. Edges:
        //   S → A (silent)
        //   A → B (silent forward in original order: A=2 → B=1, BACK)
        //   B → E (silent)
        // Reordering [S, A, B, E] → indices [0, 2, 1, 3]: A→B becomes
        // forward (A at index 1, B at index 2).
        let m = Machine::from_json(&json!({
            "state": [
                {"id": "S", "trans": [{"to": 2}]},
                {"id": "B", "trans": [{"to": 3}]},
                {"id": "A", "trans": [{"to": 1}]},
                {"id": "E"}
            ]
        }));
        assert_eq!(m.n_silent_back_transitions(), 1);
        let sorted = m.advance_sort();
        assert_eq!(sorted.n_silent_back_transitions(), 0);
    }

    #[test]
    fn process_cycles_collapses_silent_self_loop_via_geomsum() {
        // Self-loop A→A with weight p; A→E loud. After advancingMachine:
        // A→E should be multiplied by geomsum(p) = 1/(1-p).
        let m = Machine::from_json(&json!({
            "state": [
                {"id": "S", "trans": [{"to": 1}]},
                {"id": "A", "trans": [
                    {"to": 1, "weight": "p"},
                    {"to": 2, "in": "x", "out": "y", "weight": 1.0}
                ]},
                {"id": "E"}
            ]
        }));
        assert_eq!(m.n_silent_back_transitions(), 1);
        let am = m.process_cycles();
        assert!(am.is_advancing_machine());
        // State 1 should have exactly one outgoing transition (the loud one),
        // weight = geomsum(p) * 1 = {"geomsum": "p"}.
        assert_eq!(am.state[1].trans.len(), 1);
        let t = &am.state[1].trans[0];
        assert_eq!(t.to, 2);
        assert_eq!(t.in_sym, "x");
        // Evaluate at p=0.5: 1/(1-0.5) = 2.0.
        let mut params = weight_algebra::Params::new();
        params.insert("p".into(), 0.5);
        let w = weight_algebra::evaluate(&t.weight, &params, &am.defs);
        assert!((w - 2.0).abs() < 1e-15, "geomsum(0.5) = {}", w);
    }

    #[test]
    fn process_cycles_eliminates_two_state_silent_cycle() {
        // Silent cycle A↔B with back-edge B→A (silent), plus B→E (loud).
        // After advancingMachine:
        // - From state A's perspective, the path A→B→A→ ... → B → E is
        //   summed: A's effective outgoing ⊃ B→E with combined weight
        //   geomsum(p*q) * (p * 1) = p / (1 - p*q).
        let m = Machine::from_json(&json!({
            "state": [
                {"id": "S", "trans": [{"to": 1}]},
                {"id": "A", "trans": [{"to": 2, "weight": "p"}]},
                {"id": "B", "trans": [
                    {"to": 1, "weight": "q"},
                    {"to": 3, "in": "x", "out": "y", "weight": 1.0}
                ]},
                {"id": "E"}
            ]
        }));
        let processed = m.process_cycles();
        assert!(processed.is_advancing_machine(),
                "process_cycles must eliminate all silent back-transitions");

        // Compose with ergodic_machine (final stage of compose pipeline) to
        // drop any newly inaccessible / collapsible states.
        let compacted = processed.ergodic_machine();
        // Evaluate the start→end log-weight by running the (now silent-
        // back-free) DP. Easier: pluck out a path to E and verify weight.
        // Find the loud emit transition.
        let emit = compacted.state.iter()
            .flat_map(|s| s.trans.iter())
            .find(|t| !t.in_sym.is_empty())
            .expect("expected an emit transition");

        // Evaluate the whole start→end weight by tracing: follow weight-1
        // silent edges from start until we hit a state with a unique loud
        // out-edge or a non-trivial silent expansion.
        // The structurally most reliable check: compute start state's exit
        // via single-source DP over the (now advancing) silent subgraph.
        let mut params = weight_algebra::Params::new();
        params.insert("p".into(), 0.4);
        params.insert("q".into(), 0.5);
        // Simpler check: the compose-pipeline-final machine is small. We
        // expect total weight from 0 → end via emit = p / (1 - p*q) (the
        // closed form for the silent-cycle reduction). Run a tiny in-Rust
        // forward over symbolic weights then evaluate.
        let n = compacted.n_states();
        let mut f = vec![Value::Null; n];
        f[0] = Value::from(1.0_f64);
        // We assume the topology is now advancing, so a single forward
        // sweep suffices.
        let mut values: Vec<f64> = vec![0.0; n];
        values[0] = 1.0;
        for s in 0..n {
            for t in &compacted.state[s].trans {
                let w = weight_algebra::evaluate(&t.weight, &params, &compacted.defs);
                if t.in_sym.is_empty() && t.out_sym.is_empty() {
                    values[t.to] += values[s] * w;
                }
            }
        }
        // Now apply the single emit transition: contribution to end is
        // values[emit.from] * eval(emit.weight). Find the source state.
        let (src, t) = compacted.state.iter().enumerate()
            .flat_map(|(i, s)| s.trans.iter().map(move |t| (i, t)))
            .find(|(_, t)| !t.in_sym.is_empty())
            .unwrap();
        let _ = emit; // silence unused warning
        let w_emit = weight_algebra::evaluate(&t.weight, &params, &compacted.defs);
        let path_weight = values[src] * w_emit;
        // Closed-form: p / (1 - p*q) = 0.4 / (1 - 0.2) = 0.5.
        assert!((path_weight - 0.5).abs() < 1e-12,
                "expected 0.5, got {} (values={:?})", path_weight, values);
    }
}
)RUST";
  }

  // src/phylo.rs — Newick parser + buildSubtree recursion.
  // Mirrors the public surface of src/phylo_intersect.cpp: parse a Newick
  // string into a PhyloTree, then walk it recursively, calling
  // machine::compose / intersect / rename_for_branch to produce M_full.
  {
    std::ofstream f (outputDir + "/src/phylo.rs");
    f << R"RUST(//! Newick parser + phylo-intersect recursion. Rust port of
//! `src/phylo_intersect.cpp`. Given a branch transducer T (already parsed
//! into a `Machine`), a Newick tree, and the time-parameter name, returns
//! the phylo-composed Machine that mirrors the legacy `phyloIntersect`.

use serde_json::Value;
use crate::machine::{Machine, State, Transition, compose, intersect, rename_for_branch};

#[derive(Debug, Clone)]
pub struct PhyloNode {
    pub name: String,
    pub children: Vec<usize>,
    pub parent: Option<usize>,
    pub branch_length: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct PhyloTree {
    pub nodes: Vec<PhyloNode>,
    pub root: usize,
}

impl PhyloTree {
    pub fn parse_newick(s: &str) -> Self {
        let mut p = NewickParser { src: s.as_bytes(), pos: 0,
                                    nodes: Vec::new() };
        let root = p.parse_subtree(None);
        p.skip_ws();
        if p.pos < p.src.len() && p.src[p.pos] == b';' { p.pos += 1; }
        PhyloTree { nodes: p.nodes, root }
    }

    pub fn is_leaf(&self, i: usize) -> bool { self.nodes[i].children.is_empty() }
}

struct NewickParser<'a> {
    src: &'a [u8],
    pos: usize,
    nodes: Vec<PhyloNode>,
}

impl<'a> NewickParser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.src.len() && (self.src[self.pos] as char).is_whitespace() {
            self.pos += 1;
        }
    }
    fn peek(&mut self, c: u8) -> bool {
        self.skip_ws();
        self.pos < self.src.len() && self.src[self.pos] == c
    }
    fn consume(&mut self, c: u8) -> bool {
        if self.peek(c) { self.pos += 1; true } else { false }
    }
    fn expect(&mut self, c: u8) {
        if !self.consume(c) {
            panic!("Newick parse error: expected '{}' at position {}", c as char, self.pos);
        }
    }
    fn is_name_char(c: u8) -> bool {
        !matches!(c, b'(' | b')' | b',' | b':' | b';' | b'\'' | b'[' | b']')
            && !(c as char).is_whitespace()
    }
    fn parse_name(&mut self) -> String {
        self.skip_ws();
        let mut name = String::new();
        if self.pos < self.src.len() && self.src[self.pos] == b'\'' {
            self.pos += 1;
            while self.pos < self.src.len() && self.src[self.pos] != b'\'' {
                name.push(self.src[self.pos] as char);
                self.pos += 1;
            }
            if self.pos == self.src.len() {
                panic!("Newick parse error: unterminated quoted name");
            }
            self.pos += 1;
        } else {
            while self.pos < self.src.len() && Self::is_name_char(self.src[self.pos]) {
                name.push(self.src[self.pos] as char);
                self.pos += 1;
            }
        }
        name
    }
    fn parse_branch_length(&mut self) -> Option<f64> {
        self.skip_ws();
        if !self.consume(b':') { return None; }
        self.skip_ws();
        let start = self.pos;
        if self.pos < self.src.len() && (self.src[self.pos] == b'+' || self.src[self.pos] == b'-') {
            self.pos += 1;
        }
        while self.pos < self.src.len() {
            let c = self.src[self.pos];
            if (c as char).is_ascii_digit() || c == b'.' || c == b'e' || c == b'E'
                || c == b'+' || c == b'-' {
                self.pos += 1;
            } else { break; }
        }
        if self.pos == start {
            panic!("Newick parse error: expected branch length number at position {}", start);
        }
        let s = std::str::from_utf8(&self.src[start..self.pos]).unwrap();
        Some(s.parse().expect("Newick parse error: bad branch length"))
    }
    fn parse_subtree(&mut self, parent: Option<usize>) -> usize {
        let self_idx = self.nodes.len();
        self.nodes.push(PhyloNode {
            name: String::new(),
            children: Vec::new(),
            parent,
            branch_length: None,
        });
        self.skip_ws();
        if self.peek(b'(') {
            self.expect(b'(');
            let first = self.parse_subtree(Some(self_idx));
            self.nodes[self_idx].children.push(first);
            while self.peek(b',') {
                self.expect(b',');
                let next = self.parse_subtree(Some(self_idx));
                self.nodes[self_idx].children.push(next);
            }
            self.expect(b')');
        }
        let nm = self.parse_name();
        self.nodes[self_idx].name = nm;
        if let Some(bl) = self.parse_branch_length() {
            self.nodes[self_idx].branch_length = Some(bl);
        }
        self_idx
    }
}

// ---- Helpers for buildSubtree -------------------------------------------

/// Sorted set of distinct non-empty output symbols across all transitions.
pub fn output_alphabet(m: &Machine) -> Vec<String> {
    let mut set: std::collections::BTreeSet<String> = Default::default();
    for s in &m.state {
        for t in &s.trans {
            if !t.out_sym.is_empty() {
                set.insert(t.out_sym.clone());
            }
        }
    }
    set.into_iter().collect()
}

/// Mirror of `Machine::wildEcho(symbols)`: 1-state machine with a self-loop
/// transition (sym → sym, weight 1) for each symbol.
pub fn wild_echo(symbols: &[String]) -> Machine {
    let mut trans = Vec::with_capacity(symbols.len());
    for sym in symbols {
        trans.push(Transition {
            to: 0,
            in_sym: sym.clone(),
            out_sym: sym.clone(),
            weight: Value::from(1.0_f64),
        });
    }
    let id = Value::Array(symbols.iter().map(|s| Value::String(s.clone())).collect());
    Machine {
        state: vec![State { id, trans }],
        defs: Default::default(),
        cons: Default::default(),
    }
}

// ---- Phylo recursion ----------------------------------------------------

fn branch_transducer_for_child(tree: &PhyloTree, child_v: usize,
                               t: &Machine, time_param: &str,
                               rename_time: bool) -> Machine {
    let child = &tree.nodes[child_v];
    let t_copy = if rename_time {
        rename_for_branch(t, time_param, &child.name)
    } else {
        t.clone()
    };
    let sub = build_subtree(tree, child_v, t, time_param, rename_time);
    compose(&t_copy, &sub)
}

fn build_subtree(tree: &PhyloTree, v: usize,
                 t: &Machine, time_param: &str, rename_time: bool) -> Machine {
    let node = &tree.nodes[v];
    if node.children.is_empty() {
        // Leaf: wildEcho over T's output alphabet. (Leaf clamps not yet
        // ported — clamped-leaf workflow is a follow-up.)
        return wild_echo(&output_alphabet(t));
    }
    if node.children.len() == 1 {
        return branch_transducer_for_child(tree, node.children[0],
                                            t, time_param, rename_time);
    }
    // Degree ≥ 2: fold-left intersect.
    let mut acc = branch_transducer_for_child(tree, node.children[0],
                                               t, time_param, rename_time);
    for &c in &node.children[1..] {
        let sib = branch_transducer_for_child(tree, c, t, time_param, rename_time);
        acc = intersect(&acc, &sib);
    }
    acc
}

/// Top-level entry point: build the phylo-composed Machine from a branch
/// transducer T, a parsed PhyloTree, and the per-branch time-parameter
/// name. Mirrors `phyloIntersect` in src/phylo_intersect.cpp (without leaf
/// clamps and without `phylo-no-felsenstein` switching, both of which are
/// follow-ups).
pub fn phylo_intersect(t: &Machine, tree: &PhyloTree, time_param: &str) -> Machine {
    let rename_time = t.has_param(time_param);
    if rename_time {
        // Validate that every non-root node has a non-empty unique name
        // (matches the C++ guard in phyloIntersect).
        let mut seen = std::collections::HashSet::new();
        for (i, n) in tree.nodes.iter().enumerate() {
            if i == tree.root { continue; }
            assert!(!n.name.is_empty(),
                    "phylo: branch transducer has parameter \"{}\" but node {} has no name",
                    time_param, i);
            assert!(seen.insert(n.name.clone()),
                    "phylo: duplicate node name \"{}\"", n.name);
        }
    }
    build_subtree(tree, tree.root, t, time_param, rename_time)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newick_parse_binary() {
        let tree = PhyloTree::parse_newick("(A:0.1,B:0.2)P;");
        assert_eq!(tree.nodes.len(), 3);
        let p = &tree.nodes[tree.root];
        assert_eq!(p.name, "P");
        assert_eq!(p.children.len(), 2);
        assert_eq!(tree.nodes[p.children[0]].name, "A");
        assert_eq!(tree.nodes[p.children[0]].branch_length, Some(0.1));
        assert_eq!(tree.nodes[p.children[1]].name, "B");
        assert_eq!(tree.nodes[p.children[1]].branch_length, Some(0.2));
    }

    #[test]
    fn newick_parse_quartet() {
        let tree = PhyloTree::parse_newick("((A,B)P,(C,D)Q)R;");
        assert_eq!(tree.nodes.len(), 7);
        let r = &tree.nodes[tree.root];
        assert_eq!(r.name, "R");
        assert_eq!(r.children.len(), 2);
    }

    #[test]
    fn wild_echo_shape() {
        let alph = vec!["A".to_string(), "C".to_string(), "G".to_string(), "T".to_string()];
        let we = wild_echo(&alph);
        assert_eq!(we.n_states(), 1);
        assert_eq!(we.state[0].trans.len(), 4);
        for t in &we.state[0].trans {
            assert_eq!(t.in_sym, t.out_sym);
            assert_eq!(t.to, 0);
        }
    }

    #[test]
    fn output_alphabet_is_sorted_unique() {
        let m = Machine::from_json(&serde_json::json!({
            "state": [
                {"id": 0, "trans": [
                    {"to": 1, "in": "x", "out": "B"},
                    {"to": 1, "in": "x", "out": "A"},
                    {"to": 1, "in": "x", "out": "B"}
                ]},
                {"id": 1}
            ]
        }));
        assert_eq!(output_alphabet(&m), vec!["A".to_string(), "B".to_string()]);
    }
}
)RUST";
  }

  // src/forward.rs — multidim Forward DP over a Machine in symbolic-weight
  // form. Mirrors `multidim_forward` in t/rust/_phylo_ref.py: same DP
  // recurrences, same exact log_sum_exp via `(lo - hi).exp().ln_1p()`. This
  // is the runtime side of Increment 6b — see compileRust (the older
  // codegen) for the JIT-friendly straight-line variant.
  {
    std::ofstream f (outputDir + "/src/forward.rs");
    f << R"RUST(//! Multidim Forward DP over a `machine::Machine` whose transition
//! weights are symbolic `WeightAlgebra` JSON expressions. Returns the
//! log-likelihood from the start state at the all-zero cell to the end
//! state at the all-leaves-consumed cell.
//!
//! Bit-exact mirror of `t/rust/_phylo_ref.py::multidim_forward` (the
//! reference Python implementation). Same outer iteration order, same
//! exact `log_sum_exp`, same emit-then-silent within-cell ordering.

use serde_json::Value;
use crate::machine::Machine;
use crate::weight_algebra::{Params, evaluate};

/// Exact log_sum_exp via `(lo - hi).exp().ln_1p()`. Symmetric and
/// `-inf`-safe (returns the other operand if either side is `-inf`).
#[inline]
pub fn lse(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY { return b; }
    if b == f64::NEG_INFINITY { return a; }
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

/// Parse a pair-token string into a `Value` tree of leaf strings, mirroring
/// `parseTokenJson` in src/rust_codegen.cpp and `parse_tok` in
/// `_phylo_ref.py`. JSON-looking tokens are parsed as JSON; bare names
/// (the L=1 case) are returned as a single-leaf `Value::String`.
fn parse_token(s: &str) -> Value {
    if s.is_empty() { return Value::String(String::new()); }
    let c = s.as_bytes()[0];
    if c == b'[' || c == b'"' || c == b'-' || c.is_ascii_digit() {
        if let Ok(v) = serde_json::from_str::<Value>(s) {
            return canonicalize_token(v);
        }
    }
    Value::String(s.to_string())
}

fn canonicalize_token(v: Value) -> Value {
    match v {
        Value::Array(arr) => Value::Array(arr.into_iter().map(canonicalize_token).collect()),
        Value::String(s) => Value::String(s),
        Value::Number(n) => Value::String(n.to_string()),
        Value::Bool(b)   => Value::String(b.to_string()),
        Value::Null      => Value::String(String::new()),
        Value::Object(_) => panic!("unexpected object in pair token: {:?}", v),
    }
}

/// Merge two pair-token shapes: at each position, the deeper structure
/// wins. An empty leaf at an array position is allowed and is replaced by
/// the array shape from the other side. Symbol values are forgotten in
/// the result (only shape is preserved).
fn merge_shape(a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Array(aa), Value::Array(bb)) => {
            assert_eq!(aa.len(), bb.len(),
                       "pair-token arity mismatch: {} vs {}", aa.len(), bb.len());
            Value::Array(aa.iter().zip(bb.iter()).map(|(x, y)| merge_shape(x, y)).collect())
        }
        (Value::Array(_), Value::String(s)) => {
            assert!(s.is_empty(), "pair-token shape conflict (array vs non-empty leaf)");
            a.clone()
        }
        (Value::String(s), Value::Array(_)) => {
            assert!(s.is_empty(), "pair-token shape conflict (array vs non-empty leaf)");
            b.clone()
        }
        _ => Value::String(String::new()),
    }
}

fn count_leaves(v: &Value) -> usize {
    if let Value::Array(arr) = v { arr.iter().map(count_leaves).sum() } else { 1 }
}

/// Walk `tok` against the canonical `tmpl`. For each leaf position of
/// `tmpl` (in left-to-right traversal order), record the emitted symbol
/// (or empty for ε).
fn decode(tok: &Value, tmpl: &Value, out: &mut Vec<String>) {
    if let Value::Array(t_arr) = tmpl {
        if let Value::Array(tok_arr) = tok {
            assert_eq!(tok_arr.len(), t_arr.len(),
                       "token arity mismatch with template");
            for (ti, tt) in tok_arr.iter().zip(t_arr.iter()) {
                decode(ti, tt, out);
            }
        } else {
            // Token is a leaf — must be empty (means "all leaves under
            // this subtree are silent in this token").
            let s = tok.as_str().unwrap_or("");
            assert!(s.is_empty(), "non-empty leaf where template expects array");
            count_silent(tmpl, out);
        }
    } else {
        // Template position is a leaf.
        let s = match tok {
            Value::String(s) => s.clone(),
            _ => panic!("token has array where template expects leaf"),
        };
        out.push(s);
    }
}

fn count_silent(tmpl: &Value, out: &mut Vec<String>) {
    if let Value::Array(arr) = tmpl {
        for c in arr { count_silent(c, out); }
    } else {
        out.push(String::new());
    }
}

/// Forward log-likelihood over `m` for the given concrete `params` and
/// per-leaf input symbol sequences.
///
/// Returns `f[end_state, all-leaves-consumed]`. Bit-exact mirror of
/// `t/rust/_phylo_ref.py::multidim_forward` — same DP recurrences, same
/// exact log_sum_exp, same iteration order. Panics if the machine has
/// no emitting transitions or if `leaves.len()` differs from L (the
/// number of leaf positions in the merged pair-token template).
pub fn forward(m: &Machine, params: &Params, leaves: &[Vec<String>]) -> f64 {
    let n = m.n_states();
    if n == 0 { return f64::NEG_INFINITY; }

    // 1) Identify the canonical pair-token template by merging shapes
    //    across all emit transitions (mirrors `merge_shape` in Python).
    let mut tmpl: Option<Value> = None;
    for s in &m.state {
        for t in &s.trans {
            if t.out_sym.is_empty() { continue; }
            let tt = parse_token(&t.out_sym);
            tmpl = Some(match tmpl.take() {
                None => tt,
                Some(prev) => merge_shape(&prev, &tt),
            });
        }
    }
    let tmpl = tmpl.expect("forward: machine has no emitting transitions");
    let l = count_leaves(&tmpl);
    assert_eq!(l, leaves.len(),
               "forward: template has {} leaves but got {} leaf sequences",
               l, leaves.len());

    // 2) Build silent + emitting transition lists with log-weights.
    let mut silent: Vec<(usize, usize, f64)> = Vec::new();
    let mut emitting: Vec<(usize, usize, Vec<u8>, Vec<String>, f64)> = Vec::new();
    for (s_idx, s) in m.state.iter().enumerate() {
        for t in &s.trans {
            let w = evaluate(&t.weight, params, &m.defs);
            if !(w > 0.0) { continue; }  // skip 0 / NaN / negative
            let lw = w.ln();
            let d = t.to;
            if t.out_sym.is_empty() {
                silent.push((s_idx, d, lw));
            } else {
                let mut profile: Vec<String> = Vec::new();
                decode(&parse_token(&t.out_sym), &tmpl, &mut profile);
                let deltas: Vec<u8> = profile.iter()
                    .map(|x| if x.is_empty() { 0 } else { 1 })
                    .collect();
                emitting.push((s_idx, d, deltas, profile, lw));
            }
        }
    }
    silent.sort_by_key(|e| e.1);

    // 3) Allocate flat F[N * total].
    let lens: Vec<usize> = leaves.iter().map(|x| x.len()).collect();
    let total: usize = lens.iter().map(|&x| x + 1).product();
    let mut strides: Vec<usize> = vec![1; l];
    if l >= 2 {
        for k in (0..l-1).rev() {
            strides[k] = strides[k+1] * (lens[k+1] + 1);
        }
    }
    let mut f = vec![f64::NEG_INFINITY; n * total];
    f[0] = 0.0;

    // Apply silent transitions at the all-zero cell first.
    for &(s_, d_, lw) in &silent {
        let sv = f[s_ * total + 0];
        if sv.is_finite() {
            let off = d_ * total + 0;
            f[off] = lse(f[off], sv + lw);
        }
    }

    // 4) Iterate over (idx[0], ..., idx[L-1]) cells in row-major order.
    let mut idx: Vec<usize> = vec![0; l];
    loop {
        // Advance idx, rightmost dim varies fastest (mirrors Python).
        let mut k = l;
        let mut advanced = false;
        while k > 0 {
            k -= 1;
            if idx[k] < lens[k] {
                idx[k] += 1;
                for j in (k+1)..l { idx[j] = 0; }
                advanced = true;
                break;
            }
        }
        if !advanced { break; }

        let cell: usize = idx.iter().enumerate()
            .map(|(k, &i)| i * strides[k]).sum();

        for (s_, d_, deltas, syms, lw) in &emitting {
            let mut ok = true;
            let mut prev = cell;
            for k in 0..l {
                if deltas[k] != 0 {
                    if idx[k] == 0 { ok = false; break; }
                    if leaves[k][idx[k] - 1] != syms[k] { ok = false; break; }
                    prev -= strides[k];
                }
            }
            if ok {
                let sv = f[s_ * total + prev];
                if sv.is_finite() {
                    let off = d_ * total + cell;
                    f[off] = lse(f[off], sv + *lw);
                }
            }
        }
        for &(s_, d_, lw) in &silent {
            let sv = f[s_ * total + cell];
            if sv.is_finite() {
                let off = d_ * total + cell;
                f[off] = lse(f[off], sv + lw);
            }
        }
    }

    f[(n - 1) * total + (total - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn lse_basics() {
        assert_eq!(lse(f64::NEG_INFINITY, 0.0), 0.0);
        assert_eq!(lse(0.0, f64::NEG_INFINITY), 0.0);
        // log(2) = lse(0, 0)
        assert!((lse(0.0, 0.0) - std::f64::consts::LN_2).abs() < 1e-15);
    }

    #[test]
    fn parse_token_handles_pair_token() {
        let t = parse_token("[\"A\",\"B\"]");
        assert_eq!(count_leaves(&t), 2);
        let t = parse_token("[[\"A\",\"B\"],\"C\"]");
        assert_eq!(count_leaves(&t), 3);
    }

    #[test]
    fn merge_shape_resolves_silent_position() {
        let a = parse_token("[\"\",\"B\"]");
        let b = parse_token("[\"A\",\"\"]");
        let m = merge_shape(&a, &b);
        assert_eq!(count_leaves(&m), 2);
    }

    #[test]
    fn forward_single_emit_machine() {
        // 1-leaf machine: S → E via single emit "x". Weight 1.0 on the emit.
        // F[end, idx=1] = log(1) = 0.0 when leaf == ["x"].
        // Note: there must exist at least one silent transition for the
        // standard convention; we add a leading silent S → mid → emit → E.
        use crate::machine::Machine;
        let m = Machine::from_json(&json!({
            "state": [
                {"id": "S", "trans": [{"to": 1}]},
                {"id": "M", "trans": [{"to": 2, "in": "x", "out": "x", "weight": 1.0}]},
                {"id": "E"}
            ]
        }));
        let params = Params::new();
        let leaves: Vec<Vec<String>> = vec![vec!["x".to_string()]];
        let v = forward(&m, &params, &leaves);
        assert!((v - 0.0).abs() < 1e-15, "expected 0.0, got {}", v);
    }

    #[test]
    fn forward_no_match_returns_neg_inf() {
        use crate::machine::Machine;
        let m = Machine::from_json(&json!({
            "state": [
                {"id": "S", "trans": [{"to": 1}]},
                {"id": "M", "trans": [{"to": 2, "in": "x", "out": "x", "weight": 1.0}]},
                {"id": "E"}
            ]
        }));
        let params = Params::new();
        // Wrong leaf symbol → no path consumes it.
        let leaves: Vec<Vec<String>> = vec![vec!["y".to_string()]];
        let v = forward(&m, &params, &leaves);
        assert_eq!(v, f64::NEG_INFINITY);
    }
}
)RUST";
  }

  // src/lib.rs — bake the JSON inputs as `&'static str` consts, expose
  // the weight_algebra / machine / phylo / forward modules, plus the
  // `prebuild()` entry point.
  {
    std::ofstream f (outputDir + "/src/lib.rs");
    f << "// Auto-generated by Machine Boss --phylo-skeleton --codegen --rust.\n"
         "// Do not edit.\n"
         "\n"
         "#![allow(dead_code)]\n"
         "\n"
         "pub mod weight_algebra;\n"
         "pub mod machine;\n"
         "pub mod phylo;\n"
         "pub mod forward;\n"
         "\n"
         "/// Canonical branch transducer T (no per-branch time-parameter\n"
         "/// renaming). Format is the standard Machine Boss machine JSON.\n"
         "pub static T_JSON: &str = " << rustRawStringLit (tStream.str()) << ";\n"
         "\n"
         "/// Unary phylo skeleton (output of --phylo-skeleton on the tree\n"
         "/// below applied to T). Has the same accessible state set and same\n"
         "/// (src, dst) class enumeration as the full phylo machine; emit\n"
         "/// transitions are placeholder, silent transitions carry their full\n"
         "/// (chain-collapsed) structural weights from ergodicMachine.\n"
         "pub static M_SKEL_JSON: &str = " << rustRawStringLit (skelStream.str()) << ";\n"
         "\n"
         "/// Newick tree string. Branch names appear as parameter suffixes\n"
         "/// in M_SKEL's WeightAlgebra expressions (e.g. `time[A]`).\n"
         "pub static TREE_NEWICK: &str = " << rustRawStringLit (tree_newick) << ";\n"
         "\n"
         "/// Name of T's per-branch time parameter. The phylo recursion\n"
         "/// renames this to `<TIME_PARAM>[<branch_name>]` per non-root node.\n"
         "pub static TIME_PARAM: &str = " << rustRawStringLit (time_param) << ";\n"
         "\n"
         "/// Build the phylo-composed `Machine` from the baked T + tree by\n"
         "/// running the Rust port of the WFST algebra (compose / intersect /\n"
         "/// waitingMachine / ergodicMachine + buildSubtree recursion). This\n"
         "/// is the same machine the legacy `boss --phylo-tree*` produces,\n"
         "/// modulo `advance_sort` + `process_cycles` post-processing (TODO:\n"
         "/// only matters for inputs that have silent cycles in intermediate\n"
         "/// machines — TKF inputs typically do not).\n"
         "///\n"
         "/// Returns a `machine::Machine` whose transition weights are still\n"
         "/// symbolic `WeightAlgebra` JSON expressions. Evaluate them with\n"
         "/// `weight_algebra::evaluate(&t.weight, &params, &m.defs)` to get\n"
         "/// numeric weights. A subsequent increment will glue this to the\n"
         "/// existing Forward / Viterbi DP tables (`EMIT_SHARDS`,\n"
         "/// `SILENT_TRANSITIONS`, `lw[]`).\n"
         "pub fn prebuild() -> machine::Machine {\n"
         "    let t_json: serde_json::Value =\n"
         "        serde_json::from_str(T_JSON).expect(\"T_JSON parses\");\n"
         "    let t = machine::Machine::from_json(&t_json);\n"
         "    let tree = phylo::PhyloTree::parse_newick(TREE_NEWICK);\n"
         "    phylo::phylo_intersect(&t, &tree, TIME_PARAM)\n"
         "}\n";
  }

  LogThisAt (2, "Wrote skeleton-bake Rust crate to " << outputDir
             << " (M_skel states=" << M_skel.nStates()
             << ", T states=" << T.nStates()
             << ", tree=" << tree_newick.size() << " bytes)" << std::endl);
}

// ---------------------------------------------------------------------------
// compileRustTransducer: regular in/out transducer Forward / Viterbi.
//
// Bakes the machine JSON as a `&'static str`, parses it lazily on first
// call (or on every call — there's no static memoisation in this minimal
// version), pre-evaluates per-call weights once via the WeightAlgebra
// evaluator, and runs a 2D DP over (input_index, output_index) with a
// state dimension. This is the in/out counterpart to compileRust (the
// phylo-multidim path) — separate function, separate CLI flag, no
// touching of the existing API.

void compileRustTransducer (const Machine& m, const std::string& outputDir, bool emitViterbi) {
  if (m.inputAlphabet().empty() && m.outputAlphabet().empty())
    throw std::runtime_error ("rust-codegen-transducer: machine has no input AND no output alphabet (no consuming or emitting transitions)");

  mkdirP (outputDir);  // creates outputDir/ and outputDir/src/

  // Defensively advance-sort so silent transitions go forward.
  Machine sorted = m.advanceSort();

  std::ostringstream mjson;
  sorted.writeJson (mjson);

  // ---------- Cargo.toml ----------
  {
    std::ofstream f (outputDir + "/Cargo.toml");
    f << "[package]\n"
         "name = \"transducer_dp\"\n"
         "version = \"0.1.0\"\n"
         "edition = \"2021\"\n"
         "\n"
         "[lib]\n"
         "name = \"transducer_dp\"\n"
         "path = \"src/lib.rs\"\n"
         "\n"
         "[dependencies]\n"
         "serde_json = \"1\"\n"
         "\n"
         "[profile.release]\n"
         "opt-level = 3\n"
         "lto = true\n";
  }

  // ---------- src/weight_algebra.rs ----------
  // Reuse the same WeightExpr evaluator the skeleton-bake mode generates.
  // Inlined here to keep this crate fully self-contained (no path deps).
  {
    std::ofstream f (outputDir + "/src/weight_algebra.rs");
    f << R"RUST(//! Weight algebra evaluator (subset). Mirrors `src/weight.cpp`'s
//! `WeightAlgebra::fromJson` evaluator: parses the JSON weight-expression
//! form used in machine.json and evaluates it against a Params map.
use std::collections::HashMap;
use serde_json::Value;

pub type Params = HashMap<String, f64>;
pub type Defs   = HashMap<String, Value>;

pub fn parse_defs(machine_json: &Value) -> Defs {
    let mut defs = Defs::new();
    if let Some(d) = machine_json.get("defs").and_then(|d| d.as_object()) {
        for (k, v) in d.iter() { defs.insert(k.clone(), v.clone()); }
    }
    defs
}

pub fn evaluate(expr: &Value, params: &Params, defs: &Defs) -> f64 {
    let mut visiting: Vec<String> = Vec::new();
    eval_inner(expr, params, defs, &mut visiting)
}

fn eval_inner(expr: &Value, params: &Params, defs: &Defs, visiting: &mut Vec<String>) -> f64 {
    match expr {
        Value::Number(n) => n.as_f64().expect("WeightExpr number not f64"),
        Value::Bool(b)   => if *b { 1.0 } else { 0.0 },
        Value::Null      => 0.0,
        Value::String(s) => {
            if visiting.iter().any(|v| v == s) {
                panic!("WeightExpr cyclic def reference: {}", s);
            }
            if let Some(d) = defs.get(s) {
                visiting.push(s.clone());
                let v = eval_inner(d, params, defs, visiting);
                visiting.pop();
                v
            } else if let Some(p) = params.get(s) {
                *p
            } else {
                panic!("WeightExpr unknown name: {}", s);
            }
        }
        Value::Object(map) => {
            let (op, arg) = map.iter().next().expect("WeightExpr object empty");
            match op.as_str() {
                "log"     => eval_inner(arg, params, defs, visiting).ln(),
                "exp"     => eval_inner(arg, params, defs, visiting).exp(),
                "not"     => 1.0 - eval_inner(arg, params, defs, visiting),
                "geomsum" => 1.0 / (1.0 - eval_inner(arg, params, defs, visiting)),
                "*" | "/" | "+" | "-" | "pow" => {
                    let arr = arg.as_array().expect("binary-op arg not array");
                    assert_eq!(arr.len(), 2);
                    let l = eval_inner(&arr[0], params, defs, visiting);
                    let r = eval_inner(&arr[1], params, defs, visiting);
                    match op.as_str() {
                        "*" => l * r, "/" => l / r,
                        "+" => l + r, "-" => l - r,
                        "pow" => l.powf(r),
                        _ => unreachable!(),
                    }
                }
                _ => panic!("WeightExpr unknown opcode: {}", op),
            }
        }
        _ => panic!("WeightExpr unsupported JSON type"),
    }
}
)RUST";
  }

  // ---------- src/lib.rs ----------
  // Bakes the machine JSON, exposes forward / viterbi taking &[&str] for
  // both input and output. Per call: parse JSON → evaluate weights →
  // bucket transitions into {silent, match, insert, delete} → run the 2D
  // DP. JSON parse + evaluate dominate runtime for short sequences; the
  // DP itself is straight f64 arithmetic.
  {
    std::ofstream f (outputDir + "/src/lib.rs");
    f << "// Auto-generated by Machine Boss --codegen --rust-transducer.\n"
         "// Do not edit.\n"
         "\n"
         "#![allow(dead_code)]\n"
         "\n"
         "pub mod weight_algebra;\n"
         "\n"
         "pub use weight_algebra::Params;\n"
         "use serde_json::Value;\n"
         "\n"
         "/// The machine, baked as canonical Machine Boss JSON. Parsed on each\n"
         "/// `forward` / `viterbi` call.\n"
         "pub static MACHINE_JSON: &str = " << rustRawStringLit (mjson.str()) << ";\n"
         "\n"
         "#[inline]\n"
         "fn lse(a: f64, b: f64) -> f64 {\n"
         "    if a == f64::NEG_INFINITY { return b; }\n"
         "    if b == f64::NEG_INFINITY { return a; }\n"
         "    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };\n"
         "    hi + (lo - hi).exp().ln_1p()\n"
         "}\n"
         "\n"
         "#[derive(Clone)]\n"
         "struct Trans {\n"
         "    src: usize, dst: usize,\n"
         "    in_sym: String, out_sym: String,\n"
         "    lw: f64,\n"
         "}\n"
         "\n"
         "fn resolve_dst(to: &Value, name_to_idx: &std::collections::HashMap<String, usize>) -> usize {\n"
         "    if let Some(n) = to.as_u64() { return n as usize; }\n"
         "    if let Some(s) = to.as_str() {\n"
         "        if let Some(i) = name_to_idx.get(s) { return *i; }\n"
         "    }\n"
         "    let key = to.to_string();\n"
         "    *name_to_idx.get(&key).unwrap_or_else(|| panic!(\"cannot resolve `to` reference: {}\", key))\n"
         "}\n"
         "\n"
         "/// Per-call setup: parse JSON, pre-evaluate every transition's weight\n"
         "/// against `params`, bucket into silent / match / insert / delete.\n"
         "/// Returns (n_states, silent_sorted_by_dst, match_trans, insert_trans, delete_trans).\n"
         "fn prepare(params: &Params)\n"
         "  -> (usize, Vec<Trans>, Vec<Trans>, Vec<Trans>, Vec<Trans>)\n"
         "{\n"
         "    let machine: Value = serde_json::from_str(MACHINE_JSON).expect(\"MACHINE_JSON parse\");\n"
         "    let defs = weight_algebra::parse_defs(&machine);\n"
         "    let states = machine[\"state\"].as_array().expect(\"machine.state not an array\");\n"
         "    let n = states.len();\n"
         "\n"
         "    // Build name→index map for transitions whose `to` is a state name.\n"
         "    let mut name_to_idx: std::collections::HashMap<String, usize> = std::collections::HashMap::new();\n"
         "    for (i, s) in states.iter().enumerate() {\n"
         "        if let Some(id) = s.get(\"id\") {\n"
         "            name_to_idx.insert(id.to_string(), i);\n"
         "            if let Some(s_str) = id.as_str() { name_to_idx.insert(s_str.to_string(), i); }\n"
         "        }\n"
         "        name_to_idx.insert(format!(\"{}\", i), i);\n"
         "    }\n"
         "\n"
         "    let mut silent: Vec<Trans> = Vec::new();\n"
         "    let mut match_t: Vec<Trans> = Vec::new();\n"
         "    let mut insert_t: Vec<Trans> = Vec::new();\n"
         "    let mut delete_t: Vec<Trans> = Vec::new();\n"
         "    for (s_idx, s) in states.iter().enumerate() {\n"
         "        let trans_arr = s.get(\"trans\").and_then(|x| x.as_array());\n"
         "        let empty: Vec<Value> = Vec::new();\n"
         "        let arr = trans_arr.unwrap_or(&empty);\n"
         "        for t in arr.iter() {\n"
         "            let dst = resolve_dst(t.get(\"to\").expect(\"trans missing `to`\"), &name_to_idx);\n"
         "            let in_sym  = t.get(\"in\").and_then(|x| x.as_str()).unwrap_or(\"\").to_string();\n"
         "            let out_sym = t.get(\"out\").and_then(|x| x.as_str()).unwrap_or(\"\").to_string();\n"
         "            let weight  = t.get(\"weight\").cloned().unwrap_or(Value::from(1.0_f64));\n"
         "            let w = weight_algebra::evaluate(&weight, params, &defs);\n"
         "            if !(w > 0.0) { continue; }   // skip 0 / NaN / negative\n"
         "            let lw = w.ln();\n"
         "            let tr = Trans { src: s_idx, dst, in_sym: in_sym.clone(), out_sym: out_sym.clone(), lw };\n"
         "            match (in_sym.is_empty(), out_sym.is_empty()) {\n"
         "                (true,  true ) => silent.push(tr),\n"
         "                (false, false) => match_t.push(tr),\n"
         "                (true,  false) => insert_t.push(tr),\n"
         "                (false, true ) => delete_t.push(tr),\n"
         "            }\n"
         "        }\n"
         "    }\n"
         "    silent.sort_by_key(|t| t.dst);\n"
         "    (n, silent, match_t, insert_t, delete_t)\n"
         "}\n"
         "\n"
         "// 2D DP shared by forward / viterbi. `combine` is `lse` for forward,\n"
         "// `f64::max` for viterbi. Returns f[end_state, m_in, m_out].\n"
         "fn run_dp<F>(input: &[&str], output: &[&str], params: &Params, combine: F) -> f64\n"
         "where F: Fn(f64, f64) -> f64\n"
         "{\n"
         "    let (n, silent, match_t, insert_t, delete_t) = prepare(params);\n"
         "    if n == 0 { return f64::NEG_INFINITY; }\n"
         "    let m_in = input.len();\n"
         "    let m_out = output.len();\n"
         "    let row = m_out + 1;\n"
         "    let plane = (m_in + 1) * row;\n"
         "    let cell = |i: usize, j: usize, s: usize| s * plane + i * row + j;\n"
         "    let mut f = vec![f64::NEG_INFINITY; n * plane];\n"
         "    f[cell(0, 0, 0)] = 0.0;\n"
         "    // Apply silent transitions at the all-zero cell.\n"
         "    for t in &silent {\n"
         "        let v = f[cell(0, 0, t.src)];\n"
         "        if v.is_finite() {\n"
         "            let off = cell(0, 0, t.dst);\n"
         "            f[off] = combine(f[off], v + t.lw);\n"
         "        }\n"
         "    }\n"
         "    for i in 0..=m_in {\n"
         "        for j in 0..=m_out {\n"
         "            if i == 0 && j == 0 { continue; }\n"
         "            // Match transitions: consume input[i-1] AND emit output[j-1].\n"
         "            if i > 0 && j > 0 {\n"
         "                for t in &match_t {\n"
         "                    if input[i-1] == t.in_sym && output[j-1] == t.out_sym {\n"
         "                        let v = f[cell(i-1, j-1, t.src)];\n"
         "                        if v.is_finite() {\n"
         "                            let off = cell(i, j, t.dst);\n"
         "                            f[off] = combine(f[off], v + t.lw);\n"
         "                        }\n"
         "                    }\n"
         "                }\n"
         "            }\n"
         "            // Insert: emit output[j-1] without consuming input.\n"
         "            if j > 0 {\n"
         "                for t in &insert_t {\n"
         "                    if output[j-1] == t.out_sym {\n"
         "                        let v = f[cell(i, j-1, t.src)];\n"
         "                        if v.is_finite() {\n"
         "                            let off = cell(i, j, t.dst);\n"
         "                            f[off] = combine(f[off], v + t.lw);\n"
         "                        }\n"
         "                    }\n"
         "                }\n"
         "            }\n"
         "            // Delete: consume input[i-1] without emitting output.\n"
         "            if i > 0 {\n"
         "                for t in &delete_t {\n"
         "                    if input[i-1] == t.in_sym {\n"
         "                        let v = f[cell(i-1, j, t.src)];\n"
         "                        if v.is_finite() {\n"
         "                            let off = cell(i, j, t.dst);\n"
         "                            f[off] = combine(f[off], v + t.lw);\n"
         "                        }\n"
         "                    }\n"
         "                }\n"
         "            }\n"
         "            // Silent transitions at (i, j).\n"
         "            for t in &silent {\n"
         "                let v = f[cell(i, j, t.src)];\n"
         "                if v.is_finite() {\n"
         "                    let off = cell(i, j, t.dst);\n"
         "                    f[off] = combine(f[off], v + t.lw);\n"
         "                }\n"
         "            }\n"
         "        }\n"
         "    }\n"
         "    f[cell(m_in, m_out, n - 1)]\n"
         "}\n"
         "\n"
         "/// Forward log-likelihood for the given input / output strings under\n"
         "/// `params`. Returns `f64::NEG_INFINITY` for impossible alignments.\n"
         "pub fn forward(params: &Params, input: &[&str], output: &[&str]) -> f64 {\n"
         "    run_dp(input, output, params, lse)\n"
         "}\n";
    if (emitViterbi)
      f << "\n"
           "/// Viterbi log-score for the given input / output strings under `params`.\n"
           "pub fn viterbi(params: &Params, input: &[&str], output: &[&str]) -> f64 {\n"
           "    run_dp(input, output, params, f64::max)\n"
           "}\n";
  }

  LogThisAt (2, "Wrote regular-transducer Rust crate to " << outputDir
             << " (states=" << sorted.nStates()
             << ", input |Σ|=" << m.inputAlphabet().size()
             << ", output |Σ|=" << m.outputAlphabet().size()
             << ")" << std::endl);
}

}  // namespace MachineBoss
