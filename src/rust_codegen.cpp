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
  std::map<BucketKey, WeightExpr> buckets;
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
      if (it == buckets.end()) buckets[k] = t.weight;
      else it->second = WeightAlgebra::add (it->second, t.weight);
    }
  }

  // Split buckets into silent (all deltas == 0) and emitting.
  struct Entry { StateIndex src, dst; std::vector<uint8_t> deltas; std::vector<int> symIdx; size_t weightIndex; };
  std::vector<Entry> silent, emitting;
  std::vector<WeightExpr> weightsInOrder;
  for (auto& kv : buckets) {
    Entry e;
    e.src = kv.first.src; e.dst = kv.first.dst;
    e.deltas = kv.first.deltas; e.symIdx = kv.first.symIdx;
    e.weightIndex = weightsInOrder.size();
    weightsInOrder.push_back (kv.second);
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
  f << "#![allow(non_snake_case, unused_parens, unused_variables, dead_code, clippy::needless_range_loop)]\n\n";

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

  // Emit constant tables for silent and emitting transitions.
  // Layout: silent -> [(src, dst, weight_idx)]; emitting -> [(src, dst, deltas[L], syms[L], weight_idx)]
  f << "const SILENT_TRANSITIONS: &[(u32, u32, u32)] = &[\n";
  for (const auto& e : silent)
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

  f << "const NUM_DELTA: usize = " << deltaVecs.size() << ";\n";
  f << "const DELTA_VEC: [[u8; " << L << "]; NUM_DELTA] = [";
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

  // ----- DP runner template (parameterised over reduce op) -----
  // We emit two functions, forward (reduce = log_sum_exp) and viterbi
  // (reduce = max), sharing the same DP body via a generic helper.
  f << R"RUST(
// log_sum_exp(a, b) = log(exp(a) + exp(b)). The early return when the gap
// between hi and lo exceeds 36 nats is exact in f64: exp(-36) ≈ 2.3e-16 is
// below f64's relative epsilon (2.2e-16), so log1p(exp(lo - hi)) rounds to
// 0.0 and `hi + 0.0 == hi` regardless of hi's magnitude.
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

)RUST";

  // Generate the DP function body, parametrised by reducer.
  // The body iterates cells in lex order, processes emitting transitions
  // (each consumes a single cell-position offset), then processes silent
  // transitions in dst-state order.
  // The DP body assumes `lw: &[f64]` is in scope (the wrapper functions
  // provide it: either by calling precompute_log_weights, or by accepting
  // a precomputed slice).
  auto emitDpBody = [&](std::ofstream& f, const char* reduce) {
    f << "    let lens: [usize; NUM_LEAVES] = [";
    for (size_t i = 0; i < L; ++i) f << (i ? ", " : "") << "leaves[" << i << "].len()";
    f << "];\n";
    f << "    let total: usize = ";
    for (size_t i = 0; i < L; ++i) f << (i ? " * " : "") << "(lens[" << i << "] + 1)";
    f << ";\n";
    // strides[k] = product of (lens[k+1]+1)...(lens[L-1]+1); strides[L-1] = 1
    f << "    let mut strides: [usize; NUM_LEAVES] = [1; NUM_LEAVES];\n";
    f << "    for k in (0..NUM_LEAVES-1).rev() { strides[k] = strides[k+1] * (lens[k+1] + 1); }\n";
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
    // Iterate cells in lex order (skipping origin). We use L nested loops.
    f << "    let mut idx: [usize; NUM_LEAVES] = [0; NUM_LEAVES];\n";
    f << "    loop {\n";
    f << "        // advance idx in lex order\n";
    f << "        let mut k = NUM_LEAVES;\n";
    f << "        loop {\n";
    f << "            if k == 0 { return g[(END_STATE as usize) * total + total - 1]; }\n";
    f << "            k -= 1;\n";
    f << "            if idx[k] < lens[k] { idx[k] += 1; for j in (k+1)..NUM_LEAVES { idx[j] = 0; } break; }\n";
    f << "        }\n";
    f << "        let cell: usize = ";
    for (size_t i = 0; i < L; ++i) f << (i ? " + " : "") << "idx[" << i << "] * strides[" << i << "]";
    f << ";\n";
    f << "        // Emitting transitions into this cell, sharded by delta vector.\n";
    f << "        for d_idx in 0..NUM_DELTA {\n";
    f << "            let dvec = unsafe { *DELTA_VEC.get_unchecked(d_idx) };\n";
    // Compute per-shard prev cell (subtract strides[k] for each emitting
    // position) and feasibility (idx[k] >= 1 for each emitting position).
    f << "            let mut prev = cell;\n";
    f << "            let mut feasible = true;\n";
    f << "            for k in 0..NUM_LEAVES {\n";
    f << "                if dvec[k] == 1 {\n";
    f << "                    if idx[k] == 0 { feasible = false; break; }\n";
    f << "                    prev -= strides[k];\n";
    f << "                }\n";
    f << "            }\n";
    f << "            if !feasible { continue; }\n";
    // Cache the observed leaf symbols at the predecessor positions for this
    // shard. Since dvec is a per-shard constant, the compiler can specialize
    // these reads once and lift them out of the inner transition loop.
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
    f << "        // Silent closure (in topological dst-state order).\n";
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

  // Body-taking-precomputed-lw variants (the meat).
  f << "/// Forward log-likelihood with precomputed log-weights.\n";
  f << "/// `lw` must be the result of `precompute_log_weights(p)` for the\n";
  f << "/// `Params` you want to evaluate at; pass it instead of `Params`\n";
  f << "/// when amortizing the prelude across many calls.\n";
  f << "pub fn forward_with_log_weights(lw: &[f64], leaves: [&[u32]; NUM_LEAVES]) -> f64 {\n";
  emitDpBody (f, "lse");
  f << "}\n\n";

  if (emitViterbi) {
    f << "/// Viterbi log-likelihood with precomputed log-weights.\n";
    f << "pub fn viterbi_with_log_weights(lw: &[f64], leaves: [&[u32]; NUM_LEAVES]) -> f64 {\n";
    emitDpBody (f, "max2");
    f << "}\n\n";
  }

  // Convenience wrappers: compute weights and forward in a single call.
  f << "/// Forward log-likelihood (computes log-weights internally).\n";
  f << "#[inline]\n";
  f << "pub fn forward(p: &Params, leaves: [&[u32]; NUM_LEAVES]) -> f64 {\n";
  f << "    forward_with_log_weights(&precompute_log_weights(p), leaves)\n";
  f << "}\n\n";

  if (emitViterbi) {
    f << "/// Viterbi log-likelihood (computes log-weights internally).\n";
    f << "#[inline]\n";
    f << "pub fn viterbi(p: &Params, leaves: [&[u32]; NUM_LEAVES]) -> f64 {\n";
    f << "    viterbi_with_log_weights(&precompute_log_weights(p), leaves)\n";
    f << "}\n";
  }

  LogThisAt(2,"Wrote Rust crate to " << outputDir
            << " (states=" << N << ", leaves=" << L
            << ", alphabet=" << alph.size()
            << ", silent=" << silent.size() << ", emitting=" << emitting.size()
            << ", weights=" << weightsInOrder.size() << ")" << std::endl);
}

}  // namespace MachineBoss
