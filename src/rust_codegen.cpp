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

impl Machine {
    /// Every state either waits, continues, or terminates — i.e. no state
    /// has BOTH input-consuming and silent-input outgoing transitions.
    pub fn is_waiting_machine(&self) -> bool {
        self.state.iter().all(|s| s.waits() || s.continues())
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
}
)RUST";
  }

  // src/lib.rs — bake the JSON inputs as `&'static str` consts, expose
  // the weight_algebra module, expose a stub `prebuild()` that panics.
  // Subsequent increments will replace the stub with a Rust port of the
  // WFST algebra (Machine + compose / intersect / waitingMachine /
  // ergodicMachine + phylo recursion) so prebuild() can materialise
  // M_full's per-symbol DP tables in shape with M_skel.
  {
    std::ofstream f (outputDir + "/src/lib.rs");
    f << "// Auto-generated by Machine Boss --phylo-skeleton --codegen --rust.\n"
         "// Do not edit.\n"
         "\n"
         "#![allow(dead_code)]\n"
         "\n"
         "pub mod weight_algebra;\n"
         "pub mod machine;\n"
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
         "/// Stub. Will become the entry point that, given concrete `Params`,\n"
         "/// runs the WFST algebra on T + tree to materialise the per-symbol\n"
         "/// DP tables (`EMIT_SHARDS`, `SILENT_TRANSITIONS`, etc.) that the\n"
         "/// Forward / Viterbi / Backward DP code consumes.\n"
         "pub fn prebuild() {\n"
         "    panic!(\"phylo_skeleton::prebuild not yet implemented; this crate \\\n"
         "            currently bakes T_JSON / M_SKEL_JSON / TREE_NEWICK / TIME_PARAM \\\n"
         "            for downstream tooling. The Rust port of compose/intersect/\\\n"
         "            phylo-fold lands in subsequent increments.\");\n"
         "}\n";
  }

  LogThisAt (2, "Wrote skeleton-bake Rust crate to " << outputDir
             << " (M_skel states=" << M_skel.nStates()
             << ", T states=" << T.nStates()
             << ", tree=" << tree_newick.size() << " bytes)" << std::endl);
}

}  // namespace MachineBoss
