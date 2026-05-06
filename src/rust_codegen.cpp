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

// Parse a token string under JSON-mode encoding into a PTok.
PTok parseTokenJson (const std::string& s) {
  if (s.empty()) {
    PTok t; t.isLeaf = true; t.leaf = ""; return t;
  }
  return fromJson (json::parse (s));
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

class RustExpr {
public:
  // funcVar: maps def-name -> Rust local variable name (e.g. "pSame" -> "def_pSame")
  // paramVar: maps free-param-name -> Rust expression (e.g. "time[A]" -> "p.time__A")
  std::map<std::string,std::string> funcVar;
  std::map<std::string,std::string> paramVar;

  // Wrap with parentheses if needed based on operator precedence.
  // Precedence: 0 = full expr, 1 = +/-, 2 = */, 3 = unary, 4 = atom
  std::string emit (WeightExpr w, int parentPrec = 0) const {
    if (!w) return "1.0";
    std::ostringstream o;
    auto wrap = [&](int myPrec, const std::string& s) -> std::string {
      if (myPrec < parentPrec) return "(" + s + ")";
      return s;
    };
    switch (w->type) {
      case Null: return "1.0";
      case Int: {
        o << w->args.intValue << ".0_f64";
        return o.str();
      }
      case Dbl: {
        o << std::setprecision(17) << w->args.doubleValue << "_f64";
        return o.str();
      }
      case Param: {
        const std::string& name = *w->args.param;
        auto it = funcVar.find (name);
        if (it != funcVar.end()) return it->second;
        auto it2 = paramVar.find (name);
        if (it2 != paramVar.end()) return it2->second;
        throw std::runtime_error ("rust-codegen: unbound parameter \"" + name + "\"");
      }
      case Mul: {
        std::string s = emit (w->args.binary.l, 2) + " * " + emit (w->args.binary.r, 2);
        return wrap (2, s);
      }
      case Div: {
        std::string s = emit (w->args.binary.l, 2) + " / " + emit (w->args.binary.r, 3);
        return wrap (2, s);
      }
      case Add: {
        std::string s = emit (w->args.binary.l, 1) + " + " + emit (w->args.binary.r, 1);
        return wrap (1, s);
      }
      case Sub: {
        std::string s = emit (w->args.binary.l, 1) + " - " + emit (w->args.binary.r, 2);
        return wrap (1, s);
      }
      case Pow: {
        std::string s = emit (w->args.binary.l, 4) + ".powf(" + emit (w->args.binary.r, 0) + ")";
        return s;
      }
      case Log: {
        return emit (w->args.arg, 4) + ".ln()";
      }
      case Exp: {
        return emit (w->args.arg, 4) + ".exp()";
      }
    }
    throw std::runtime_error ("rust-codegen: unknown WeightExpr type");
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

  // Free parameters: all params referenced anywhere, minus those defined in funcs.defs.
  std::set<std::string> defNames;
  for (const auto& d : sorted.funcs.defs) defNames.insert (d.first);

  std::set<std::string> referenced;
  std::function<void(WeightExpr)> collectParams = [&](WeightExpr w) {
    if (!w) return;
    switch (w->type) {
      case Param: referenced.insert (*w->args.param); break;
      case Mul: case Div: case Add: case Sub: case Pow:
        collectParams (w->args.binary.l);
        collectParams (w->args.binary.r);
        break;
      case Log: case Exp: collectParams (w->args.arg); break;
      default: break;
    }
  };
  for (const auto& d : sorted.funcs.defs) collectParams (d.second);
  for (StateIndex s = 0; s < (StateIndex)N; ++s)
    for (const auto& t : states[s].trans)
      collectParams (t.weight);
  std::set<std::string> freeParams;
  for (const auto& p : referenced) if (!defNames.count(p)) freeParams.insert (p);

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

  // compute_log_weights: emit defs in topological order, then evaluate each
  // bucket's weight expression and take ln().
  {
    RustExpr re;
    for (const auto& p : freeParams)
      re.paramVar[p] = "p." + paramRustName[p];
    auto topoOrder = topoSortDefs (sorted.funcs.defs);
    for (const auto& n : topoOrder)
      re.funcVar[n] = "def_" + sanitize(n);

    f << "fn compute_log_weights(p: &Params) -> [f64; " << weightsInOrder.size() << "] {\n";
    for (const auto& n : topoOrder) {
      WeightExpr e = sorted.funcs.defs.at(n);
      f << "    let " << re.funcVar[n] << ": f64 = " << re.emit (e, 0) << ";\n";
    }
    f << "    [\n";
    for (size_t i = 0; i < weightsInOrder.size(); ++i) {
      f << "        ({ " << re.emit (weightsInOrder[i], 0) << " }).ln(),\n";
    }
    f << "    ]\n}\n\n";
  }

  // Emit constant tables for silent and emitting transitions.
  // Layout: silent -> [(src, dst, weight_idx)]; emitting -> [(src, dst, deltas[L], syms[L], weight_idx)]
  f << "const SILENT_TRANSITIONS: &[(u32, u32, u32)] = &[\n";
  for (const auto& e : silent)
    f << "    (" << e.src << ", " << e.dst << ", " << e.weightIndex << "),\n";
  f << "];\n\n";

  f << "const EMITTING_TRANSITIONS: &[(u32, u32, [u8; " << L << "], [i32; " << L << "], u32)] = &[\n";
  for (const auto& e : emitting) {
    f << "    (" << e.src << ", " << e.dst << ", [";
    for (size_t i = 0; i < L; ++i) f << (i ? ", " : "") << (int)e.deltas[i];
    f << "], [";
    for (size_t i = 0; i < L; ++i) f << (i ? ", " : "") << e.symIdx[i];
    f << "], " << e.weightIndex << "),\n";
  }
  f << "];\n\n";

  // ----- DP runner template (parameterised over reduce op) -----
  // We emit two functions, forward (reduce = log_sum_exp) and viterbi
  // (reduce = max), sharing the same DP body via a generic helper.
  f << R"RUST(
#[inline(always)]
fn lse(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY { return b; }
    if b == f64::NEG_INFINITY { return a; }
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
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
  auto emitDpBody = [&](std::ofstream& f, const char* reduce) {
    f << "    let lw = compute_log_weights(p);\n";
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
    f << "        // Emitting transitions into this cell.\n";
    f << "        for &(src, dst, deltas, syms, widx) in EMITTING_TRANSITIONS.iter() {\n";
    f << "            let mut ok = true;\n";
    f << "            let mut prev = cell;\n";
    f << "            for k in 0..NUM_LEAVES {\n";
    f << "                if deltas[k] == 1 {\n";
    f << "                    if idx[k] == 0 { ok = false; break; }\n";
    f << "                    unsafe {\n";
    f << "                        if *leaves[k].get_unchecked(idx[k] - 1) as i32 != syms[k] { ok = false; break; }\n";
    f << "                    }\n";
    f << "                    prev -= strides[k];\n";
    f << "                }\n";
    f << "            }\n";
    f << "            if ok {\n";
    f << "                unsafe {\n";
    f << "                    let sv = *g.get_unchecked((src as usize) * total + prev);\n";
    f << "                    if sv != f64::NEG_INFINITY {\n";
    f << "                        let dst_off = (dst as usize) * total + cell;\n";
    f << "                        let dv = *g.get_unchecked(dst_off);\n";
    f << "                        *g.get_unchecked_mut(dst_off) = " << reduce << "(dv, sv + *lw.get_unchecked(widx as usize));\n";
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

  f << "pub fn forward(p: &Params, leaves: [&[u32]; NUM_LEAVES]) -> f64 {\n";
  emitDpBody (f, "lse");
  f << "}\n\n";

  if (emitViterbi) {
    f << "pub fn viterbi(p: &Params, leaves: [&[u32]; NUM_LEAVES]) -> f64 {\n";
    emitDpBody (f, "max2");
    f << "}\n";
  }

  LogThisAt(2,"Wrote Rust crate to " << outputDir
            << " (states=" << N << ", leaves=" << L
            << ", alphabet=" << alph.size()
            << ", silent=" << silent.size() << ", emitting=" << emitting.size()
            << ", weights=" << weightsInOrder.size() << ")" << std::endl);
}

}  // namespace MachineBoss
