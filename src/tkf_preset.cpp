#include <stdexcept>
#include <iostream>
#include <set>

#include "tkf_preset.h"
#include "weight.h"

namespace MachineBoss {
namespace TkfPreset {

// ---------- Alphabet helpers ----------

vguard<string> alphabetSymbols (AlphabetKind kind, const string& customAlphabet) {
  switch (kind) {
  case AlphabetKind::DNA:    return { "A", "C", "G", "T" };
  case AlphabetKind::RNA:    return { "A", "C", "G", "U" };
  case AlphabetKind::Protein:
    return { "A","C","D","E","F","G","H","I","K","L",
             "M","N","P","Q","R","S","T","V","W","Y" };
  case AlphabetKind::Binary: return { "0", "1" };
  case AlphabetKind::Unary:  return { "X" };
  case AlphabetKind::Custom: {
    vguard<string> out;
    for (char c: customAlphabet) out.push_back (string (1, c));
    if (out.empty()) throw runtime_error ("custom alphabet must be non-empty");
    return out;
  }
  }
  throw runtime_error ("unknown alphabet kind");
}

bool isPurine (const string& s) { return s == "A" || s == "G"; }
static bool isNucleotideAlphabet (AlphabetKind k) {
  return k == AlphabetKind::DNA || k == AlphabetKind::RNA;
}

bool parseAlphabetKind (const string& s, AlphabetKind& out) {
  if (s == "dna")    { out = AlphabetKind::DNA;    return true; }
  if (s == "rna")    { out = AlphabetKind::RNA;    return true; }
  if (s == "prot")   { out = AlphabetKind::Protein; return true; }
  if (s == "binary") { out = AlphabetKind::Binary; return true; }
  if (s == "unary")  { out = AlphabetKind::Unary;  return true; }
  if (s == "custom") { out = AlphabetKind::Custom; return true; }
  return false;
}

bool parseModel (const string& s, Model& out) {
  if (s == "jc")        { out = Model::JC;        return true; }
  if (s == "f81")       { out = Model::F81;       return true; }
  if (s == "k80")       { out = Model::K80;       return true; }
  if (s == "hky85")     { out = Model::HKY85;     return true; }
  if (s == "id")        { out = Model::ID;        return true; }
  if (s == "telegraph") { out = Model::Telegraph; return true; }
  if (s == "bsc")       { out = Model::BSC;       return true; }
  if (s == "erasure")   { out = Model::Erasure;   return true; }
  return false;
}

static bool isBinaryOnlyModel (Model m) {
  return m == Model::Telegraph || m == Model::BSC || m == Model::Erasure;
}

// ---------- Validation ----------

static void validateSpec (const Spec& spec, const vguard<string>& alph) {
  if ((spec.model == Model::K80 || spec.model == Model::HKY85)
      && !isNucleotideAlphabet (spec.alphabetKind))
    throw runtime_error ("substitution model k80/hky85 requires DNA or RNA alphabet");
  if (isBinaryOnlyModel (spec.model)
      && spec.alphabetKind != AlphabetKind::Binary)
    throw runtime_error ("substitution model telegraph/bsc/erasure requires binary alphabet (use --...-binary-{telegraph,bsc,erasure})");
  if (spec.alphabetKind == AlphabetKind::Unary && spec.model != Model::ID)
    cerr << "Warning: unary alphabet with non-identity substitution model is degenerate; consider --tkfNN-XXX-unary-id" << endl;
  if (spec.model == Model::ID && alph.size() > 1 && spec.kind == Kind::Root)
    throw runtime_error ("identity substitution model on a root preset requires unary alphabet (no equilibrium frequencies are defined)");
  if (spec.model == Model::Erasure && spec.kind == Kind::Root)
    throw runtime_error ("erasure substitution model has a degenerate equilibrium (pi=(1,0)) and is not meaningful as a root preset");
}

// ---------- Substitution model expressions ----------

// Time parameter name per TKF version (matches the existing presets).
// IID reuses TKF91's "time" name.
static string timeParamName (Version v) {
  if (v == Version::TKF92) return string("t");
  return string("time");
}

// Equilibrium frequency expression for symbol `sym` under a given model.
// JC and K80 use uniform pi = 1/n (an int division expression).
// F81 and HKY85 use a free parameter pi_<sym>.
// ID does not have an equilibrium-frequency notion (root requires unary).
static WeightExpr piExpr (Model model, size_t alphSize, const string& sym) {
  switch (model) {
  case Model::JC:
  case Model::K80:
    return WeightAlgebra::divide (WeightAlgebra::one(),
                                  WeightAlgebra::intConstant ((int) alphSize));
  case Model::F81:
  case Model::HKY85:
    return WeightAlgebra::param (string("pi_") + sym);
  case Model::ID:
    if (alphSize == 1) return WeightAlgebra::one();
    throw runtime_error ("piExpr: identity model requires unary alphabet");
  case Model::Telegraph:
    // Stationary: pi_0 = rate10/(rate01+rate10), pi_1 = rate01/(rate01+rate10).
    return WeightAlgebra::param (string("pi_") + sym);
  case Model::BSC:
    // Symmetric: pi_0 = pi_1 = 1/2.
    return WeightAlgebra::divide (WeightAlgebra::one(),
                                  WeightAlgebra::intConstant (2));
  case Model::Erasure:
    throw runtime_error ("piExpr: erasure model has degenerate equilibrium and no root preset");
  }
  throw runtime_error ("unknown model");
}

// Map: in_sym, out_sym -> WeightExpr giving P(out | in, t).
//
// JC/F81: P(j|i,t) = pNoSub*delta(i,j) + pSub*pi_j, with pNoSub = exp(-t).
// K80   : standard 3-eigenvalue formula (uniform pi); transitions vs transversions distinguished.
// HKY85 : closed-form per Hasegawa-Kishino-Yano with free pi and ts/tv ratio.
// ID    : 1 if i == j else 0 (transitions with weight 0 are not emitted).
//
// All four make_*_defs functions populate ParamFuncs with the model's named
// helper expressions (pNoSub, pSub, ts/tv eigenvalue components, etc.) and
// return a function that maps (in, out) to its substitution weight.
struct SubModel {
  ParamDefs defs;
  // If returns null, the (in, out) transition is omitted (ID model off-diagonal).
  function<WeightExpr(const string&, const string&)> emit;
};

static SubModel makeSubModel (const Spec& spec, const vguard<string>& alph) {
  SubModel sm;
  const string tParam = timeParamName (spec.version);

  switch (spec.model) {

  case Model::JC: {
    // pNoSub = exp(-t); pSub = 1 - pNoSub; pSame = pNoSub + pDiff; pDiff = pSub/n.
    sm.defs["pNoSub"] = WeightAlgebra::expOf (
      WeightAlgebra::multiply (WeightAlgebra::intConstant(-1),
                               WeightAlgebra::param (tParam)));
    sm.defs["pSub"]  = WeightAlgebra::negate (WeightAlgebra::param ("pNoSub"));
    sm.defs["pDiff"] = WeightAlgebra::divide (WeightAlgebra::param ("pSub"),
                                              WeightAlgebra::intConstant ((int) alph.size()));
    sm.defs["pSame"] = WeightAlgebra::add (WeightAlgebra::param ("pNoSub"),
                                           WeightAlgebra::param ("pDiff"));
    sm.emit = [](const string& a, const string& b) -> WeightExpr {
      return a == b ? WeightAlgebra::param ("pSame") : WeightAlgebra::param ("pDiff");
    };
    return sm;
  }

  case Model::F81: {
    // pNoSub = exp(-t); pSub = 1 - pNoSub. P(j|i,t) = pNoSub*delta + pSub*pi_j.
    sm.defs["pNoSub"] = WeightAlgebra::expOf (
      WeightAlgebra::multiply (WeightAlgebra::intConstant(-1),
                               WeightAlgebra::param (tParam)));
    sm.defs["pSub"]   = WeightAlgebra::negate (WeightAlgebra::param ("pNoSub"));
    sm.emit = [](const string& a, const string& b) -> WeightExpr {
      WeightExpr piB = WeightAlgebra::param (string("pi_") + b);
      WeightExpr off = WeightAlgebra::multiply (WeightAlgebra::param ("pSub"), piB);
      if (a == b)
        return WeightAlgebra::add (WeightAlgebra::param ("pNoSub"), off);
      return off;
    };
    return sm;
  }

  case Model::K80: {
    // K80 with raw rates: transversion rate = 1, transition rate = tsRatio.
    // Eigenvalues of P(t):
    //   e_tv = exp(-(2 + 2*tsRatio*0)*t)... actually for K80 with raw rates:
    //   off-diagonal transversion = 1, transition = tsRatio. Diagonal = -(2 + tsRatio).
    //   Eigenvalues: 0, -2 (from purine vs pyrimidine asymmetry)... let me re-derive.
    // Following Kimura (1980) with rate normalisation alpha=tsRatio*beta, beta=1
    // (transversion rate = 1 per pair), alpha = tsRatio:
    //   P(stay,t)        = 1/4 (1 + exp(-4*t)         + 2*exp(-2*(1+tsRatio)*t))
    //   P(transition,t)  = 1/4 (1 + exp(-4*t)         - 2*exp(-2*(1+tsRatio)*t))
    //   P(transversion,t)= 1/4 (1 - exp(-4*t))
    WeightExpr tExpr = WeightAlgebra::param (tParam);
    WeightExpr neg4t = WeightAlgebra::multiply (WeightAlgebra::intConstant(-4), tExpr);
    sm.defs["k80_b"] = WeightAlgebra::expOf (neg4t);  // exp(-4*t)
    WeightExpr negTwoTsP1 = WeightAlgebra::multiply (
      WeightAlgebra::intConstant(-2),
      WeightAlgebra::multiply (
        WeightAlgebra::add (WeightAlgebra::one(), WeightAlgebra::param ("tsRatio")),
        tExpr));
    sm.defs["k80_a"] = WeightAlgebra::expOf (negTwoTsP1);  // exp(-2*(1+tsRatio)*t)
    WeightExpr quarter = WeightAlgebra::divide (WeightAlgebra::one(),
                                                 WeightAlgebra::intConstant(4));
    // pSame        = (1 + k80_b + 2*k80_a) / 4
    sm.defs["pSame"]  = WeightAlgebra::multiply (
      quarter,
      WeightAlgebra::add (
        WeightAlgebra::add (WeightAlgebra::one(), WeightAlgebra::param ("k80_b")),
        WeightAlgebra::multiply (WeightAlgebra::intConstant(2),
                                  WeightAlgebra::param ("k80_a"))));
    // pTransition  = (1 + k80_b - 2*k80_a) / 4
    sm.defs["pTransition"]  = WeightAlgebra::multiply (
      quarter,
      WeightAlgebra::subtract (
        WeightAlgebra::add (WeightAlgebra::one(), WeightAlgebra::param ("k80_b")),
        WeightAlgebra::multiply (WeightAlgebra::intConstant(2),
                                  WeightAlgebra::param ("k80_a"))));
    // pTransversion= (1 - k80_b) / 4   (per specific transversion target)
    sm.defs["pTransversion"]  = WeightAlgebra::multiply (
      quarter,
      WeightAlgebra::subtract (WeightAlgebra::one(), WeightAlgebra::param ("k80_b")));
    sm.emit = [](const string& a, const string& b) -> WeightExpr {
      if (a == b) return WeightAlgebra::param ("pSame");
      const bool aPur = isPurine (a), bPur = isPurine (b);
      if (aPur == bPur) return WeightAlgebra::param ("pTransition");  // A↔G or C↔T(/U)
      return WeightAlgebra::param ("pTransversion");
    };
    return sm;
  }

  case Model::HKY85: {
    // Hasegawa-Kishino-Yano (1985) with raw rate matrix Q_ij = pi_j*(tsRatio if
    // transition else 1) for i != j. Three nonzero eigenvalues:
    //   -1 (transversions decay with this rate)
    //   -A_R = -(pi_Y + tsRatio*pi_R)
    //   -A_Y = -(pi_R + tsRatio*pi_Y)
    // P(j|i,t) for self / same-group transition / transversion:
    //   self          : pi_i + pi_i*(1/pi_K - 1)*e_b + (pi_K - pi_i)/pi_K * e_a_K
    //   transition    : pi_j + pi_j*(1/pi_K - 1)*e_b - (pi_j / pi_K) * e_a_K
    //   transversion  : pi_j * (1 - e_b)
    // where K is i's group (R if i is purine, Y if pyrimidine), and e_b = exp(-t),
    // e_a_K = exp(-A_K * t).
    WeightExpr tExpr = WeightAlgebra::param (tParam);
    WeightExpr negT = WeightAlgebra::multiply (WeightAlgebra::intConstant(-1), tExpr);
    sm.defs["pNoSub"] = WeightAlgebra::expOf (negT);  // = e_b
    sm.defs["pSub"]   = WeightAlgebra::negate (WeightAlgebra::param ("pNoSub"));
    // For DNA the pyrimidines are C, T; for RNA they are C, U.
    const string pyrIs = (spec.alphabetKind == AlphabetKind::RNA) ? "U" : "T";
    sm.defs["pi_R"] = WeightAlgebra::add (WeightAlgebra::param ("pi_A"),
                                          WeightAlgebra::param ("pi_G"));
    sm.defs["pi_Y"] = WeightAlgebra::add (WeightAlgebra::param ("pi_C"),
                                          WeightAlgebra::param (string("pi_") + pyrIs));
    // A_R = pi_Y + tsRatio * pi_R; A_Y = pi_R + tsRatio * pi_Y.
    sm.defs["hky_A_R"] = WeightAlgebra::add (
      WeightAlgebra::param ("pi_Y"),
      WeightAlgebra::multiply (WeightAlgebra::param ("tsRatio"), WeightAlgebra::param ("pi_R")));
    sm.defs["hky_A_Y"] = WeightAlgebra::add (
      WeightAlgebra::param ("pi_R"),
      WeightAlgebra::multiply (WeightAlgebra::param ("tsRatio"), WeightAlgebra::param ("pi_Y")));
    sm.defs["hky_e_R"] = WeightAlgebra::expOf (
      WeightAlgebra::multiply (WeightAlgebra::param ("hky_A_R"),
                                WeightAlgebra::multiply (WeightAlgebra::intConstant(-1), tExpr)));
    sm.defs["hky_e_Y"] = WeightAlgebra::expOf (
      WeightAlgebra::multiply (WeightAlgebra::param ("hky_A_Y"),
                                WeightAlgebra::multiply (WeightAlgebra::intConstant(-1), tExpr)));
    sm.emit = [pyrIs](const string& a, const string& b) -> WeightExpr {
      const bool aPur = isPurine (a), bPur = isPurine (b);
      WeightExpr piB = WeightAlgebra::param (string("pi_") + b);
      WeightExpr piA = WeightAlgebra::param (string("pi_") + a);
      const string piK_a = aPur ? "pi_R" : "pi_Y";
      const string e_K_a = aPur ? "hky_e_R" : "hky_e_Y";
      WeightExpr piK = WeightAlgebra::param (piK_a);
      WeightExpr eK  = WeightAlgebra::param (e_K_a);
      WeightExpr eB  = WeightAlgebra::param ("pNoSub");
      // Common factor f = (1/piK - 1) * eB
      WeightExpr piKminusOne = WeightAlgebra::subtract (
        WeightAlgebra::reciprocal (piK), WeightAlgebra::one());
      WeightExpr f = WeightAlgebra::multiply (piKminusOne, eB);
      if (aPur != bPur) {
        // transversion: pi_b * (1 - eB)
        return WeightAlgebra::multiply (
          piB,
          WeightAlgebra::subtract (WeightAlgebra::one(), eB));
      }
      if (a == b) {
        // self: piA + piA*f + (piK - piA)/piK * eK
        WeightExpr term1 = WeightAlgebra::multiply (piA, f);
        WeightExpr term2 = WeightAlgebra::multiply (
          WeightAlgebra::divide (
            WeightAlgebra::subtract (piK, piA), piK),
          eK);
        return WeightAlgebra::add (
          WeightAlgebra::add (piA, term1), term2);
      }
      // same-group transition: piB + piB*f - (piB / piK) * eK
      WeightExpr term1 = WeightAlgebra::multiply (piB, f);
      WeightExpr term2 = WeightAlgebra::multiply (
        WeightAlgebra::divide (piB, piK), eK);
      return WeightAlgebra::subtract (
        WeightAlgebra::add (piB, term1), term2);
    };
    return sm;
  }

  case Model::ID: {
    // No substitution: P(j|i) = 1 if j == i else 0.
    sm.emit = [](const string& a, const string& b) -> WeightExpr {
      return a == b ? WeightAlgebra::one() : WeightExpr();  // null = omit
    };
    return sm;
  }

  case Model::Telegraph: {
    // Asymmetric 2-state CTMC on {0, 1}:
    //   Q = [[-rate01, rate01], [rate10, -rate10]]
    //   λ = rate01 + rate10  (non-zero eigenvalue)
    //   pi_0 = rate10/λ, pi_1 = rate01/λ (stationary)
    //   P(0|0,t) = pi_0 + pi_1 * exp(-λt)
    //   P(1|0,t) = pi_1 * (1 - exp(-λt))
    //   P(0|1,t) = pi_0 * (1 - exp(-λt))
    //   P(1|1,t) = pi_1 + pi_0 * exp(-λt)
    WeightExpr tExpr = WeightAlgebra::param (tParam);
    WeightExpr sumRate = WeightAlgebra::add (WeightAlgebra::param ("rate01"),
                                              WeightAlgebra::param ("rate10"));
    sm.defs["telLambdaT"] = WeightAlgebra::multiply (sumRate, tExpr);
    sm.defs["telPNoSub"]  = WeightAlgebra::expOf (
      WeightAlgebra::multiply (WeightAlgebra::intConstant(-1),
                                WeightAlgebra::param ("telLambdaT")));
    sm.defs["telPSub"]    = WeightAlgebra::negate (WeightAlgebra::param ("telPNoSub"));
    // pi_0 and pi_1 are functions of (rate01, rate10), but we expose them
    // as the per-symbol pi parameters so the rest of the pipeline (root
    // preset, intersection sums) can use the same `pi_X` convention.
    sm.defs["pi_0"] = WeightAlgebra::divide (WeightAlgebra::param ("rate10"), sumRate);
    sm.defs["pi_1"] = WeightAlgebra::divide (WeightAlgebra::param ("rate01"), sumRate);
    sm.emit = [](const string& a, const string& b) -> WeightExpr {
      WeightExpr piB = WeightAlgebra::param (string("pi_") + b);
      WeightExpr offDiag = WeightAlgebra::multiply (WeightAlgebra::param ("telPSub"), piB);
      if (a == b) {
        // P(a|a) = pi_a + pi_~a * pNoSub = pi_a*(1 - pNoSub) + pNoSub = pi_a*pSub + pNoSub
        WeightExpr piA = WeightAlgebra::param (string("pi_") + a);
        WeightExpr selfTerm = WeightAlgebra::multiply (
          WeightAlgebra::param ("telPSub"), piA);
        return WeightAlgebra::add (WeightAlgebra::param ("telPNoSub"), selfTerm);
      }
      return offDiag;
    };
    return sm;
  }

  case Model::BSC: {
    // Binary symmetric channel: rate01 = rate10 = flipRate. λ = 2*flipRate.
    //   pi_0 = pi_1 = 1/2.
    //   P(stay) = 1/2 + 1/2 * exp(-2*flipRate*t)
    //   P(flip) = 1/2 - 1/2 * exp(-2*flipRate*t)
    WeightExpr tExpr = WeightAlgebra::param (tParam);
    WeightExpr negLam = WeightAlgebra::multiply (
      WeightAlgebra::intConstant (-2),
      WeightAlgebra::multiply (WeightAlgebra::param ("flipRate"), tExpr));
    sm.defs["bscPNoFlip"] = WeightAlgebra::expOf (negLam);
    WeightExpr half = WeightAlgebra::divide (WeightAlgebra::one(),
                                              WeightAlgebra::intConstant (2));
    sm.defs["pStay"] = WeightAlgebra::multiply (
      half,
      WeightAlgebra::add (WeightAlgebra::one(), WeightAlgebra::param ("bscPNoFlip")));
    sm.defs["pFlip"] = WeightAlgebra::multiply (
      half,
      WeightAlgebra::subtract (WeightAlgebra::one(), WeightAlgebra::param ("bscPNoFlip")));
    sm.emit = [](const string& a, const string& b) -> WeightExpr {
      return a == b ? WeightAlgebra::param ("pStay") : WeightAlgebra::param ("pFlip");
    };
    return sm;
  }

  case Model::Erasure: {
    // Binary erasure channel: 0 is absorbing; 1 → 0 with rate eraseRate.
    //   Q = [[0, 0], [eraseRate, -eraseRate]]
    //   P(0|0,t) = 1; P(1|0,t) = 0 (omit)
    //   P(1|1,t) = exp(-eraseRate*t); P(0|1,t) = 1 - exp(-eraseRate*t)
    WeightExpr tExpr = WeightAlgebra::param (tParam);
    WeightExpr negET = WeightAlgebra::multiply (
      WeightAlgebra::intConstant (-1),
      WeightAlgebra::multiply (WeightAlgebra::param ("eraseRate"), tExpr));
    sm.defs["pPersist"] = WeightAlgebra::expOf (negET);
    sm.defs["pErased"]  = WeightAlgebra::negate (WeightAlgebra::param ("pPersist"));
    sm.emit = [](const string& a, const string& b) -> WeightExpr {
      if (a == "0" && b == "0") return WeightAlgebra::one();
      if (a == "0" && b == "1") return WeightExpr();        // 0 absorbing → omit
      if (a == "1" && b == "1") return WeightAlgebra::param ("pPersist");
      if (a == "1" && b == "0") return WeightAlgebra::param ("pErased");
      throw runtime_error ("erasure model expects binary alphabet {\"0\",\"1\"}");
    };
    return sm;
  }

  } // switch
  throw runtime_error ("unknown substitution model");
}

// ---------- Constraints helper ----------
//
// Populate cons.{rate,prob,norm} so that --use-defaults, normalisation, and
// training operations have sensible defaults for every free parameter.

// Rate parameters for the substitution model only (NOT insRate/delRate).
// Used by both branch and root constraint helpers.
static void addSubModelRateConstraints (Machine& m, Model model) {
  switch (model) {
  case Model::HKY85:                 m.cons.rate.push_back ("tsRatio");   break;
  case Model::Telegraph:
    m.cons.rate.push_back ("rate01");
    m.cons.rate.push_back ("rate10");
    break;
  case Model::BSC:                   m.cons.rate.push_back ("flipRate");  break;
  case Model::Erasure:               m.cons.rate.push_back ("eraseRate"); break;
  default: break;  // JC, F81, K80, ID — no extra rates beyond the per-branch time
  }
}

static void addConstraintsForSpec (Machine& m, const Spec& spec, const vguard<string>& alph) {
  // Rate parameters (default 1).
  m.cons.rate.push_back (timeParamName (spec.version));
  if (spec.version != Version::IID) {
    m.cons.rate.push_back ("insRate");
    m.cons.rate.push_back ("delRate");
  }
  addSubModelRateConstraints (m, spec.model);
  // TKF92 fragment-extension is a probability (default 0.5).
  if (spec.version == Version::TKF92)
    m.cons.prob.push_back ("r");
  // Free pi parameters (default uniform 1/n) — only for F81 and HKY85.
  // Telegraph's pi_0/pi_1 are DERIVED from (rate01, rate10) via defs,
  // so they are not added to cons (no free pi parameters).
  if (spec.model == Model::F81 || spec.model == Model::HKY85) {
    vguard<string> piGroup;
    for (const auto& s: alph) piGroup.push_back (string("pi_") + s);
    m.cons.norm.push_back (piGroup);
  }
}

// ---------- TKF BDI defs ----------
//
// Common TKF91/TKF92 derived quantities: pNoDeletion, pDescendants, etc.
// These are used by both the root (kappa = pExtend) and branch transducers.

static void addTkfBdiDefs (ParamDefs& defs, Version version) {
  const string tp = timeParamName (version);
  defs["pNoDeletion"] = WeightAlgebra::expOf (
    WeightAlgebra::multiply (WeightAlgebra::intConstant(-1),
                             WeightAlgebra::multiply (WeightAlgebra::param ("delRate"),
                                                      WeightAlgebra::param (tp))));
  defs["pDeletion"] = WeightAlgebra::negate (WeightAlgebra::param ("pNoDeletion"));
  defs["pNoInsertion"] = WeightAlgebra::expOf (
    WeightAlgebra::multiply (WeightAlgebra::intConstant(-1),
                             WeightAlgebra::multiply (WeightAlgebra::param ("insRate"),
                                                      WeightAlgebra::param (tp))));
  defs["pInsertion"] = WeightAlgebra::negate (WeightAlgebra::param ("pNoInsertion"));
  defs["delInsRatio"] = WeightAlgebra::divide (WeightAlgebra::param ("pNoDeletion"),
                                                WeightAlgebra::param ("pNoInsertion"));
  // pDescendants = (insRate * (1 - delInsRatio)) / (delRate - insRate * delInsRatio)
  defs["pDescendants"] = WeightAlgebra::divide (
    WeightAlgebra::multiply (WeightAlgebra::param ("insRate"),
                             WeightAlgebra::negate (WeightAlgebra::param ("delInsRatio"))),
    WeightAlgebra::subtract (
      WeightAlgebra::param ("delRate"),
      WeightAlgebra::multiply (WeightAlgebra::param ("insRate"),
                               WeightAlgebra::param ("delInsRatio"))));
  defs["pNoDescendants"] = WeightAlgebra::negate (WeightAlgebra::param ("pDescendants"));
}

// ---------- Root machine: TKF91-flavour or TKF92-flavour ----------
//
// TKF91 root (geometric):
//   P(L=k) = κ^k (1−κ)
//   2-state machine: emit-or-stop (single emit state with self-loops + stop edge).
//
// TKF92 root (ν-modified geometric, κ = insRate/delRate, ν = r):
//   P(L=0) = 1−κ
//   P(L≥1) = κ · ν^(k−1) · (1−ν)
//   4-state machine: begin → emit → decide → stop, where decide either loops
//   back to emit (with weight ν) or goes to stop (with weight 1−ν). The two
//   distinct length-0 (begin → stop directly) and length≥1 (begin → emit
//   first, then geometric-with-ν tail) regimes are exactly what the
//   ihh/tkf-mixdom derivation specifies for the TKF92 singlet.
//
// `pi_X` defaults to 1/|alphabet| for JC/K80/Identity, and is a free
// stationary-frequency parameter for F81/HKY85.

static void addRootConstraints (Machine& m, const Spec& spec, const vguard<string>& alph) {
  addSubModelRateConstraints (m, spec.model);
  if (spec.version == Version::IID) {
    m.cons.prob.push_back ("pExtend");      // free length-extension probability
  } else {
    m.cons.rate.push_back ("insRate");
    m.cons.rate.push_back ("delRate");
    if (spec.version == Version::TKF92) m.cons.prob.push_back ("r");
  }
  if (spec.model == Model::F81 || spec.model == Model::HKY85
      || spec.model == Model::Telegraph) {
    vguard<string> piGroup;
    for (const auto& s: alph) piGroup.push_back (string("pi_") + s);
    m.cons.norm.push_back (piGroup);
  }
}

static Machine buildTkf91Root (const Spec& spec, const vguard<string>& alph, const SubModel& sm) {
  Machine m;
  m.state.resize(2);
  m.state[0].name = json("emit");
  m.state[1].name = json("stop");

  m.funcs.defs = sm.defs;
  m.funcs.defs["pExtend"]   = WeightAlgebra::divide (WeightAlgebra::param ("insRate"),
                                                      WeightAlgebra::param ("delRate"));
  m.funcs.defs["pNoExtend"] = WeightAlgebra::negate (WeightAlgebra::param ("pExtend"));

  for (size_t k = 0; k < alph.size(); ++k) {
    WeightExpr pi = piExpr (spec.model, alph.size(), alph[k]);
    WeightExpr w  = WeightAlgebra::multiply (WeightAlgebra::param ("pExtend"), pi);
    m.state[0].trans.push_back (MachineTransition (string(), alph[k], 0, w));
  }
  m.state[0].trans.push_back (MachineTransition (string(), string(), 1,
                                                  WeightAlgebra::param ("pNoExtend")));

  addRootConstraints (m, spec, alph);
  return m;
}

static Machine buildTkf92Root (const Spec& spec, const vguard<string>& alph, const SubModel& sm) {
  // TKF92 ν-modified geometric, P(L=0)=1−κ, P(L=k≥1)=κ·ν^(k−1)·(1−ν),
  // where κ = insRate/delRate, ν = r + (1−r)·κ. Realised as a 3-state
  // S/I/E shape (the M, D states of the branch SMIDE collapse for a
  // root: there's no input, so neither match nor delete applies):
  //
  //   0 start   emit X (κ·π_X) → insert;  silent → end (1−κ)
  //   1 insert  emit X (ν·π_X) → insert;  silent → end (1−ν)
  //   2 end
  //
  // The *first* emission carries κ; subsequent emissions carry ν. All
  // silent transitions are strictly forward, so standalone Forward DP
  // runs without any composition pre-processing.
  //
  // Earlier 4-state designs put a silent decide-state loop between
  // `emit` and `decide`, which (a) was incorrectly using `r` directly
  // instead of ν, producing P(L=k≥1) = κ·r^(k−1)·(1−r), and (b) caused
  // standalone Forward to fail topological sort. Both are fixed here.
  Machine m;
  m.state.resize(3);
  m.state[0].name = json("start");
  m.state[1].name = json("insert");
  m.state[2].name = json("end");

  m.funcs.defs = sm.defs;
  m.funcs.defs["pExtend"]   = WeightAlgebra::divide (WeightAlgebra::param ("insRate"),
                                                      WeightAlgebra::param ("delRate"));
  m.funcs.defs["pNoExtend"] = WeightAlgebra::negate (WeightAlgebra::param ("pExtend"));
  // ν = r + (1−r)·κ  (κ = pExtend = insRate/delRate)
  m.funcs.defs["nu"]        = WeightAlgebra::add (
    WeightAlgebra::param ("r"),
    WeightAlgebra::multiply (
      WeightAlgebra::negate (WeightAlgebra::param ("r")),
      WeightAlgebra::param ("pExtend")));
  m.funcs.defs["pNoNu"]     = WeightAlgebra::negate (WeightAlgebra::param ("nu"));

  // state 0 (emit_first): emit X with weight κ·π_X → emit_more
  for (size_t k = 0; k < alph.size(); ++k) {
    WeightExpr pi = piExpr (spec.model, alph.size(), alph[k]);
    WeightExpr w  = WeightAlgebra::multiply (WeightAlgebra::param ("pExtend"), pi);
    m.state[0].trans.push_back (MachineTransition (string(), alph[k], 1, w));
  }
  // state 0 → stop (silent, weight 1−κ)
  m.state[0].trans.push_back (MachineTransition (string(), string(), 2,
                                                  WeightAlgebra::param ("pNoExtend")));
  // state 1 (emit_more): emit X with weight ν·π_X → emit_more (self-loop)
  for (size_t k = 0; k < alph.size(); ++k) {
    WeightExpr pi = piExpr (spec.model, alph.size(), alph[k]);
    WeightExpr w  = WeightAlgebra::multiply (WeightAlgebra::param ("nu"), pi);
    m.state[1].trans.push_back (MachineTransition (string(), alph[k], 1, w));
  }
  // state 1 → stop (silent, weight 1−ν)
  m.state[1].trans.push_back (MachineTransition (string(), string(), 2,
                                                  WeightAlgebra::param ("pNoNu")));

  addRootConstraints (m, spec, alph);
  return m;
}

// IID root: same shape as TKF91 root (emit-or-stop) but with a free
// `pExtend` probability instead of pExtend = insRate/delRate. Matches the
// "geometric-length emitter with substitution-model emission frequencies"
// interpretation of iid as a zero-indel-rate degeneration of TKF91.
static Machine buildIidRoot (const Spec& spec, const vguard<string>& alph, const SubModel& sm) {
  Machine m;
  m.state.resize(2);
  m.state[0].name = json("emit");
  m.state[1].name = json("stop");

  m.funcs.defs = sm.defs;
  m.funcs.defs["pNoExtend"] = WeightAlgebra::negate (WeightAlgebra::param ("pExtend"));

  for (size_t k = 0; k < alph.size(); ++k) {
    WeightExpr pi = piExpr (spec.model, alph.size(), alph[k]);
    WeightExpr w  = WeightAlgebra::multiply (WeightAlgebra::param ("pExtend"), pi);
    m.state[0].trans.push_back (MachineTransition (string(), alph[k], 0, w));
  }
  m.state[0].trans.push_back (MachineTransition (string(), string(), 1,
                                                  WeightAlgebra::param ("pNoExtend")));
  addRootConstraints (m, spec, alph);
  return m;
}

static Machine buildRoot (const Spec& spec, const vguard<string>& alph, const SubModel& sm) {
  if (spec.version == Version::IID)   return buildIidRoot   (spec, alph, sm);
  if (spec.version == Version::TKF91) return buildTkf91Root (spec, alph, sm);
  return buildTkf92Root (spec, alph, sm);
}

// ---------- TKF91 branch: 7-state structure ----------
//
// Matches preset/tkf91-branch-dna-jc.json. States: begin (0), orphan (1),
// wait (2), match (3), insert (4), delete (5), end (6).
//
// Match-state emission weight = sm.emit(a, b) (e.g. pSame/pDiff for JC).
// Insert-state emission weight = pi_b for the descendant symbol.

static Machine buildTkf91Branch (const Spec& spec, const vguard<string>& alph, const SubModel& sm) {
  Machine m;
  m.state.resize(7);
  m.state[0].name = json("begin");
  m.state[1].name = json("orphan");
  m.state[2].name = json("wait");
  m.state[3].name = json("match");
  m.state[4].name = json("insert");
  m.state[5].name = json("delete");
  m.state[6].name = json("end");

  m.funcs.defs = sm.defs;
  addTkfBdiDefs (m.funcs.defs, spec.version);
  m.funcs.defs["pNoOrphans"] = WeightAlgebra::multiply (
    WeightAlgebra::divide (WeightAlgebra::param ("delRate"), WeightAlgebra::param ("insRate")),
    WeightAlgebra::divide (WeightAlgebra::param ("pDescendants"), WeightAlgebra::param ("pDeletion")));
  m.funcs.defs["pOrphans"] = WeightAlgebra::negate (WeightAlgebra::param ("pNoOrphans"));

  // begin -> insert (pDescendants), begin -> wait (pNoDescendants)
  m.state[0].trans.push_back (MachineTransition (string(), string(), 4,
    WeightAlgebra::param ("pDescendants")));
  m.state[0].trans.push_back (MachineTransition (string(), string(), 2,
    WeightAlgebra::param ("pNoDescendants")));
  // orphan -> insert (pOrphans), orphan -> wait (pNoOrphans)
  m.state[1].trans.push_back (MachineTransition (string(), string(), 4,
    WeightAlgebra::param ("pOrphans")));
  m.state[1].trans.push_back (MachineTransition (string(), string(), 2,
    WeightAlgebra::param ("pNoOrphans")));
  // wait -> match (pNoDeletion), wait -> delete (pDeletion), wait -> end (1)
  m.state[2].trans.push_back (MachineTransition (string(), string(), 3,
    WeightAlgebra::param ("pNoDeletion")));
  m.state[2].trans.push_back (MachineTransition (string(), string(), 5,
    WeightAlgebra::param ("pDeletion")));
  m.state[2].trans.push_back (MachineTransition (string(), string(), 6,
    WeightAlgebra::one()));
  // match -> begin: in=a, out=b, weight=sm.emit(a,b)
  for (const auto& a: alph)
    for (const auto& b: alph) {
      WeightExpr w = sm.emit (a, b);
      if (!w) continue;  // ID off-diagonal: no transition
      m.state[3].trans.push_back (MachineTransition (a, b, 0, w));
    }
  // insert -> begin: out=b, weight=pi_b
  for (const auto& b: alph) {
    WeightExpr pi = piExpr (spec.model, alph.size(), b);
    m.state[4].trans.push_back (MachineTransition (string(), b, 0, pi));
  }
  // delete -> orphan: in=a, weight=1
  for (const auto& a: alph)
    m.state[5].trans.push_back (MachineTransition (a, string(), 1, WeightAlgebra::one()));

  addConstraintsForSpec (m, spec, alph);
  return m;
}

// ---------- TKF92 branch: 5-state canonical WFST ----------
//
// Matches preset/tkf92-branch-prot-f81.json. States: begin (0), match (1),
// insert (2), delete (3), end (4). Each non-end source has full mat/ins/del/fin
// transitions with named t_{a,b} weights from tkf92-wfst-derivation.tex.

static Machine buildTkf92Branch (const Spec& spec, const vguard<string>& alph, const SubModel& sm) {
  Machine m;
  m.state.resize(5);
  m.state[0].name = json("begin");
  m.state[1].name = json("match");
  m.state[2].name = json("insert");
  m.state[3].name = json("delete");
  m.state[4].name = json("end");

  m.funcs.defs = sm.defs;
  addTkfBdiDefs (m.funcs.defs, spec.version);
  m.funcs.defs["pOrphans"] = WeightAlgebra::negate (
    WeightAlgebra::multiply (
      WeightAlgebra::divide (WeightAlgebra::param ("delRate"), WeightAlgebra::param ("insRate")),
      WeightAlgebra::divide (WeightAlgebra::param ("pDescendants"), WeightAlgebra::param ("pDeletion"))));
  m.funcs.defs["pNoOrphans"] = WeightAlgebra::negate (WeightAlgebra::param ("pOrphans"));
  // TKF92 singlet: kappa = lambda/mu (continuation prob at equilibrium),
  // pSinglet = r + (1-r)*kappa (ext + (1-ext)*kappa from the derivation).
  m.funcs.defs["kappa"] = WeightAlgebra::divide (WeightAlgebra::param ("insRate"),
                                                 WeightAlgebra::param ("delRate"));
  m.funcs.defs["pSinglet"] = WeightAlgebra::add (
    WeightAlgebra::param ("r"),
    WeightAlgebra::multiply (
      WeightAlgebra::negate (WeightAlgebra::param ("r")),
      WeightAlgebra::param ("kappa")));

  // sta row
  m.funcs.defs["tStaMat"] = WeightAlgebra::multiply (
    WeightAlgebra::param ("pNoDescendants"), WeightAlgebra::param ("pNoDeletion"));
  m.funcs.defs["tStaIns"] = WeightAlgebra::param ("pDescendants");
  m.funcs.defs["tStaDel"] = WeightAlgebra::multiply (
    WeightAlgebra::param ("pNoDescendants"), WeightAlgebra::param ("pDeletion"));
  m.funcs.defs["tStaFin"] = WeightAlgebra::param ("pNoDescendants");

  auto build_TmatNum_or_TdelNum = [](bool selfLoop, const string& bdiNoIns, const string& alphaOrComp) {
    // [r + (1-r)*pNoDescOrOrphans*kappa*(pNoDel or pDel)] / pSinglet (selfLoop=true)
    //  or
    // [(1-r)*pNoDescOrOrphans*kappa*(pNoDel or pDel)] / pSinglet (selfLoop=false)
    WeightExpr inner = WeightAlgebra::multiply (
      WeightAlgebra::param (bdiNoIns),
      WeightAlgebra::multiply (WeightAlgebra::param ("kappa"),
                                WeightAlgebra::param (alphaOrComp)));
    WeightExpr scaled = WeightAlgebra::multiply (
      WeightAlgebra::negate (WeightAlgebra::param ("r")), inner);
    WeightExpr num = selfLoop
      ? WeightAlgebra::add (WeightAlgebra::param ("r"), scaled)
      : scaled;
    return WeightAlgebra::divide (num, WeightAlgebra::param ("pSinglet"));
  };

  // mat row
  m.funcs.defs["tMatMat"] = build_TmatNum_or_TdelNum (true,  "pNoDescendants", "pNoDeletion");
  m.funcs.defs["tMatIns"] = WeightAlgebra::multiply (
    WeightAlgebra::negate (WeightAlgebra::param ("r")), WeightAlgebra::param ("pDescendants"));
  m.funcs.defs["tMatDel"] = build_TmatNum_or_TdelNum (false, "pNoDescendants", "pDeletion");
  m.funcs.defs["tMatFin"] = WeightAlgebra::param ("pNoDescendants");

  // ins row
  m.funcs.defs["tInsMat"] = build_TmatNum_or_TdelNum (false, "pNoDescendants", "pNoDeletion");
  m.funcs.defs["tInsIns"] = WeightAlgebra::add (
    WeightAlgebra::param ("r"),
    WeightAlgebra::multiply (WeightAlgebra::negate (WeightAlgebra::param ("r")),
                              WeightAlgebra::param ("pDescendants")));
  m.funcs.defs["tInsDel"] = WeightAlgebra::param ("tMatDel");
  m.funcs.defs["tInsFin"] = WeightAlgebra::param ("pNoDescendants");

  // del row
  m.funcs.defs["tDelMat"] = build_TmatNum_or_TdelNum (false, "pNoOrphans", "pNoDeletion");
  m.funcs.defs["tDelIns"] = WeightAlgebra::multiply (
    WeightAlgebra::negate (WeightAlgebra::param ("r")), WeightAlgebra::param ("pOrphans"));
  m.funcs.defs["tDelDel"] = build_TmatNum_or_TdelNum (true,  "pNoOrphans", "pDeletion");
  m.funcs.defs["tDelFin"] = WeightAlgebra::param ("pNoOrphans");

  // Per-source row builder.
  auto rowFor = [&](StateIndex src,
                    const string& tMatRef, const string& tInsRef,
                    const string& tDelRef, const string& tFinRef) {
    // src -> match (X | Y)
    for (const auto& x: alph)
      for (const auto& y: alph) {
        WeightExpr emit = sm.emit (x, y);
        if (!emit) continue;
        WeightExpr w = WeightAlgebra::multiply (WeightAlgebra::param (tMatRef), emit);
        m.state[src].trans.push_back (MachineTransition (x, y, 1, w));
      }
    // src -> insert (eps | Y)
    for (const auto& y: alph) {
      WeightExpr pi = piExpr (spec.model, alph.size(), y);
      WeightExpr w = WeightAlgebra::multiply (WeightAlgebra::param (tInsRef), pi);
      m.state[src].trans.push_back (MachineTransition (string(), y, 2, w));
    }
    // src -> delete (X | eps)
    for (const auto& x: alph)
      m.state[src].trans.push_back (MachineTransition (x, string(), 3,
        WeightAlgebra::param (tDelRef)));
    // src -> end (eps | eps)
    m.state[src].trans.push_back (MachineTransition (string(), string(), 4,
      WeightAlgebra::param (tFinRef)));
  };
  rowFor (0, "tStaMat", "tStaIns", "tStaDel", "tStaFin");
  rowFor (1, "tMatMat", "tMatIns", "tMatDel", "tMatFin");
  rowFor (2, "tInsMat", "tInsIns", "tInsDel", "tInsFin");
  rowFor (3, "tDelMat", "tDelIns", "tDelDel", "tDelFin");

  addConstraintsForSpec (m, spec, alph);
  return m;
}

// IID branch: zero-indel-rate limit of TKF91. Single emit state (with
// optional substitution-model omissions, e.g. for ID or Erasure) plus a
// silent exit edge to end. No insert/delete states.
static Machine buildIidBranch (const Spec& spec, const vguard<string>& alph, const SubModel& sm) {
  Machine m;
  m.state.resize(2);
  m.state[0].name = json("begin");
  m.state[1].name = json("end");

  m.funcs.defs = sm.defs;

  for (const auto& a: alph)
    for (const auto& b: alph) {
      WeightExpr w = sm.emit (a, b);
      if (!w) continue;  // ID off-diagonal / Erasure 0→1: omitted
      m.state[0].trans.push_back (MachineTransition (a, b, 0, w));
    }
  m.state[0].trans.push_back (MachineTransition (string(), string(), 1,
                                                  WeightAlgebra::one()));

  addConstraintsForSpec (m, spec, alph);
  return m;
}

Machine build (const Spec& spec) {
  const vguard<string> alph = alphabetSymbols (spec.alphabetKind, spec.customAlphabet);
  validateSpec (spec, alph);
  const SubModel sm = makeSubModel (spec, alph);
  if (spec.kind == Kind::Root)        return buildRoot         (spec, alph, sm);
  if (spec.version == Version::IID)   return buildIidBranch   (spec, alph, sm);
  if (spec.version == Version::TKF91) return buildTkf91Branch (spec, alph, sm);
  return buildTkf92Branch (spec, alph, sm);
}

}  // namespace TkfPreset
}  // namespace MachineBoss
