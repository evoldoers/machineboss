#ifndef TKF_PRESET_INCLUDED
#define TKF_PRESET_INCLUDED

#include <string>
#include "machine.h"

namespace MachineBoss {

namespace TkfPreset {

// IID is a zero-indel-rate degenerate sibling of TKF91/TKF92: the branch
// transducer has no insert/delete states, only matches; the root is a
// geometric-length emitter with a free `pExtend` probability.
//
// EVOLMOVES is a TKF92-with-non-zero-inflated-singlet variant: the root
// is a plain ν-geometric (P(L=0)=1−ν, P(L=k≥1)=ν^k(1−ν)) instead of the
// standard TKF92 ν-modified geometric (P(L=0)=1−κ, P(L=k≥1)=κ·ν^(k−1)·(1−ν));
// the branch is the 5-state regularised conditional pair HMM that
// composes with this singlet to recover the standard TKF92 joint pair
// HMM matrix. evolmoves uses these for its move proposals and likelihood
// computations.
enum class Version { TKF91, TKF92, IID, Evolmoves };
enum class Kind { Root, Branch };
enum class AlphabetKind { DNA, RNA, Protein, Binary, Unary, Custom };
// Telegraph, BSC, and Erasure are binary-only substitution kernels:
//   Telegraph: 2-state CTMC with independent rates rate01, rate10.
//   BSC:       symmetric Telegraph (rate01 == rate10 == flipRate).
//   Erasure:   0-absorbing Telegraph (rate01 = 0; only 1→0 with eraseRate).
enum class Model { JC, F81, K80, HKY85, ID, Telegraph, BSC, Erasure };

struct Spec {
  Version version = Version::TKF91;
  Kind kind = Kind::Branch;
  AlphabetKind alphabetKind = AlphabetKind::DNA;
  Model model = Model::JC;
  string customAlphabet;  // only used when alphabetKind == Custom
};

// Resolve the alphabet symbols. For Custom, customAlphabet is interpreted as
// a string of single-character tokens.
vguard<string> alphabetSymbols (AlphabetKind kind, const string& customAlphabet = "");

// True if a symbol is a purine when the alphabet kind is DNA or RNA.
// (A and G are purines; C, T, U are pyrimidines.) Unused for non-nucleotide alphabets.
bool isPurine (const string& symbol);

// Build the TKF machine for the given spec. Throws on validation errors
// (e.g. K80/HKY85 with non-nucleotide alphabet) and emits a warning to
// stderr for unary alphabets with a non-Identity model.
Machine build (const Spec& spec);

// Parsers used by the CLI dispatcher.
bool parseAlphabetKind (const string& s, AlphabetKind& out);
bool parseModel (const string& s, Model& out);

}  // namespace TkfPreset

}  // namespace MachineBoss

#endif /* TKF_PRESET_INCLUDED */
