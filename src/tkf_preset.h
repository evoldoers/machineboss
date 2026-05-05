#ifndef TKF_PRESET_INCLUDED
#define TKF_PRESET_INCLUDED

#include <string>
#include "machine.h"

namespace MachineBoss {

namespace TkfPreset {

enum class Version { TKF91, TKF92 };
enum class Kind { Root, Branch };
enum class AlphabetKind { DNA, RNA, Protein, Binary, Unary, Custom };
enum class Model { JC, F81, K80, HKY85, ID };

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
