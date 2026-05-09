#ifndef ARGPARSE_INCLUDED
#define ARGPARSE_INCLUDED

#include <iosfwd>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// Lightweight command-line option parser, modelled closely on the subset
// of boost::program_options used by boss.cpp and Logger::parseLogArgs.
//
// Supported features:
//   - long options:  --name, --name=value, --name value
//   - short options: -x, -x value (attached `-xvalue` is NOT supported;
//                    boss never used that form)
//   - "name,short" combined declaration
//   - typed values: string, int, vector<string> (multi-occurrence)
//   - optional default for the int/string types
//   - flag-only options (no associated value)
//   - allow_unregistered: unknown long-options and positionals are kept
//     in declaration order in the unrecognized list so callers can
//     dispatch on them (boss uses this for its postfix-operator pseudo-
//     flags like --tkf92-branch-dna-hky85)
//   - help-text rendering grouped by OptionsDescription title, two-column
//     "  --long-name [=arg]   first line of description …
//                             continuation indented" formatting
//
// Out of scope: required-options validation (boss has no required opts);
// positional-options binding; environment-variable fallbacks; multitoken;
// composing/zero_tokens semantics.

namespace MachineBoss {
namespace argparse {

using std::map;
using std::string;
using std::vector;

// ---- Value spec ----

struct ValueSpec {
  enum Kind { Flag, Str, Int, VecStr };
  Kind kind = Str;
  bool has_default = false;
  string default_str;

  // Chainable in the boost style: `value<int>()->default_value("2")`.
  ValueSpec* default_value (const string& v);
  ValueSpec* default_value (int v);
};

// Allocate a fresh ValueSpec on a process-wide owning slab.
// Specialised in argparse.cpp for string, int, vector<string>.
template<typename T> ValueSpec* value();

// ---- Option / group declarations ----

struct OptDesc {
  string _long_name;     // without the leading --
  string _short_name;    // single char, may be empty
  ValueSpec* spec = nullptr;  // null => pure flag
  string _description;

  // Boost-compatible accessors (boss.cpp uses desc->long_name(), etc.)
  const string& long_name() const { return _long_name; }
  const string& description() const { return _description; }
};

struct OptionsDescription {
  string title;
  vector<OptDesc> opts;

  OptionsDescription() = default;
  explicit OptionsDescription (const string& t) : title(t) {}

  OptionsDescription& add (const OptionsDescription& other) {
    for (const auto& o : other.opts) opts.push_back (o);
    return *this;
  }

  // boost::options_description::find_nothrow — match by long or short name.
  // boss uses this for help-text dispatch on already-stripped command tokens
  // ("-v", "--verbose", "verbose"); we accept any of those forms.
  const OptDesc* find_nothrow (const string& name, bool /*approx*/) const {
    string n = name;
    if (n.size() > 2 && n[0] == '-' && n[1] == '-') n = n.substr(2);
    else if (n.size() == 2 && n[0] == '-')           n = n.substr(1);
    for (const auto& o : opts) {
      if (o._long_name == n) return &o;
      if (!o._short_name.empty() && o._short_name == n) return &o;
    }
    return nullptr;
  }

  // Builder returned by add_options() lets boost-style chaining work:
  //   group.add_options()
  //     ("foo,f", value<string>(), "description")
  //     ("flag",                    "description")
  //     ;
  struct Builder {
    OptionsDescription* g;
    Builder& operator() (const char* name, ValueSpec* spec, const char* desc);
    Builder& operator() (const char* name, const char* desc);
  };
  Builder add_options() { return Builder{this}; }
};

// ---- Variables map ----

struct Variable {
  bool present = false;
  vector<string> raw;
  bool has_default = false;
  string default_str;
  ValueSpec::Kind kind = ValueSpec::Flag;

  template<typename T> T as() const;
};

template<> string Variable::as<string>() const;
template<> int Variable::as<int>() const;
template<> long Variable::as<long>() const;
template<> unsigned long Variable::as<unsigned long>() const;
template<> double Variable::as<double>() const;
template<> vector<string> Variable::as<vector<string>>() const;

struct VariablesMap {
  map<string, Variable> vars;

  size_t count (const string& name) const {
    auto it = vars.find (name);
    return (it != vars.end() && it->second.present) ? 1 : 0;
  }
  const Variable& at (const string& name) const {
    auto it = vars.find (name);
    if (it == vars.end()) throw std::runtime_error ("argparse: unknown option '" + name + "'");
    return it->second;
  }
};

// ---- Parsing ----

struct ParsedOptions {
  vector<string> unrecognized;  // in argv order; preserves --flag and following positionals
};

// Parse argv against `parseOpts`. Unknown long-options and positionals
// land in `parsed.unrecognized` (matching boost's `allow_unregistered` +
// `collect_unrecognized(... include_positional)` behaviour).
//
// Throws std::runtime_error on parse errors (missing required value,
// unrecognized short option, etc.).
void parse (int argc, char** argv,
            const OptionsDescription& parseOpts,
            VariablesMap& vm,
            ParsedOptions& parsed);

// ---- Help text ----

std::ostream& operator<< (std::ostream& os, const OptionsDescription& g);

// ---- boost-style lowercase aliases (so callers using `namespace po =`
// can keep boost-flavoured spellings: po::options_description, etc.) ----

using option_description  = OptDesc;
using options_description = OptionsDescription;
using variables_map       = VariablesMap;
using parsed_options      = ParsedOptions;

}  // namespace argparse
}  // namespace MachineBoss

#endif  // ARGPARSE_INCLUDED
