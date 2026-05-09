#include "argparse.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

namespace MachineBoss {
namespace argparse {

// ---- ValueSpec slab ----------------------------------------------------
//
// boss.cpp uses the boost-style chaining `value<int>()->default_value(2)`,
// which requires a stable pointer the caller can dereference and that
// outlives both the call and the OptionsDescription that borrows it.
// We allocate ValueSpecs from a process-lifetime slab. The CLI tool is
// short-lived so we just leak — all reachable from this static vector.

static std::vector<std::unique_ptr<ValueSpec>>& slab() {
  static std::vector<std::unique_ptr<ValueSpec>> s;
  return s;
}

static ValueSpec* fresh (ValueSpec::Kind k) {
  slab().emplace_back (new ValueSpec);
  ValueSpec* v = slab().back().get();
  v->kind = k;
  return v;
}

ValueSpec* ValueSpec::default_value (const string& v) {
  default_str = v;
  has_default = true;
  return this;
}
ValueSpec* ValueSpec::default_value (int v) {
  std::ostringstream o; o << v;
  return default_value (o.str());
}

template<> ValueSpec* value<string>() { return fresh (ValueSpec::Str); }
template<> ValueSpec* value<int>()    { return fresh (ValueSpec::Int); }
template<> ValueSpec* value<long>()   { return fresh (ValueSpec::Int); }
template<> ValueSpec* value<unsigned long>() { return fresh (ValueSpec::Int); }
template<> ValueSpec* value<double>() { return fresh (ValueSpec::Str); }
template<> ValueSpec* value<vector<string>>() { return fresh (ValueSpec::VecStr); }

// ---- Builder -----------------------------------------------------------
//
// "name,s" — split into long_name and (optional) short_name.
static void split_name (const char* spec, string& longName, string& shortName) {
  string s = spec;
  auto comma = s.find (',');
  if (comma == string::npos) { longName = s; shortName.clear(); return; }
  longName = s.substr (0, comma);
  shortName = s.substr (comma + 1);
}

OptionsDescription::Builder& OptionsDescription::Builder::operator() (
    const char* name, ValueSpec* spec, const char* desc) {
  OptDesc o;
  split_name (name, o._long_name, o._short_name);
  o.spec = spec;
  o._description = desc ? desc : "";
  g->opts.push_back (std::move (o));
  return *this;
}

OptionsDescription::Builder& OptionsDescription::Builder::operator() (
    const char* name, const char* desc) {
  OptDesc o;
  split_name (name, o._long_name, o._short_name);
  o.spec = nullptr;
  o._description = desc ? desc : "";
  g->opts.push_back (std::move (o));
  return *this;
}

// ---- Variable::as<T>() -------------------------------------------------

template<> string Variable::as<string>() const {
  if (!raw.empty()) return raw.back();
  if (has_default)  return default_str;
  throw std::runtime_error ("argparse: option has no value or default");
}

template<> int Variable::as<int>() const {
  const string s = as<string>();
  return std::atoi (s.c_str());
}

template<> long Variable::as<long>() const {
  const string s = as<string>();
  return std::atol (s.c_str());
}

template<> unsigned long Variable::as<unsigned long>() const {
  const string s = as<string>();
  return std::strtoul (s.c_str(), nullptr, 10);
}

template<> double Variable::as<double>() const {
  const string s = as<string>();
  return std::atof (s.c_str());
}

template<> vector<string> Variable::as<vector<string>>() const {
  if (!raw.empty()) return raw;
  if (has_default)  return { default_str };
  return {};
}

// ---- Parsing -----------------------------------------------------------

namespace {

const OptDesc* find_long (const OptionsDescription& g, const string& name) {
  for (const auto& o : g.opts) if (o._long_name == name) return &o;
  return nullptr;
}
const OptDesc* find_short (const OptionsDescription& g, const string& name) {
  for (const auto& o : g.opts) if (!o._short_name.empty() && o._short_name == name) return &o;
  return nullptr;
}

bool wants_value (const OptDesc& o) {
  return o.spec && o.spec->kind != ValueSpec::Flag;
}

void store_value (Variable& v, const string& s) {
  v.raw.push_back (s);
  v.present = true;
}

void mark_present (Variable& v) {
  v.present = true;
}

}  // anon

void parse (int argc, char** argv,
            const OptionsDescription& parseOpts,
            VariablesMap& vm,
            ParsedOptions& parsed) {
  // Pre-create entries in vm for every known option so vm.at(name) works
  // even when the option wasn't on the command line. Default values are
  // copied in here too.
  for (const auto& o : parseOpts.opts) {
    Variable& v = vm.vars[o._long_name];
    v.kind = o.spec ? o.spec->kind : ValueSpec::Flag;
    if (o.spec && o.spec->has_default) {
      v.has_default = true;
      v.default_str = o.spec->default_str;
    }
  }

  for (int i = 1; i < argc; ++i) {
    string tok = argv[i];

    // "--" separates options from positionals.
    if (tok == "--") {
      for (++i; i < argc; ++i) parsed.unrecognized.emplace_back (argv[i]);
      break;
    }

    // long options: --name or --name=value
    if (tok.size() > 2 && tok[0] == '-' && tok[1] == '-') {
      string name, val;
      bool have_eq = false;
      auto eq = tok.find ('=', 2);
      if (eq != string::npos) {
        name = tok.substr (2, eq - 2);
        val = tok.substr (eq + 1);
        have_eq = true;
      } else {
        name = tok.substr (2);
      }

      const OptDesc* o = find_long (parseOpts, name);
      if (!o) {
        // unknown long option: keep raw in order (boost's collect_unrecognized
        // includes the original token verbatim, including any "=value")
        parsed.unrecognized.push_back (tok);
        continue;
      }

      Variable& v = vm.vars[o->_long_name];
      v.kind = o->spec ? o->spec->kind : ValueSpec::Flag;

      if (wants_value (*o)) {
        if (have_eq) {
          store_value (v, val);
        } else {
          if (i + 1 >= argc)
            throw std::runtime_error ("argparse: missing value for --" + name);
          store_value (v, argv[++i]);
        }
      } else {
        if (have_eq)
          throw std::runtime_error ("argparse: --" + name + " does not take a value");
        mark_present (v);
      }
      continue;
    }

    // short options: -x or -x value (no -xvalue support)
    if (tok.size() == 2 && tok[0] == '-' && tok[1] != '-') {
      const string sname (1, tok[1]);
      const OptDesc* o = find_short (parseOpts, sname);
      if (!o) {
        parsed.unrecognized.push_back (tok);
        continue;
      }
      Variable& v = vm.vars[o->_long_name];
      v.kind = o->spec ? o->spec->kind : ValueSpec::Flag;
      if (wants_value (*o)) {
        if (i + 1 >= argc)
          throw std::runtime_error ("argparse: missing value for -" + sname);
        store_value (v, argv[++i]);
      } else {
        mark_present (v);
      }
      continue;
    }

    // Anything else (positional, single '-', long-with-attached-arg-on-short, etc.)
    // is unrecognized.
    parsed.unrecognized.push_back (tok);
  }
}

// ---- Help text ---------------------------------------------------------

namespace {

string format_invocation (const OptDesc& o) {
  std::ostringstream os;
  os << "  ";
  if (!o._short_name.empty())
    os << "-" << o._short_name << " [ --" << o._long_name << " ]";
  else
    os << "    --" << o._long_name;
  if (o.spec && o.spec->kind != ValueSpec::Flag) {
    os << " arg";
    if (o.spec->has_default)
      os << " (=" << o.spec->default_str << ")";
  }
  return os.str();
}

constexpr size_t COL = 32;
constexpr size_t WIDTH = 80;

void word_wrap (std::ostream& os, const string& invoc, const string& desc) {
  os << invoc;
  // Pad to column COL; if invoc is already past it, start description on a new line.
  if (invoc.size() < COL) {
    os << string (COL - invoc.size(), ' ');
  } else {
    os << "\n" << string (COL, ' ');
  }
  // Word-wrap description into the remainder.
  const size_t avail = (WIDTH > COL) ? (WIDTH - COL) : 40;
  size_t col = 0;
  std::istringstream iss (desc);
  string word;
  bool first = true;
  while (iss >> word) {
    if (!first && col + 1 + word.size() > avail) {
      os << "\n" << string (COL, ' ');
      col = 0;
      first = true;
    }
    if (!first) { os << ' '; ++col; }
    os << word;
    col += word.size();
    first = false;
  }
}

}  // anon

std::ostream& operator<< (std::ostream& os, const OptionsDescription& g) {
  if (!g.title.empty()) {
    os << g.title << ":\n";
  }
  for (const auto& o : g.opts) {
    word_wrap (os, format_invocation (o), o._description);
    os << "\n";
  }
  return os;
}

}  // namespace argparse
}  // namespace MachineBoss
