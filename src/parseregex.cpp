#include "parseregex.h"
#include "../ext/cpp-peglib/peglib.h"

using namespace peg;

RegexParser::RegexParser()
  : white (" \t\n")
{
  const char cStart = '!', cEnd = '~';
  nonwhite.reserve (cEnd + 1 - cStart);
  for (char c = cStart; c <= cEnd; ++c)
    nonwhite.push_back (c);
}

string RegexParser::alphabet() const {
  string a = white;
  a.append (nonwhite);
  return a;
}

vguard<InputSymbol> RegexParser::stringToSymbols (const string& s) {
  return splitToChars (s);
}

Machine RegexParser::parse (const string& str) const {
  // TODO: WRITE ME

  auto grammar =
#include "grammars/regex.h"
    ;
  
  parser parser;

  parser.log = [](size_t line, size_t col, const string& msg) {
    cerr << line << ":" << col << ": " << msg << "\n";
  };

  auto ok = parser.load_grammar(grammar);
  assert(ok);

  const Machine dotStar = Machine::wildRecognizer (stringToSymbols (alphabet()));
  
  parser["REGEX"] = [&](const SemanticValues& sv) {
    Machine m = any_cast<Machine> (sv[1]);
    if (!sv.token(0).length())
      m = Machine::concatenate (dotStar, m);
    auto dollars = sv.token(2).length();
    if (dollars) {
      if (dollars > 1)
	m = Machine::concatenate (m, Machine::recognizer (vguard<InputSymbol> (dollars - 1, InputSymbol("$"))));
    } else
      m = Machine::concatenate (m, dotStar);
    return m.eliminateRedundantStates().stripNames();
  };

  parser["REGEX_BODY"] = [](const SemanticValues& sv) {
    return sv.choice() ? Machine::null() : any_cast<Machine> (sv[0]);
  };

  parser["REGEX_BODY"] = [](const SemanticValues& sv) {
    return sv.choice() ? Machine::null() : any_cast<Machine> (sv[0]);
  };

  parser["NONEMPTY_REGEX_BODY"] = [](const SemanticValues& sv) {
    return Machine::concatenate (any_cast<Machine> (sv[0]), any_cast<Machine> (sv[1]));
  };

  parser["END_ANCHOR"] = [](const SemanticValues& sv) {
    return Machine::recognizer (vguard<InputSymbol> (1, InputSymbol("$")));
  };

  parser["QUANT_SYMBOLS"] = [](const SemanticValues& sv) {
    Machine l = any_cast<Machine> (sv[0]);
    return sv.choice() ? l : Machine::concatenate (l, any_cast<Machine> (sv[1]));
  };

  parser["QUANT_SYMBOL"] = [](const SemanticValues& sv) {
    Machine m = any_cast<Machine> (sv[0]);
    if (sv.choice() < 2) {
      const string qstr = sv.token(1);
      if (qstr[0] == '*')
	m = Machine::kleeneStar (m);
      else if (qstr[1] == '+')
	m = Machine::kleenePlus (m);
      else {
	auto min_max = any_cast<pair<int,int>> (sv[1]);
	Machine reps = Machine::null();
	for (size_t n = min_max.first; n < min_max.second; ++n)
	  reps = Machine::zeroOrOne (Machine::concatenate (m, reps));
	for (size_t n = 0; n < min_max.first; ++n)
	  reps = Machine::concatenate (m, reps);
	m = reps;
      }
    }
    return m;
  };

  parser["SYMBOL"] = [](const SemanticValues& sv) {
    return any_cast<Machine> (sv[0]);
  };

  parser["LITERAL_CHAR"] = [](const SemanticValues& sv) {
    const char c = any_cast<char> (sv[0]);
    return Machine::recognizer (vguard<InputSymbol> (1, InputSymbol (1, c)));
  };
  
  parser["QUANTIFIER"] = [](const SemanticValues& sv) {
    if (sv.choice() == 2) {
      const int reps = any_cast<int> (sv[1]);
      return make_pair (reps, reps);
    } else if (sv.choice() == 3)
      return make_pair (any_cast<int> (sv[1]), any_cast<int> (sv[3]));
    return make_pair ((int) 0, (int) 0);
  };

  parser["INTEGER"] = [](const SemanticValues& sv) {
    return stoi(sv.token(), nullptr, 10);
  };
  
  Machine m;
  parser.parse (str.c_str(), m);
  return m;
}
