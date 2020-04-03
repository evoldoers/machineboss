#include "parsers.h"
#include "../ext/cpp-peglib/peglib.h"

using namespace peg;
using namespace MachineBoss;

// Regular expression parser

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
  auto grammar =
#include "grammars/regex.h"
    ;
  
  parser parser;

  parser.log = [](size_t line, size_t col, const string& msg) {
    cerr << line << ":" << col << ": " << msg << "\n";
  };

  auto ok = parser.load_grammar(grammar);
  assert(ok);

  const string w = white, nw = nonwhite, alph = alphabet();
  const auto alphVec = stringToSymbols (alph);
  const Machine dotStar = Machine::wildRecognizer (alphVec);

  auto quantify = [] (const Machine& m, const SemanticValues& sv) {
    auto min_max = any_cast<pair<int,int>> (sv[1]);
    Machine qm;
    if (min_max.first == -1)  // *
      qm = Machine::kleeneStar (m);
    else if (min_max.first == -2)  // +
      qm = Machine::kleenePlus (m);
    else {
      qm = Machine::null();
      for (size_t n = min_max.first; n < min_max.second; ++n)
	qm = Machine::zeroOrOne (Machine::concatenate (m, qm));
      for (size_t n = 0; n < min_max.first; ++n)
	qm = Machine::concatenate (m, qm);
    }
    return qm;
  };
  
  parser["REGEX"] = [&](const SemanticValues& sv) {
    // cerr << "REGEX " << sv.str() << endl;
    auto carets = any_cast<int> (sv[0]);
    Machine m = any_cast<Machine> (sv[1]);
    auto dollars = any_cast<int> (sv[2]);
    if (!carets)
      m = Machine::concatenate (dotStar, m);
    if (dollars) {
      if (dollars > 1)
	m = Machine::concatenate (m, Machine::recognizer (vguard<InputSymbol> (dollars - 1, InputSymbol("$"))));
    } else
      m = Machine::concatenate (m, dotStar);
    return m.eliminateRedundantStates().stripNames();
  };

  parser["NONEMPTY_REGEX_BODY"] = [](const SemanticValues& sv) {
    // cerr << "NONEMPTY_REGEX_BODY " << sv.str() << endl;
    return Machine::concatenate (any_cast<Machine> (sv[0]), any_cast<Machine> (sv[1]));
  };

  parser["REGEX_BODY"] = [](const SemanticValues& sv) {
    // cerr << "REGEX_BODY " << sv.str() << endl;
    return sv.choice() ? Machine::null() : any_cast<Machine> (sv[0]);
  };

  parser["BEGIN_ANCHOR"] =
    parser["END_ANCHOR"] = [](const SemanticValues& sv) {
    return (int) sv.length();
  };

  parser["DOLLAR"] = [](const SemanticValues& sv) {
    return Machine::recognizer (vguard<InputSymbol> (1, InputSymbol("$")));
  };

  parser["QUANT_SYMBOLS"] = [](const SemanticValues& sv) {
    // cerr << "QUANT_SYMBOLS " << sv.str() << endl;
    Machine l = any_cast<Machine> (sv[0]);
    return sv.choice() ? l : Machine::concatenate (l, any_cast<Machine> (sv[1]));
  };

  parser["QUANT_SYMBOL"] = [&](const SemanticValues& sv) {
    // cerr << "QUANT_SYMBOL " << sv.str() << endl;
    Machine m = any_cast<Machine> (sv[0]);
    if (sv.choice() == 0)
      m = quantify (m, sv);
    return m;
  };

  parser["SYMBOL"] = [](const SemanticValues& sv) {
    return any_cast<Machine> (sv[0]);
  };

  parser["TOP_SYMBOL"] = [](const SemanticValues& sv) {
    // cerr << "TOP_SYMBOL " << sv.str() << endl;
    auto m = any_cast<Machine> (sv[0]);
    return m;
  };

  parser["MACHINE_SYMBOL"] = [](const SemanticValues& sv) {
    // cerr << "MACHINE_SYMBOL " << sv.str() << endl;
    auto m = any_cast<Machine> (sv[0]);
    return m;
  };

  parser["MACHINE_CHAR"] = [](const SemanticValues& sv) {
    // cerr << "MACHINE_CHAR " << sv.str() << endl;
    const char c = any_cast<char> (sv[0]);
    Machine m = Machine::wildSingleRecognizer (vguard<InputSymbol> (1, InputSymbol (1, c)));
    return m;
  };

  parser["QUANTIFIER"] = [](const SemanticValues& sv) {
    // cerr << "QUANTIFIER " << sv.choice() << endl;
    pair<int,int> result;
    switch (sv.choice()) {
    case 0: result = make_pair ((int) -1, (int) -1); break;
    case 1: result = make_pair ((int) -2, (int) -2); break;
    case 2: {
      const int reps = any_cast<int> (sv[0]);
      result = make_pair (reps, reps);
      break;
    }
    case 3:
      result = make_pair (any_cast<int> (sv[0]), any_cast<int> (sv[1]));
      break;
    default: break;
    }
    return result;
  };

  parser["INTEGER"] = [](const SemanticValues& sv) {
    // cerr << "INTEGER " << sv.str() << endl;
    return stoi(sv.str(), nullptr, 10);
  };

  parser["CHAR_CLASS"] =
    parser["IMPLICIT_CHAR_CLASS"] = [](const SemanticValues& sv) {
    // cerr << "CHAR_CLASS or IMPLICIT_CHAR_CLASS " << sv.str() << endl;
    const string s = any_cast<string> (sv[0]);
    return Machine::wildSingleRecognizer (RegexParser::stringToSymbols (s));
  };

  parser["NEGATED_CHAR_CLASS"] = [&](const SemanticValues& sv) {
    // cerr << "NEGATED_CHAR_CLASS " << sv.str() << endl;
    const auto str = RegexParser::stringToSymbols (any_cast<string> (sv[0]));
    const set<InputSymbol> negated (str.begin(), str.end());
    vguard<InputSymbol> nc;
    nc.reserve (alphVec.size());
    for (const auto& sym: alphVec)
      if (negated.find(sym) == negated.end())
	nc.push_back (sym);
    return Machine::wildSingleRecognizer (nc);
  };

  parser["PRESET_CHAR_CLASS"] = [](const SemanticValues& sv) {
    // cerr << "PRESET_CHAR_CLASS " << sv.str() << endl;
    return any_cast<string> (sv[0]);
  };

  parser["CHARS"] = [](const SemanticValues& sv) {
    // cerr << "CHARS " << sv.str() << endl;
    string s = any_cast<string> (sv[0]);
    if (sv.choice() == 0)
      s.append (any_cast<string> (sv[1]));
    return s;
  };

  parser["CHAR"] = [](const SemanticValues& sv) {
    // cerr << "CHAR " << sv.str() << endl;
    return any_cast<string> (sv[0]);
  };

  parser["DIGIT_CHAR"] = [](const SemanticValues& sv) {
    // cerr << "DIGIT_CHAR " << sv.str() << endl;
    return string ("0123456789");
  };

  parser["WHITE_CHAR"] = [&](const SemanticValues& sv) {
    // cerr << "WHITE_CHAR " << sv.str() << endl;
    return w;
  };

  parser["NONWHITE_CHAR"] = [&](const SemanticValues& sv) {
    // cerr << "NONWHITE_CHAR " << sv.str() << endl;
    return nw;
  };

  parser["WILD_CHAR"] = [&](const SemanticValues& sv) {
    return alph;
  };

  parser["CHAR_RANGE"] = [](const SemanticValues& sv) {
    // cerr << "CHAR_RANGE " << sv.str() << endl;
    const string b = any_cast<string> (sv[0]), e = any_cast<string> (sv[1]);
    const char bc = b.at(0), ec = e.at(0);
    if (ec < bc)
      throw peg::parse_error("illegal range in character class");
    string s;
    s.reserve (ec + 1 - bc);
    for (char c = bc; c <= ec; ++c)
      s.push_back (c);
    return s;
  };

  parser["CHAR_INSIDE_CLASS"] = [](const SemanticValues& sv) {
    // cerr << "CHAR_INSIDE_CLASS " << sv.str() << endl;
    const char c = any_cast<char> (sv[0]);
    return string (1, c);
  };

  parser["ESCAPED_OR_SINGLE_CHAR"] = [](const SemanticValues& sv) {
    // cerr << "ESCAPED_OR_SINGLE_CHAR " << sv.str() << endl;
    return any_cast<char> (sv[0]);
  };

  parser["ESCAPED_CHAR"] = [](const SemanticValues& sv) {
    // cerr << "ESCAPED_CHAR " << sv.str() << endl;
    return any_cast<char> (sv[0]);
  };

  parser["SINGLE_CHAR"] = [](const SemanticValues& sv) {
    // cerr << "SINGLE_CHAR " << sv.str() << endl;
    return (char) sv.str()[0];
  };

  parser["ESCAPE_CHAR"] = [](const SemanticValues& sv) {
    // cerr << "ESCAPE_CHAR " << sv.str() << endl;
    const char c = sv.str()[0];
    switch (c) {
    case 'n': return '\n';
    case 'r': return '\r';
    case 't': return '\t';
    default: break;
    }
    return c;
  };

  parser["OCTAL"] = [](const SemanticValues& sv) {
    // cerr << "OCTAL " << sv.str() << endl;
    return (char) stoi (sv.str(), nullptr, 8);
  };

  parser["HEX"] = [](const SemanticValues& sv) {
    // cerr << "HEX " << sv.str() << endl;
    return (char) stoi (sv.str(), nullptr, 16);
  };

  parser["ALTERNATION"] = [](const SemanticValues& sv) {
    // cerr << "ALTERNATION " << sv.str() << endl;
    return any_cast<Machine> (sv[0]);
  };

  parser["ALT_OPTIONS"] = [](const SemanticValues& sv) {
    // cerr << "ALT_OPTIONS " << sv.str() << endl;
    auto m = any_cast<Machine> (sv[0]);
    return sv.choice() == 0 ? Machine::takeUnion (m, any_cast<Machine> (sv[1])) : m;
  };

  parser["ALT_SYMBOLS"] = [](const SemanticValues& sv) {
    // cerr << "ALT_SYMBOLS " << sv.str() << endl;
    return sv.choice() ? Machine::null() : Machine::concatenate (any_cast<Machine> (sv[0]),
								 any_cast<Machine> (sv[1]));
  };

  parser["ALT_SYMBOL"] = [](const SemanticValues& sv) {
    // cerr << "ALT_SYMBOL " << sv.str() << endl;
    return any_cast<Machine> (sv[0]);
  };

  parser["QUANT_ALT_SYMBOL"] = [&](const SemanticValues& sv) {
    // cerr << "QUANT_ALT_SYMBOL " << sv.str() << endl;
    auto m = any_cast<Machine> (sv[0]);
    if (sv.choice() == 0)
      m = quantify (m, sv);
    return m;
  };

  Machine m;
  parser.parse (str.c_str(), m);
  return m;
}


// Weight expression parser
struct ExprParser {
  parser parser;
  ExprParser();
};
ExprParser exprParser;
ExprParser::ExprParser() {
  auto grammar =
#include "grammars/expr.h"
    ;

  parser.log = [](size_t line, size_t col, const string& msg) {
    cerr << line << ":" << col << ": " << msg << "\n";
  };

  auto ok = parser.load_grammar(grammar);
  assert(ok);

  //      if (sv.choice() == 0)
  //      s.append (any_cast<string> (sv[1]));

  parser["Term"] = [&](const SemanticValues& sv) {
    cerr << "Term " << sv.str() << endl;
    WeightExpr w = any_cast<WeightExpr> (sv[0]);
    for (int i = 1; i < sv.size(); ++i)
      w = WeightAlgebra::add (w, any_cast<WeightExpr> (sv[i]));
    return w;
  };

  parser["Add"] = [&](const SemanticValues& sv) {
    cerr << "Add " << sv.str() << endl;
    return any_cast<WeightExpr> (sv[0]);
  };

  parser["Sub"] = [&](const SemanticValues& sv) {
    cerr << "Sub " << sv.str() << endl;
    return WeightAlgebra::minus (any_cast<WeightExpr> (sv[0]));
  };

  parser["Factor"] = [&](const SemanticValues& sv) {
    cerr << "Factor " << sv.str() << endl;
    WeightExpr w = any_cast<WeightExpr> (sv[0]);
    for (int i = 1; i < sv.size(); ++i)
      w = WeightAlgebra::multiply (w, any_cast<WeightExpr> (sv[i]));
    return w;
  };

  parser["Mul"] = [&](const SemanticValues& sv) {
    cerr << "Mul " << sv.str() << endl;
    return any_cast<WeightExpr> (sv[0]);
  };

  parser["Div"] = [&](const SemanticValues& sv) {
    cerr << "Div " << sv.str() << endl;
    return WeightAlgebra::reciprocal (any_cast<WeightExpr> (sv[0]));
  };

  parser["Power"] = [&](const SemanticValues& sv) {
    cerr << "Power " << sv.str() << endl;
    return WeightAlgebra::power (any_cast<WeightExpr> (sv[0]),
				 any_cast<WeightExpr> (sv[1]));
  };

  parser["Primary"] = [&](const SemanticValues& sv) {
    cerr << "Primary " << sv.str() << endl;
    return any_cast<WeightExpr> (sv[0]);
  };

  parser["Parens"] = [&](const SemanticValues& sv) {
    cerr << "Parens " << sv.str() << endl;
    return any_cast<WeightExpr> (sv[0]);
  };

  parser["Function"] = [&](const SemanticValues& sv) {
    cerr << "Function " << sv.str() << endl;
    return any_cast<WeightExpr> (sv[0]);
  };

  parser["Exp"] = [&](const SemanticValues& sv) {
    cerr << "Exp " << sv.str() << endl;
    return WeightAlgebra::expOf (any_cast<WeightExpr> (sv[0]));
  };

  parser["Log"] = [&](const SemanticValues& sv) {
    cerr << "Log " << sv.str() << endl;
    return WeightAlgebra::logOf (any_cast<WeightExpr> (sv[0]));
  };

  parser["NegateProb"] = [&](const SemanticValues& sv) {
    cerr << "NegateProb " << sv.str() << endl;
    return WeightAlgebra::negate (any_cast<WeightExpr> (sv[0]));
  };

  parser["Negative"] = [&](const SemanticValues& sv) {
    cerr << "Negative " << sv.str() << endl;
    return WeightAlgebra::minus (any_cast<WeightExpr> (sv[0]));
  };

  parser["Sign"] = [&](const SemanticValues& sv) {
    cerr << "Sign " << sv.str() << endl;
    return WeightAlgebra::one();
  };

  parser["Number"] = [&](const SemanticValues& sv) {
    cerr << "Number " << sv.str() << endl;
    return WeightAlgebra::doubleConstant (stof (sv.str()));
  };

  parser["IntOrFloat"] = [&](const SemanticValues& sv) {
    cerr << "IntOrFloat " << sv.str() << endl;
    return WeightAlgebra::one();
  };

  parser["Integer"] = [&](const SemanticValues& sv) {
    cerr << "Integer " << sv.str() << endl;
    return WeightAlgebra::one();
  };

  parser["Float"] = [&](const SemanticValues& sv) {
    cerr << "Float " << sv.str() << endl;
    return WeightAlgebra::one();
  };

  parser["Scientific"] = [&](const SemanticValues& sv) {
    cerr << "Scientific " << sv.str() << endl;
    return WeightAlgebra::one();
  };

  parser["Variable"] = [&](const SemanticValues& sv) {
    cerr << "Variable " << sv.str() << endl;
    return WeightAlgebra::param (sv.token(1));
  };

  parser["identifier"] = [&](const SemanticValues& sv) {
    cerr << "identifier " << sv.str() << endl;
    return WeightAlgebra::one();
  };

  parser["~_"] = [&](const SemanticValues& sv) {
    cerr << "~_ " << sv.str() << endl;
    return WeightAlgebra::one();
  };
}
  
WeightExpr MachineBoss::parseWeightExpr (const string& str) {
  WeightExpr w = WeightAlgebra::one();
  exprParser.parser.parse (str.c_str(), w);
  return w;
}
