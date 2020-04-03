#ifndef PARSERS_INCLUDED
#define PARSERS_INCLUDED

#include "machine.h"

namespace MachineBoss {

struct RegexParser {
  string nonwhite, white;
  RegexParser();
  string alphabet() const;

  Machine parse (const string&) const;

  static vguard<InputSymbol> stringToSymbols (const string&);
};

WeightExpr parseWeightExpr (const string&);

}  // end namespace

#endif /* PARSEREGEX_INCLUDED */
