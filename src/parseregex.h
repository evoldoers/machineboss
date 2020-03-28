#ifndef PARSEREGEX_INCLUDED
#define PARSEREGEX_INCLUDED

#include "machine.h"

namespace MachineBoss {

struct RegexParser {
  string nonwhite, white;
  RegexParser();
  string alphabet() const;

  Machine parse (const string&) const;

  static vguard<InputSymbol> stringToSymbols (const string&);
};

}  // end namespace

#endif /* PARSEREGEX_INCLUDED */
