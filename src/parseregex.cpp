#include "parseregex.h"
#include "../ext/cpp-peglib/peglib.h"

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

Machine RegexParser::parse (const string& str) const {
  // TODO: WRITE ME

  /* PEG grammar:
REGEX <- BEGIN_ANCHOR SYMBOL* END_ANCHOR
BEGIN_ANCHOR <- '^'?
END_ANCHOR <- '$'?
SYMBOL <- NEGATED_CHAR_CLASS / CHAR_CLASS / ALTERNATION / .
CHAR_CLASS <- '[' CHAR+ ']'
NEGATED_CHAR_CLASS <- '[' '^' CHAR+ ']'
CHAR <- DIGIT_CHAR / WHITE_CHAR / NONWHITE_CHAR / ANY_CHAR / ESCAPED_CHAR / (!']' .)
DIGIT_CHAR <- '\\' 'd'
WHITE_CHAR <- '\\' 's'
NONWHITE_CHAR <- '\\' 'S'
ANY_CHAR <- '.'
ESCAPED_CHAR <- '\\' .
ALTERNATION <- '(' ALT_OPTION ('|' ALT_OPTION)+ ')'
ALT_OPTION <- ALT_SYMBOL*
ALT_SYMBOL <- !'|' !')' SYMBOL
  */

  return Machine::null();
}
