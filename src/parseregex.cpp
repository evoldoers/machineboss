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

REGEX <- BEGIN_ANCHOR? REGEX_TAIL
REGEX_TAIL <- NONEMPTY_REGEX_TAIL /
NONEMPTY_REGEX_TAIL <- QUANT_DOLLAR NONEMPTY_REGEX_TAIL / QUANT_SYMBOL+ REGEX_TAIL / END_ANCHOR
BEGIN_ANCHOR <- '^'
END_ANCHOR <- '$'
QUANT_DOLLAR <- '$' QUANTIFIER / '$'
QUANT_SYMBOL <- SYMBOL QUANTIFIER / SYMBOL
SYMBOL <- NEGATED_CHAR_CLASS / CHAR_CLASS / IMPLICIT_CHAR_CLASS / ALTERNATION / ESCAPED_CHAR / (!'$' .)
QUANTIFIER <- '*' / '+' / ('{' INTEGER '}') / ('{' INTEGER ',' INTEGER '}')
INTEGER <- [1-9] [0-9]*
CHAR_CLASS <- '[' CHAR+ ']'
NEGATED_CHAR_CLASS <- '[' '^' CHAR+ ']'
IMPLICIT_CHAR_CLASS <- PRESET_CHAR_CLASS
CHAR <- PRESET_CHAR_CLASS / CHAR_RANGE / SINGLE_CHAR
PRESET_CHAR_CLASS <- DIGIT_CHAR / WHITE_CHAR / NONWHITE_CHAR / WILD_CHAR
DIGIT_CHAR <- '\\d'
WHITE_CHAR <- '\\s'
NONWHITE_CHAR <- '\\S'
WILD_CHAR <- '.'
CHAR_RANGE <- SINGLE_CHAR '-' SINGLE_CHAR
SINGLE_CHAR <- ESCAPED_CHAR / (!']' .)
ESCAPED_CHAR <- '\\' [nrt|(\[\]\\]
/ '\\' [0-2][0-7][0-7]
/ '\\' [0-7][0-7]
ALTERNATION <- '(' ALT_OPTION ('|' ALT_OPTION)+ ')'
ALT_OPTION <- ALT_SYMBOL*
ALT_SYMBOL <- !'|' !')' SYMBOL

  */

  return Machine::null();
}
