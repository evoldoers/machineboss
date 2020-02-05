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
  return Machine::null();
}
