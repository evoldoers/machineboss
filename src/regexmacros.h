#ifndef REGEXMACROS_INCLUDED
#define REGEXMACROS_INCLUDED

#ifdef USE_BOOST
#include <boost/regex.hpp>
using namespace boost;
#else
#include <regex>
using namespace std;
#endif

// POSIX basic regular expressions are used for maximum compatibility,
// since g++ does not stably support ECMAScript regexes yet,
// so we resort to boost where clang is not available
// (e.g. Amazon EC2 AMI/yum, at time of writing: 8/25/2015).

// These macros use an ultra-minimal subset of POSIX basic syntax:
// character classes [], ranges a-z, repetition *, wildcards ., groups \\( \\)
// NB literal hyphens must appear at the start of a character class.

#define RE_CHAR_CLASS(STR) "[" STR "]"
#define RE_STAR(CLASS) CLASS "*"
#define RE_PLUS(CLASS) CLASS RE_STAR(CLASS)
#define RE_GROUP(EXPR) "\\(" EXPR "\\)"

#define RE_NUMERIC_RANGE "0-9"
#define RE_ALPHA_RANGE "A-Za-z"
#define RE_ALPHANUM_RANGE RE_ALPHA_RANGE RE_NUMERIC_RANGE
#define RE_NONWHITE_RANGE "!-~"

#define RE_NUMERIC_CHAR_CLASS RE_CHAR_CLASS(RE_NUMERIC_RANGE)
#define RE_VARNAME_CHAR_CLASS RE_CHAR_CLASS(RE_ALPHANUM_RANGE "_")
#define RE_DNS_CHAR_CLASS RE_CHAR_CLASS("-" RE_ALPHANUM_RANGE "\\.")
#define RE_FLOAT_CHAR_CLASS RE_CHAR_CLASS("-" RE_NUMERIC_RANGE "eE+\\.")
#define RE_NONWHITE_CHAR_CLASS RE_CHAR_CLASS("!-~")
#define RE_DOT RE_CHAR_CLASS(" -~")

#define RE_DOT_STAR RE_DOT "*"
#define RE_DOT_PLUS RE_PLUS(RE_DOT)

#define RE_NUMERIC_GROUP RE_GROUP(RE_PLUS(RE_NUMERIC_CHAR_CLASS))
#define RE_VARNAME_GROUP RE_GROUP(RE_PLUS(RE_VARNAME_CHAR_CLASS))
#define RE_DNS_GROUP RE_GROUP(RE_PLUS(RE_DNS_CHAR_CLASS))
#define RE_FLOAT_GROUP RE_GROUP(RE_PLUS(RE_FLOAT_CHAR_CLASS))
#define RE_DOT_GROUP RE_GROUP(RE_PLUS(RE_DOT))

#define RE_WHITE_CHARS " \t\n"
#define RE_WHITE_CHAR_CLASS RE_CHAR_CLASS(RE_WHITE_CHARS)
#define RE_WHITE_OR_EMPTY RE_STAR(RE_WHITE_CHAR_CLASS)
#define RE_WHITE_NONEMPTY RE_PLUS(RE_WHITE_CHAR_CLASS)

#endif /* REGEXMACROS_INCLUDED */
