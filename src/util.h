#ifndef UTIL_INCLUDED
#define UTIL_INCLUDED

#include <numeric>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <algorithm>
#include <functional>
#include <cassert>
#include <mutex>
#include <sys/stat.h>

/* uncomment to enable NaN checks */
#define NAN_DEBUG

/* Errors, warnings, assertions.
   Fail(...) and Require(...) are quieter versions of Abort(...) and Assert(...)
   that do not print a stack trace or throw an exception,
   but merely call exit().
   Test(...) does not exit or throw an exception,
   just prints a warning and returns false if the assertion fails.
   Desire(...) is a macro wrapper for Test(...)
   that returns false from the calling function if the test fails.
*/
void Abort(const char* error, ...);
void Warn(const char* warning, ...);
void Fail(const char* error, ...);
#define Test(assertion,...) ((assertion) ? true : (Warn(__VA_ARGS__), false))
#define Assert(assertion,...) do { if (!(assertion)) Abort("Assertion Failed: " __VA_ARGS__); } while (0)
#define Require(assertion,...) do { if (!(assertion)) Fail(__VA_ARGS__); } while (0)
#define Desire(...) do { if (!Test(__VA_ARGS__)) return false; } while (0)

void CheckGsl (int gslErrorCode);

/* singular or plural? */
std::string plural (long n, const char* singular);
std::string plural (long n, const char* singular, const char* plural);

/* stringify */
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

/* join */
template<class Container>
std::string join (const Container& c, const char* sep = " ") {
  std::string j;
  for (const auto& s : c) {
    if (!j.empty())
      j += sep;
    j += s;
  }
  return j;
}

/* to_string_join */
template<class Container>
std::string to_string_join (const Container& c, const char* sep = " ") {
  std::ostringstream j;
  int n = 0;
  for (const auto& s : c) {
    if (n++ > 0)
      j << sep;
    j << s;
  }
  return j.str();
}

/* transform_vector */
template<class S,class Container>
std::vector<S> transform_container (const Container& v, S (op) (typename Container::value_type const&)) {
  std::vector<S> result;
  std::transform (v.begin(), v.end(), back_inserter(result), op);
  return result;
}

template<class S,class Container>
std::vector<S> transform_container (const Container& v, S (op) (typename Container::value_type)) {
  std::vector<S> result;
  std::transform (v.begin(), v.end(), back_inserter(result), op);
  return result;
}

/* split */
std::vector<std::string> split (const std::string& s, const char* splitChars = " \t\n");

/* toupper */
std::string toupper (const std::string& s);

/* escaping a string
   http://stackoverflow.com/questions/2417588/escaping-a-c-string
 */
template<class OutIter>
OutIter write_quoted_escaped(std::string const& s, OutIter out) {
  *out++ = '"';
  for (std::string::const_iterator i = s.begin(), end = s.end(); i != end; ++i) {
    unsigned char c = *i;
    if (' ' <= c and c <= '~' and c != '\\' and c != '"') {
      *out++ = c;
    }
    else {
      *out++ = '\\';
      switch(c) {
      case '"':  *out++ = '"';  break;
      case '\\': *out++ = '\\'; break;
      case '\t': *out++ = 't';  break;
      case '\r': *out++ = 'r';  break;
      case '\n': *out++ = 'n';  break;
      default:
        char const* const hexdig = "0123456789ABCDEF";
        *out++ = 'x';
        *out++ = hexdig[c >> 4];
        *out++ = hexdig[c & 0xF];
      }
    }
  }
  *out++ = '"';
  return out;
}

/* random_double */
template<class Generator>
double random_double (Generator& generator) {
  return generator() / (((double) std::numeric_limits<typename Generator::result_type>::max()) + 1);
}

/* extract_keys */
template<typename TK, typename TV>
std::vector<TK> extract_keys(std::map<TK, TV> const& input_map) {
  std::vector<TK> retval;
  for (auto const& element : input_map) {
    retval.push_back(element.first);
  }
  return retval;
}

/* extract_values */
template<typename TK, typename TV>
std::vector<TV> extract_values(std::map<TK, TV> const& input_map) {
  std::vector<TV> retval;
  for (auto const& element : input_map) {
    retval.push_back(element.second);
  }
  return retval;
}    

/* random_element */
template<class Iterator,class Generator>
Iterator random_element (Iterator begin, Iterator end, Generator generator)
{
    const size_t n = std::distance(begin, end);
    std::uniform_int_distribution<int> distribution (0, n - 1);
    const size_t k = distribution (generator);
    std::advance(begin, k);
    return begin;
}

template<class Container,class Generator>
typename Container::const_reference random_element (const Container& container, Generator generator)
{
  return *(random_element (container.begin(), container.end(), generator));
}

template<class Container,class Generator>
typename Container::reference random_element (Container& container, Generator generator)
{
  return *(random_element (container.begin(), container.end(), generator));
}

/* random_index */
template<class T,class Generator>
size_t random_index (const std::vector<T>& weights, Generator& generator) {
  T norm = 0;
  for (const T& w : weights) {
    Assert (w >= 0, "Negative weights in random_index");
    norm += w;
  }
  Assert (norm > 0, "Zero weights in random_index");
  T variate = random_double(generator) * norm;
  for (size_t n = 0; n < weights.size(); ++n)
    if ((variate -= weights[n]) <= 0)
      return n;
  return weights.size();
}

/* random_key */
template<class T,class Generator>
const T& random_key (const std::map<T,double>& weight, Generator& generator) {
  double norm = 0;
  for (const auto& kv : weight) {
    Assert (kv.second >= 0, "Negative weights in random_key");
    norm += kv.second;
  }
  Assert (norm > 0, "Zero weights in random_key");
  T variate = random_double(generator) * norm;
  for (const auto& kv : weight)
    if ((variate -= kv.second) <= 0)
      return kv.first;
  Abort ("random_key failed");
  return *(weight.begin()).first;
}

/* random_key_log */
template<class T,class Generator>
const T& random_key_log (const std::map<T,double>& logWeight, Generator& generator) {
  double norm = 0, logmax = -std::numeric_limits<double>::infinity();
  for (const auto& kv : logWeight)
    logmax = max (logmax, kv.second);
  for (const auto& kv : logWeight)
    norm += exp (kv.second - logmax);
  Assert (norm > 0, "Zero weights in random_key_log");
  double variate = random_double(generator) * norm;
  for (const auto& kv : logWeight)
    if ((variate -= exp (kv.second - logmax)) <= 0)
      return kv.first;
  Abort ("random_key_log failed");
  const auto& iter = logWeight.begin();
  return (*iter).first;
}

/* index sort
   http://stackoverflow.com/questions/10580982/c-sort-keeping-track-of-indices
 */
template <typename S, typename T>
void sortIndices (std::vector<S>& indices, std::vector<T> const& values) {
    std::sort(
        begin(indices), end(indices),
        [&](S a, S b) { return values[a] < values[b]; }
    );
}

template <typename T>
std::vector<size_t> orderedIndices (std::vector<T> const& values) {
    std::vector<size_t> indices(values.size());
    std::iota(begin(indices), end(indices), static_cast<size_t>(0));
    sortIndices (indices, values);
    return indices;
}

#endif /* UTIL_INCLUDED */
