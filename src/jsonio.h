#ifndef JSONIO_INCLUDED
#define JSONIO_INCLUDED

#include <json.hpp>
#include <fstream>
#include <iostream>
#include "util.h"

namespace MachineBoss {

using namespace std;
  
// infinity-safe toString method for JSON output
inline string toInfinitySafeString (double x) {
  if (x == numeric_limits<double>::infinity())
    return string("\"Infinity\"");
  if (x == -numeric_limits<double>::infinity())
    return string("\"-Infinity\"");
  ostringstream out;
  out << x;
  return out.str();
}

// wrappers for readJson & writeJson methods
template<class Base>
struct JsonWriter {

  static void toFile (const Base& obj, const char* filename) {
    std::ofstream outfile (filename);
    if (!outfile)
      Fail ("Couldn't open file: %s", filename);
    obj.writeJson (outfile);
  }

  static void toFile (const Base& obj, const std::string& filename) {
    toFile (obj, filename.c_str());
  }

  static std::string toJsonString (const Base& obj) {
    std::ostringstream out;
    obj.writeJson (out);
    return out.str();
  }

  static nlohmann::json toJson (const Base& obj) {
    std::istringstream in (toJsonString (obj));
    nlohmann::json j;
    in >> j;
    return j;
  }
};

template<class Base>
struct JsonReader : Base {
  static void readJson (Base& obj, std::istream& in) {
    nlohmann::json j;
    in >> j;
    obj.readJson(j);
  }

  static Base fromJsonString (const std::string& str) {
    std::istringstream in (str);
    return fromJson (in);
  }

  static Base fromJson (const nlohmann::json& json) {
    Base obj;
    obj.readJson (json);
    return obj;
  }
  
  static Base fromJson (std::istream& in) {
    Base obj;
    readJson (obj, in);
    return obj;
  }
  
  static Base fromFile (const char* filename) {
    std::ifstream infile (filename);
    if (!infile)
      Fail ("File not found: %s", filename);
    return fromJson (infile);
  }

  static Base fromFile (const std::string& filename) {
    return fromFile (filename.c_str());
  }

  static Base fromFiles (const std::vector<std::string>& filenames) {
    JsonReader<Base> obj;
    readFiles (obj, filenames);
    return obj;
  }

  static void readFile (Base& obj, const std::string& filename) {
    std::ifstream infile (filename);
    if (!infile)
      Fail ("File not found: %s", filename.c_str());
    readJson (obj, infile);
  }

  static void readFiles (Base& obj, const std::vector<std::string>& filenames) {
    for (const auto& filename: filenames)
      readFile (obj, filename);
  }
};

template<class Base>
struct JsonLoader : JsonReader<Base>, JsonWriter<Base>
{ };

}  // end namespace

#endif /* JSONIO_INCLUDED */
