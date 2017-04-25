#ifndef JSONIO_INCLUDED
#define JSONIO_INCLUDED

#include <json.hpp>
#include <fstream>
#include <iostream>
#include "util.h"

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

  static void readFile (Base& obj, const string& filename) {
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

#endif /* JSONIO_INCLUDED */
