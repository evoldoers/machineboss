#ifndef JSONUTIL_INCLUDED
#define JSONUTIL_INCLUDED

#include <map>
#include <string>
#include "vguard.h"
#include "gason.h"

using namespace std;

// Default size of receive buffer for sockets
#define RCVBUFSIZE 1024

// JSON object index
struct JsonMap {
  map<string,JsonValue*> m;
  JsonMap() { }
  JsonMap (const JsonValue& value);
  void initMap (const JsonValue& value);
  bool contains (const char* key) const;
  bool contains (const string& key) const;
  bool containsType (const char* key, JsonTag type) const;
  bool containsType (const string& key, JsonTag type) const;
  JsonValue& operator[] (const char* key) const;
  JsonValue& operator[] (const string& key) const;
  JsonValue& getType (const char* key, JsonTag type) const;
  JsonValue& getType (const string& key, JsonTag type) const;
  JsonMap getObject (const char* key) const;
  JsonMap getObject (const string& key) const;
  double getNumber (const char* key) const;
  double getNumber (const string& key) const;
  string getString (const char* key) const;
  string getString (const string& key) const;
  bool getBool (const char* key) const;
  bool getBool (const string& key) const;
};

// wrapper for JsonValue with parent string
class ParsedJson : public JsonMap {
private:
  ParsedJson (const ParsedJson&) = delete;
  ParsedJson& operator= (const ParsedJson&) = delete;
public:
  string str;
  char *buf, *endPtr;
  JsonValue value;
  JsonAllocator allocator;
  int status;
  ParsedJson (const string& s, bool parseOrDie = true);
  ParsedJson (istream& in, bool parseOrDie = true);
  ~ParsedJson();
  void parse (const string& s, bool parseOrDie = true);
  bool parsedOk() const { return status == JSON_OK; }
};

// (mostly) JSON-related utility functions
struct JsonUtil {
  static JsonValue* find (const JsonValue& parent, const char* key);
  static JsonValue& findOrDie (const JsonValue& parent, const char* key);
  static vguard<double> doubleVec (const JsonValue& arr);
  static vguard<size_t> indexVec (const JsonValue& arr);
  static string quoteEscaped (const string& str);
  static string toString (double d);
  static string toString (const map<string,string>& tags, size_t indent = 0);

  // helpers
  static string readStringFromStream (istream& in, bool keepNewlines = false);
};

#endif /* JSONUTIL_INCLUDED */
