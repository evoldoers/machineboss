#include <float.h>
#include "jsonutil.h"
#include "util.h"
#include "logger.h"

JsonMap::JsonMap (const JsonValue& value) {
  initMap (value);
}

void JsonMap::initMap (const JsonValue& value) {
  Assert (value.getTag() == JSON_OBJECT, "JSON value is not an object");
  for (auto n : value)
    m[string(n->key)] = &n->value;
}

bool JsonMap::contains (const char* key) const {
  return contains (string (key));
}

bool JsonMap::contains (const string& key) const {
  return m.find(key) != m.end();
}

bool JsonMap::containsType (const char* key, JsonTag type) const {
  return containsType (string(key), type);
}

bool JsonMap::containsType (const string& key, JsonTag type) const {
  return contains(key) && (*this)[key].getTag() == type;
}

JsonValue& JsonMap::operator[] (const char* key) const {
  return (*this) [string (key)];
}

JsonValue& JsonMap::operator[] (const string& key) const {
  const auto i = m.find(key);
  Require (i != m.end(), "Couldn't find %s in JSON", key.c_str());
  return *i->second;
}

JsonValue& JsonMap::getType (const char* key, JsonTag type) const {
  Require (containsType (key, type), "Couldn't find %s of correct type in JSON", key);
  return (*this)[key];
}

JsonValue& JsonMap::getType (const string& key, JsonTag type) const {
  Require (containsType (key, type), "Couldn't find %s of correct type in JSON", key.c_str());
  return (*this)[key];
}

JsonMap JsonMap::getObject (const char* key) const {
  const JsonValue json = getType (key, JSON_OBJECT);
  return JsonMap (json);
}

JsonMap JsonMap::getObject (const string& key) const {
  const JsonValue json = getType (key, JSON_OBJECT);
  return JsonMap (json);
}

double JsonMap::getNumber (const char* key) const {
  return getType (key, JSON_NUMBER).toNumber();
}

double JsonMap::getNumber (const string& key) const {
  return getType (key, JSON_NUMBER).toNumber();
}

string JsonMap::getString (const char* key) const {
  return string (getType (key, JSON_STRING).toString());
}

string JsonMap::getString (const string& key) const {
  return string (getType (key, JSON_STRING).toString());
}

bool JsonMap::getBool (const char* key) const {
  const JsonTag tag = (*this)[key].getTag();
  Assert (tag == JSON_TRUE || tag == JSON_FALSE, "%s is not a boolean in JSON", key);
  return tag == JSON_TRUE;
}

bool JsonMap::getBool (const string& key) const {
  const JsonTag tag = (*this)[key].getTag();
  Assert (tag == JSON_TRUE || tag == JSON_FALSE, "%s is not a boolean in JSON", key.c_str());
  return tag == JSON_TRUE;
}

ParsedJson::ParsedJson (const string& s, bool parseOrDie) {
  parse (s, parseOrDie);
}

ParsedJson::ParsedJson (istream& in, bool parseOrDie) {
  parse (JsonUtil::readStringFromStream (in), parseOrDie);
}

void ParsedJson::parse (const string& s, bool parseOrDie) {
  LogThisAt(9, "Parsing string:\n" << s << endl);
  str = s;
  buf = new char[str.size() + 1];
  strcpy (buf, str.c_str());
  status = jsonParse (buf, &endPtr, &value, allocator);
  if (parsedOk()) {
    if (value.getTag() == JSON_OBJECT)
      initMap(value);
  } else {
    if (parseOrDie)
      Fail ("JSON parsing error: %s at byte %zd", jsonStrError(status), endPtr - buf);
    else
      Warn ("JSON parsing error: %s at byte %zd", jsonStrError(status), endPtr - buf);
  }
}

ParsedJson::~ParsedJson() {
  if (buf) delete[] buf;
}

JsonValue* JsonUtil::find (const JsonValue& parent, const char* key) {
  Assert (parent.getTag() == JSON_OBJECT, "JSON value is not an object");
  for (auto i : parent)
    if (strcmp (i->key, key) == 0)
      return &i->value;
  return NULL;
}

JsonValue& JsonUtil::findOrDie (const JsonValue& parent, const char* key) {
  JsonValue* val = find (parent, key);
  Assert (val != NULL, "Couldn't find JSON tag %s", key);
  return *val;
}

vguard<double> JsonUtil::doubleVec (const JsonValue& arr) {
  vguard<double> v;
  Assert (arr.getTag() == JSON_ARRAY, "JSON value is not an array");
  for (auto n : arr) {
    Assert (n->value.getTag() == JSON_NUMBER, "JSON value is not a number");
    v.push_back (n->value.toNumber());
  }
  return v;
}

vguard<size_t> JsonUtil::indexVec (const JsonValue& arr) {
  vguard<size_t> v;
  Assert (arr.getTag() == JSON_ARRAY, "JSON value is not an array");
  for (auto n : arr) {
    Assert (n->value.getTag() == JSON_NUMBER, "JSON value is not a number");
    v.push_back ((size_t) n->value.toNumber());
  }
  return v;
}

string JsonUtil::quoteEscaped (const string& str) {
  string esc;
  write_quoted_escaped (str, back_inserter(esc));
  return esc;
}

string JsonUtil::readStringFromStream (istream& in, bool keepNewlines) {
  string s;
  while (in && !in.eof()) {
    string line;
    getline(in,line);
    s += line;
    if (keepNewlines)
      s += '\n';
  }
  return s;
}

string JsonUtil::toString (double d) {
  if (d < -DBL_MAX)
    return string ("\"-inf\"");
  else if (d > DBL_MAX)
    return string ("\"inf\"");
  return to_string (d);
}

string JsonUtil::toString (const map<string,string>& tags, size_t indent) {
  string s;
  if (tags.empty())
    s = "{ }";
  else {
    bool first = true;
    for (auto& tag_val : tags) {
      if (first)
	s += tags.size() == 1 ? string("{ ") : (string("\n") + string(indent,' ') + "{");
      else
	s += ",";
      first = false;
      if (tags.size() > 1)
	s += "\n" + string(indent+1,' ');
      s += "\"" + tag_val.first + "\": \"" + tag_val.second + "\"";
    }
    s += (tags.size() == 1 ? string(" ") : (string("\n") + string(indent,' '))) + "}";
  }
  return s;
}
