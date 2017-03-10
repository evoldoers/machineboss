#include <iostream>

#include <valijson/adapters/nlohmann_json_adapter.hpp>
#include <valijson/utils/nlohmann_json_utils.hpp>
#include <valijson/schema.hpp>
#include <valijson/schema_parser.hpp>
#include <valijson/validator.hpp>
#include <valijson/validation_results.hpp>

#include "schema.h"
#include "util.h"

#include "schema/expr.h"
#include "schema/machine.h"
#include "schema/params.h"
#include "schema/seqpair.h"

#define SchemaUrlPrefix "https://raw.githubusercontent.com/ihh/acidbot/master/schema/"
#define SchemaUrlSuffix ".json"

using namespace std;

using json = nlohmann::json;

using valijson::Schema;
using valijson::SchemaParser;
using valijson::Validator;
using valijson::ValidationResults;
using valijson::adapters::NlohmannJsonAdapter;

struct SchemaCache {
  map<string,string> namedSchema;
  json getSchema (const string& name) const;
  bool validate (const char* schemaName, const json& data) const;
  SchemaCache();
};

#define addSchema(NAME) namedSchema[string(SchemaUrlPrefix #NAME SchemaUrlSuffix)] = string (schema_##NAME##_json, schema_##NAME##_json + schema_##NAME##_json_len);

SchemaCache::SchemaCache() {
  addSchema(machine);
  addSchema(expr);
  addSchema(seqpair);
  addSchema(params);
}

SchemaCache schemaCache;  // singleton

json SchemaCache::getSchema (const string& name) const {
  const auto schemaText = namedSchema.at (string (name));
  return json::parse (schemaText);
}

json* fetchSchema (const string& uri) {
  return new json (schemaCache.getSchema(uri));
}

void freeSchema (const json* schema) {
  delete schema;
}

bool SchemaCache::validate (const char* schemaNameStem, const json& data) const {
  Schema schema;
  SchemaParser parser;
  const string schemaName = string(SchemaUrlPrefix) + schemaNameStem + SchemaUrlSuffix;
  const json schemaDoc = schemaCache.getSchema (schemaName);
  NlohmannJsonAdapter schemaAdapter (schemaDoc);
  parser.populateSchema (schemaAdapter, schema, fetchSchema, freeSchema);
  NlohmannJsonAdapter dataAdapter (data);
  Validator validator;
  ValidationResults results;
  const bool valid = validator.validate(schema, dataAdapter, &results);
  if (!valid) {
    ValidationResults::Error err;
    while (results.popError (err))
      cerr << join(err.context,".") << ": " << err.description << endl;
  }
  return valid;
}

bool MachineSchema::validate (const char* schemaNameStem, const nlohmann::json& data) {
  return schemaCache.validate (schemaNameStem, data);
}

void MachineSchema::validateOrDie (const char* schemaNameStem, const nlohmann::json& data) {
  const string err = string(schemaNameStem) + " specification does not fit schema";
  Require (validate(schemaNameStem,data), err.c_str());
}
