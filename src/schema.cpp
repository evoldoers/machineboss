#include <iostream>

#include <valijson/adapters/nlohmann_json_adapter.hpp>
#include <valijson/utils/nlohmann_json_utils.hpp>
#include <valijson/schema.hpp>
#include <valijson/schema_parser.hpp>
#include <valijson/validator.hpp>
#include <valijson/validation_results.hpp>

#include "schema.h"
#include "schema/machine.h"
#include "schema/expr.h"
#include "util.h"

#define SchemaUrlPrefix "https://raw.githubusercontent.com/ihh/acidbot/master/schema/"

using namespace std;

using json = nlohmann::json;

using valijson::Schema;
using valijson::SchemaParser;
using valijson::Validator;
using valijson::ValidationResults;
using valijson::adapters::NlohmannJsonAdapter;

struct SchemaCache {
  map<string,string> namedSchema;
  static json getSchema (const string& name);
  SchemaCache();
};

#define addSchema(NAME) namedSchema[string(SchemaUrlPrefix #NAME ".json")] = string (schema_##NAME##_json, schema_##NAME##_json + schema_##NAME##_json_len);
SchemaCache::SchemaCache() {
  addSchema(machine);
  addSchema(expr);
}

SchemaCache schemaCache;  // singleton

json SchemaCache::getSchema (const string& name) {
  const auto schemaText = schemaCache.namedSchema.at (string (name));
  return json::parse (schemaText);
}

json* fetchSchema (const string& uri) {
  return new json (schemaCache.getSchema(uri));
}

void freeSchema (const json* schema) {
  delete schema;
}

bool MachineSchema::validate (const nlohmann::json& data) {
  Schema schema;
  SchemaParser parser;
  const json schemaDoc = schemaCache.getSchema (string (SchemaUrlPrefix "machine.json"));
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
