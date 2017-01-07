#include <iostream>

#include <valijson/adapters/nlohmann_json_adapter.hpp>
#include <valijson/utils/nlohmann_json_utils.hpp>
#include <valijson/schema.hpp>
#include <valijson/schema_parser.hpp>
#include <valijson/validator.hpp>
#include <valijson/validation_results.hpp>

#include "schema.h"
#include "util.h"
#include "../schema/machine.h"

using namespace std;

using json = nlohmann::json;

using valijson::Schema;
using valijson::SchemaParser;
using valijson::Validator;
using valijson::ValidationResults;
using valijson::adapters::NlohmannJsonAdapter;

bool MachineSchema::validate (const nlohmann::json& data) {
  Schema schema;
  SchemaParser parser;
  const json schemaDoc = json::parse (schema_machine_json);
  NlohmannJsonAdapter schemaAdapter (schemaDoc);
  parser.populateSchema(schemaAdapter, schema);
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
