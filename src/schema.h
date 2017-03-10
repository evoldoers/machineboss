#ifndef SCHEMA_INCLUDED
#define SCHEMA_INCLUDED

#include <json.hpp>

struct MachineSchema {
  static bool validate (const char* schemaName, const nlohmann::json&);
  static void validateOrDie (const char* schemaName, const nlohmann::json&);
};

#endif /* SCHEMA_INCLUDED */
