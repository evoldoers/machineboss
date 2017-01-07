#ifndef SCHEMA_INCLUDED
#define SCHEMA_INCLUDED

#include <json.hpp>

struct MachineSchema {
  static bool validate (const nlohmann::json&);
};

#endif /* SCHEMA_INCLUDED */
