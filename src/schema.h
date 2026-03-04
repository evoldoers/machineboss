#ifndef SCHEMA_INCLUDED
#define SCHEMA_INCLUDED

#include "json.hpp"

namespace MachineBoss {

namespace detail {

struct MachineSchema {
  static bool validate (const char* schemaName, const nlohmann::json&);
  static void validateOrDie (const char* schemaName, const nlohmann::json&);
};

}  // end namespace detail

using detail::MachineSchema;

}  // end namespace

#endif /* SCHEMA_INCLUDED */
