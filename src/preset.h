#ifndef PRESET_INCLUDED
#define PRESET_INCLUDED

#include "machine.h"

struct MachinePresets {
  static Machine makePreset (const char* presetName);
};

#endif /* PRESET_INCLUDED */
