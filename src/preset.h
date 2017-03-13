#ifndef PRESET_INCLUDED
#define PRESET_INCLUDED

#include "machine.h"

struct MachinePresets {
  static Machine makePreset (const char* presetName);
  static Machine makePreset (const string& presetName);
  static vector<string> presetNames();
};

#endif /* PRESET_INCLUDED */
