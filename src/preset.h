#ifndef PRESET_INCLUDED
#define PRESET_INCLUDED

#include "machine.h"

namespace MachineBoss {

struct MachinePresets {
  static Machine makePreset (const char* presetName);
  static Machine makePreset (const string& presetName);
  static vector<string> presetNames();
};

}  // end namespace

#endif /* PRESET_INCLUDED */
