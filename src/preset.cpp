#include "preset.h"

struct PresetCache {
  map<string,string> namedPreset;
  PresetCache();
};

#define addPreset(NAME) namedPreset[#NAME] = string (preset_##NAME##_json, preset_##NAME##_json + preset_##NAME##_json_len);

#include "preset/compdna.h"
#include "preset/comprna.h"

PresetCache::PresetCache() {
  addPreset(compdna);
  addPreset(comprna);
}

PresetCache presetCache;  // singleton

Machine MachinePresets::makePreset (const char* presetName) {
  const auto presetText = presetCache.namedPreset.at (string (presetName));
  return MachineLoader::fromJson (json::parse (presetText));
}
