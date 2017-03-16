#include "preset.h"
#include "util.h"

struct PresetCache {
  map<string,string> namedPreset;
  PresetCache();
};

#define addPreset(NAME) namedPreset[#NAME] = string (preset_##NAME##_json, preset_##NAME##_json + preset_##NAME##_json_len);

#include "preset/compdna.h"
#include "preset/comprna.h"
#include "preset/null.h"
#include "preset/dnapsw.h"
#include "preset/protpsw.h"

PresetCache::PresetCache() {
  addPreset(compdna);
  addPreset(comprna);
  addPreset(null);
  addPreset(dnapsw);
  addPreset(protpsw);
}

PresetCache presetCache;  // singleton

Machine MachinePresets::makePreset (const string& presetName) {
  if (!presetCache.namedPreset.count(presetName))
    throw runtime_error (string("Preset ") + presetName + " not found");
  const auto presetText = presetCache.namedPreset.at (presetName);
  return MachineLoader::fromJson (json::parse (presetText));
}

Machine MachinePresets::makePreset (const char* presetName) {
  return makePreset (string (presetName));
}

vector<string> MachinePresets::presetNames() {
  return extract_keys (presetCache.namedPreset);
}
