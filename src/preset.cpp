#include "preset.h"
#include "util.h"

struct PresetCache {
  map<string,string> namedPreset;
  vguard<string> names;
  PresetCache();
};

#define addPreset(NAME) namedPreset[#NAME] = string (preset_##NAME##_json, preset_##NAME##_json + preset_##NAME##_json_len); names.push_back (#NAME);

#include "preset/null.h"

#include "preset/compdna.h"
#include "preset/comprna.h"

#include "preset/dnapsw.h"
#include "preset/protpsw.h"

#include "preset/translate.h"
#include "preset/prot2dna.h"
#include "preset/psw2dna.h"

#include "preset/dna2rna.h"
#include "preset/rna2dna.h"

#include "preset/bintern.h"
#include "preset/terndna.h"

#include "preset/jukescantor.h"
#include "preset/dnapswnbr.h"

#include "preset/tkf91root.h"
#include "preset/tkf91branch.h"

PresetCache::PresetCache() {
  addPreset(null);

  addPreset(compdna);
  addPreset(comprna);

  addPreset(dnapsw);
  addPreset(protpsw);

  addPreset(translate);
  addPreset(prot2dna);
  addPreset(psw2dna);

  addPreset(dna2rna);
  addPreset(rna2dna);

  addPreset(bintern);
  addPreset(terndna);

  addPreset(jukescantor);
  addPreset(dnapswnbr);

  addPreset(tkf91root);
  addPreset(tkf91branch);
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
  return presetCache.names;
}
