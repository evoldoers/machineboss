#include "preset.h"
#include "util.h"

using namespace MachineBoss;

struct PresetCache {
  map<string,string> namedPreset;
  vguard<string> names;
  PresetCache();
};

#define addPreset(NAME) namedPreset[#NAME] = string (preset_##NAME##_json, preset_##NAME##_json + preset_##NAME##_json_len); names.push_back (#NAME);
// addPresetAs: register a preset whose user-facing name differs from the C identifier
// (used for hyphenated names like "tkf91-branch-dna-jc" where xxd produces preset_tkf91_branch_dna_jc_json).
#define addPresetAs(C_NAME, USER_NAME) namedPreset[USER_NAME] = string (preset_##C_NAME##_json, preset_##C_NAME##_json + preset_##C_NAME##_json_len); names.push_back (USER_NAME);

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

#include "preset/iupacdna.h"
#include "preset/iupacaa.h"

#include "preset/bintern.h"
#include "preset/terndna.h"

#include "preset/jukescantor.h"
#include "preset/dnapswnbr.h"

#include "preset/tkf91-root-dna-jc.h"
#include "preset/tkf91-branch-dna-jc.h"
#include "preset/tkf92-branch-prot-f81.h"

#include "preset/tolower.h"
#include "preset/toupper.h"

#include "preset/hamming31.h"
#include "preset/hamming74.h"

PresetCache::PresetCache() {
  addPreset(null);

  addPreset(compdna);
  addPreset(comprna);

  addPreset(dnapsw);
  addPreset(protpsw);

  addPreset(translate);
  addPreset(prot2dna);
  addPreset(psw2dna);

  addPreset(iupacdna);
  addPreset(iupacaa);

  addPreset(dna2rna);
  addPreset(rna2dna);

  addPreset(bintern);
  addPreset(terndna);

  addPreset(jukescantor);
  addPreset(dnapswnbr);

  addPresetAs(tkf91_root_dna_jc,   "tkf91-root-dna-jc");
  addPresetAs(tkf91_branch_dna_jc, "tkf91-branch-dna-jc");
  addPresetAs(tkf92_branch_prot_f81, "tkf92-branch-prot-f81");

  addPreset(tolower);
  addPreset(toupper);

  addPreset(hamming31);
  addPreset(hamming74);
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
