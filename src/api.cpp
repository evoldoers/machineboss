#include "api.h"
#include "forward.h"
#include "backward.h"
#include "viterbi.h"
#include "beam.h"
#include "ctc.h"
#include "fitter.h"

using namespace MachineBoss;

Machine MachineBoss::loadMachine (const string& filename) {
  return JsonReader<Machine>::fromFile (filename);
}

Machine MachineBoss::loadMachineJson (const string& jsonString) {
  return JsonReader<Machine>::fromJsonString (jsonString);
}

void MachineBoss::saveMachine (const Machine& machine, const string& filename) {
  JsonWriter<Machine>::toFile (machine, filename);
}

string MachineBoss::machineToJson (const Machine& machine) {
  return JsonWriter<Machine>::toJsonString (machine);
}

Machine MachineBoss::mergeEquivalentStates (const Machine& machine) {
  return machine.mergeEquivalentStates();
}

double MachineBoss::forwardLogLike (const Machine& machine, const Params& params, const SeqPair& seqPair) {
  const EvaluatedMachine eval (machine, params);
  const ForwardMatrix fwd (eval, seqPair);
  return fwd.logLike();
}

double MachineBoss::forwardLogLike (const Machine& machine, const Params& params, const SeqPair& seqPair, const Envelope& env) {
  const EvaluatedMachine eval (machine, params);
  const ForwardMatrix fwd (eval, seqPair, env);
  return fwd.logLike();
}

double MachineBoss::viterbiLogLike (const Machine& machine, const Params& params, const SeqPair& seqPair) {
  const EvaluatedMachine eval (machine, params);
  const ViterbiMatrix vit (eval, seqPair);
  return vit.logLike();
}

MachinePath MachineBoss::viterbiAlign (const Machine& machine, const Params& params, const SeqPair& seqPair) {
  const EvaluatedMachine eval (machine, params);
  const ViterbiMatrix vit (eval, seqPair);
  return vit.path (machine);
}

MachineCounts MachineBoss::forwardBackwardCounts (const Machine& machine, const Params& params, const SeqPair& seqPair) {
  const EvaluatedMachine eval (machine, params);
  MachineCounts counts (eval, seqPair);
  return counts;
}

MachineCounts MachineBoss::forwardBackwardCounts (const Machine& machine, const Params& params, const SeqPairList& seqPairList) {
  const EvaluatedMachine eval (machine, params);
  MachineCounts counts (eval, seqPairList);
  return counts;
}

Params MachineBoss::baumWelchFit (const Machine& machine, const Constraints& constraints, const SeqPairList& seqPairList,
                                  const Params& seed, const Params& constants) {
  MachineFitter fitter;
  fitter.machine = machine;
  fitter.constraints = constraints;
  fitter.seed = seed;
  fitter.constants = constants;
  return fitter.fit (seqPairList);
}

vguard<InputSymbol> MachineBoss::beamDecode (const Machine& machine, const Params& params,
                                             const vguard<OutputSymbol>& output,
                                             size_t beamWidth) {
  const EvaluatedMachine eval (machine, params);
  BeamSearchMatrix beam (eval, output, beamWidth);
  return beam.bestSeq();
}

vguard<InputSymbol> MachineBoss::prefixDecode (const Machine& machine, const Params& params,
                                               const vguard<OutputSymbol>& output,
                                               long maxBacktrack) {
  const EvaluatedMachine eval (machine, params);
  PrefixTree tree (eval, output, maxBacktrack);
  return tree.doPrefixSearch();
}
