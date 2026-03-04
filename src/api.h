#ifndef API_INCLUDED
#define API_INCLUDED

#include "machine.h"
#include "params.h"
#include "seqpair.h"
#include "constraints.h"
#include "counts.h"
#include "eval.h"
#include "vguard.h"

namespace MachineBoss {

  // Machine I/O
  Machine loadMachine (const string& filename);
  Machine loadMachineJson (const string& jsonString);
  void saveMachine (const Machine&, const string& filename);
  string machineToJson (const Machine&);

  // Forward algorithm
  double forwardLogLike (const Machine&, const Params&, const SeqPair&);
  double forwardLogLike (const Machine&, const Params&, const SeqPair&, const Envelope&);

  // Viterbi
  double viterbiLogLike (const Machine&, const Params&, const SeqPair&);
  MachinePath viterbiAlign (const Machine&, const Params&, const SeqPair&);

  // Forward-Backward counts
  MachineCounts forwardBackwardCounts (const Machine&, const Params&, const SeqPair&);
  MachineCounts forwardBackwardCounts (const Machine&, const Params&, const SeqPairList&);

  // Baum-Welch training
  Params baumWelchFit (const Machine&, const Constraints&, const SeqPairList&,
                       const Params& seed = Params(), const Params& constants = Params());

  // Beam search decoding
  vguard<InputSymbol> beamDecode (const Machine&, const Params&,
                                  const vguard<OutputSymbol>&,
                                  size_t beamWidth = 100);

  // Prefix search decoding
  vguard<InputSymbol> prefixDecode (const Machine&, const Params&,
                                    const vguard<OutputSymbol>&,
                                    long maxBacktrack = -1);

}  // end namespace

#endif /* API_INCLUDED */
