#ifndef BEAM_ALIGN_INCLUDED
#define BEAM_ALIGN_INCLUDED

#include <algorithm>
#include <tuple>
#include <map>
#include "eval.h"
#include "seqpair.h"
#include "logsumexp.h"
#include "logger.h"

#ifndef DefaultBeamWidth
#define DefaultBeamWidth 100
#endif

namespace MachineBoss {

class BeamAlignMatrix {
public:
  typedef Envelope::InputIndex InputIndex;
  typedef Envelope::OutputIndex OutputIndex;

  BeamAlignMatrix (const EvaluatedMachine& machine, const SeqPair& seqPair, size_t beamWidth = DefaultBeamWidth);

  double logLike() const;
  MachinePath path (const Machine&) const;

private:
  struct BeamCell {
    InputIndex inPos;
    OutputIndex outPos;
    StateIndex state;
    LogWeight score;
    int parentWavefront;  // wavefront index of parent (-1 for start)
    int parentIdx;        // index within parent wavefront's beam (-1 for start)
    StateIndex srcState;  // source state (for traceback)
  };

  const EvaluatedMachine& machine;
  const vguard<InputToken> input;
  const vguard<OutputToken> output;
  const InputIndex inLen;
  const OutputIndex outLen;
  const StateIndex nStates;
  const size_t beamWidth;

  // wavefronts[d] = beam at diagonal d (d = inPos + outPos)
  vguard<vguard<BeamCell> > wavefronts;

  void fill();
};

}  // end namespace

#endif /* BEAM_ALIGN_INCLUDED */
