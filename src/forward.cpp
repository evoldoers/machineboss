#include "forward.h"

using namespace MachineBoss;

ForwardMatrix::ForwardMatrix (const EvaluatedMachine& m, const SeqPair& s)
  : MappedForwardMatrix (m, s)
{ }

ForwardMatrix::ForwardMatrix (const EvaluatedMachine& m, const SeqPair& s, const Envelope& e)
  : MappedForwardMatrix (m, s, e)
{ }

ForwardMatrix::ForwardMatrix (const EvaluatedMachine& m, const SeqPair& s, const Envelope& e, StateIndex startState)
  : MappedForwardMatrix (m, s, e, startState)
{ }

MachinePath ForwardMatrix::samplePath (const Machine& m, mt19937& rng) const {
  return traceBack (m, randomTransSelector (rng));
}

MachinePath ForwardMatrix::samplePath (const Machine& m, StateIndex s, mt19937& rng) const {
  return traceBack (m, s, randomTransSelector (rng));
}
