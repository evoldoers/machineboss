#include "counts.h"
#include "backward.h"
#include "util.h"

MachineCounts::MachineCounts (const EvaluatedMachine& machine, const SeqPair& seqPair) :
  count (machine.nStates())
{
  for (StateIndex s = 0; s < machine.nStates(); ++s)
    count[s].resize (machine.state[s].nTransitions);

  const ForwardMatrix forward (machine, seqPair);
  const BackwardMatrix backward (machine, seqPair);

  backward.getCounts (forward, *this);
}

void MachineCounts::writeJson (ostream& outs) const {
  vguard<string> s;
  for (const auto& c: count)
    s.push_back (string("[") + to_string_join (c, ",") + "]");
  outs << "[" << join (s, ",\n ") << "]" << endl;
}
