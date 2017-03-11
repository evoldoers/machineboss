#include "eval.h"
#include "weight.h"
#include "util.h"

EvaluatedMachineState::Trans::Trans (StateIndex s, LogWeight lw, TransIndex ti) :
  state (s), logWeight (lw), transIndex (ti)
{ }

EvaluatedMachine::EvaluatedMachine (const Machine& machine, const Params& params) :
  inputTokenizer (machine.inputAlphabet()),
  outputTokenizer (machine.outputAlphabet()),
  state (machine.nStates())
{
  Assert (machine.isAdvancingMachine(), "Machine is not topologically sorted");
  for (StateIndex s = 0; s < nStates(); ++s) {
    state[s].name = machine.state[s].name;
    EvaluatedMachineState::TransIndex ti = 0;
    for (const auto& trans: machine.state[s].trans) {
      const StateIndex d = trans.dest;
      const InputToken in = inputTokenizer.sym2tok.at (trans.in);
      const OutputToken out = outputTokenizer.sym2tok.at (trans.out);
      const LogWeight lw = log (WeightAlgebra::eval (trans.weight, params.defs));
      state[s].outgoing[in][out].push_back (EvaluatedMachineState::Trans (d, lw, ti));
      state[d].incoming[in][out].push_back (EvaluatedMachineState::Trans (s, lw, ti));
      ++ti;
    }
    state[s].nTransitions = ti;
  }
}

StateIndex EvaluatedMachine::nStates() const {
  return state.size();
}

StateIndex EvaluatedMachine::startState() const {
  Assert (nStates() > 0, "EvaluatedMachine has no states");
  return 0;
}

StateIndex EvaluatedMachine::endState() const {
  Assert (nStates() > 0, "EvaluatedMachine has no states");
  return nStates() - 1;
}
