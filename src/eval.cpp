#include "eval.h"
#include "weight.h"

EvaluatedMachineTransition::EvaluatedMachineTransition (StateIndex src, const MachineTransition& trans, const Params& params, const InputTokenizer& inTok, const OutputTokenizer& outTok) :
  in (inTok.sym2tok.at(trans.in)),
  out (outTok.sym2tok.at(trans.out)),
  src (src),
  dest (trans.dest),
  logWeight (log (WeightAlgebra::eval (trans.weight, params)))
{ }

EvaluatedMachine::EvaluatedMachine (const Machine& machine, const Params& params) :
  inputTokenizer (machine.inputAlphabet()),
  outputTokenizer (machine.outputAlphabet()),
  state (machine.nStates())
{
  for (StateIndex s = 0; s < nStates(); ++s) {
    state[s].name = machine.state[s].name;
    for (const auto& trans: machine.state[s].trans) {
      const StateIndex d = trans.dest;
      const EvaluatedMachineTransition evalTrans (s, trans, params, inputTokenizer, outputTokenizer);
      const pair<OutputToken,EvaluatedMachineTransition> outEvalTrans (evalTrans.out, evalTrans);
      state[s].outgoing[evalTrans.in].insert (outEvalTrans);
      state[d].incoming[evalTrans.in].insert (outEvalTrans);
    }
  }
}
