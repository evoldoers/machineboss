#include "eval.h"
#include "weight.h"
#include "util.h"
#include "logger.h"

void EvaluatedMachineState::Trans::init (LogWeight lw, TransIndex ti) {
  logWeight = lw;
  transIndex = ti;
}

EvaluatedMachine::EvaluatedMachine (const Machine& machine, const Params& params) :
  inputTokenizer (machine.inputAlphabet()),
  outputTokenizer (machine.outputAlphabet()),
  state (machine.nStates())
{
  Assert (machine.isAdvancingMachine(), "Machine is not topologically sorted");
  Assert (machine.isAligningMachine(), "Machine has ambiguous transitions");

  ProgressLog(plog,6);
  plog.initProgress ("Evaluating transition weights");

  for (StateIndex s = 0; s < nStates(); ++s) {
    plog.logProgress (s / (double) nStates(), "state %lu/%lu", s, nStates());
    state[s].name = machine.state[s].name;
    EvaluatedMachineState::TransIndex ti = 0;
    for (const auto& trans: machine.state[s].trans) {
      const StateIndex d = trans.dest;
      const InputToken in = inputTokenizer.sym2tok.at (trans.in);
      const OutputToken out = outputTokenizer.sym2tok.at (trans.out);
      const LogWeight lw = log (WeightAlgebra::eval (trans.weight, params.defs));
      state[s].outgoing[in][out][d].init (lw, ti);
      state[d].incoming[in][out][s].init (lw, ti);
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

void EvaluatedMachine::writeJson (ostream& out) const {
  out << "{\"state\":" << endl << " [";
  for (StateIndex s = 0; s < nStates(); ++s) {
    const EvaluatedMachineState& ms = state[s];
    out << (s ? "  " : "") << "{\"n\":" << s;
    if (!ms.name.is_null())
      out << "," << endl << "   \"id\":" << ms.name;
    if (ms.incoming.size()) {
      out << "," << endl << "   \"incoming\":[";
      size_t nt = 0;
      for (const auto& iost: ms.incoming)
	for (const auto& ost: iost.second)
	  for (const auto& st: ost.second) {
	    const auto& t = st.second;
	    if (nt++)
	      out << "," << endl << "               ";
	    out << "{\"from\":" << st.first;
	    if (iost.first) out << ",\"in\":\"" << inputTokenizer.tok2sym[iost.first] << "\"";
	    if (ost.first) out << ",\"out\":\"" << outputTokenizer.tok2sym[ost.first] << "\"";
	    out << ",\"logWeight\":" << t.logWeight;
	    out << "}";
	  }
      out << "]";
    }
    if (ms.outgoing.size()) {
      out << "," << endl << "   \"outgoing\":[";
      size_t nt = 0;
      for (const auto& iost: ms.outgoing)
	for (const auto& ost: iost.second)
	  for (const auto& st: ost.second) {
	    const auto& t = st.second;
	    if (nt++)
	      out << "," << endl << "               ";
	    out << "{\"to\":" << st.first;
	    if (iost.first) out << ",\"in\":\"" << inputTokenizer.tok2sym[iost.first] << "\"";
	    if (ost.first) out << ",\"out\":\"" << outputTokenizer.tok2sym[ost.first] << "\"";
	    out << ",\"logWeight\":" << t.logWeight;
	    out << "}";
	  }
      out << "]";
    }
    out << "}";
    if (s < nStates() - 1)
      out << "," << endl;
  }
  out << endl << " ]" << endl << "}" << endl;
}

string EvaluatedMachine::toJsonString() const {
  ostringstream outs;
  writeJson (outs);
  return outs.str();
}
