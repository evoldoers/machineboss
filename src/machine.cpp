#include <iomanip>
#include <fstream>
#include <set>
#include <functional>
#include <json.hpp>

#include "machine.h"
#include "fastseq.h"
#include "logger.h"
#include "schema.h"
#include "params.h"
#include "preset.h"

#include "backward.h"

using namespace MachineBoss;

using json = nlohmann::json;
using placeholders::_1;

struct TransAccumulator {
  TransList* transList;  // if non-null, will accumulate transitions direct to this list, without collapsing
  map<StateIndex,map<InputSymbol,map<OutputSymbol,WeightExpr> > > t;
  TransAccumulator();
  void clear();
  void accumulate (InputSymbol in, OutputSymbol out, StateIndex dest, WeightExpr w);
  void accumulate (const MachineTransition&);
  TransList transitions() const;
};

MachineTransition::MachineTransition()
{ }

MachineTransition::MachineTransition (InputSymbol in, OutputSymbol out, StateIndex dest, WeightExpr weight)
  : in (in),
    out (out),
    dest (dest),
    weight (weight)
{ }

MachineTransition::MachineTransition (InputSymbol in, OutputSymbol out, StateIndex dest, double weight)
  : in (in),
    out (out),
    dest (dest),
    weight (WeightAlgebra::doubleConstant (weight))
{ }

bool MachineTransition::inputEmpty() const {
  return in.empty();
}

bool MachineTransition::outputEmpty() const {
  return out.empty();
}

bool MachineTransition::isSilent() const {
  return in.empty() && out.empty();
}

bool MachineTransition::isLoud() const {
  return !in.empty() || !out.empty();
}

MachineState::MachineState()
{ }

bool MachineState::terminates() const {
  return trans.empty();
}

MachineTransition MachineState::getTransition (size_t n) const {
  auto iter = trans.begin();
  while (n) { ++iter; --n; }
  return *iter;
}

size_t MachineState::findTransition (const MachineTransition& t) const {
  size_t n = 0;
  for (const auto& mt: trans) {
    if (mt.in == t.in && mt.out == t.out && mt.dest == t.dest)
      return n;
    ++n;
  }
  Abort ("Transition not found");
  return numeric_limits<size_t>::max();
}

bool MachineState::exitsWithInput() const {
  for (const auto& t: trans)
    if (!t.inputEmpty())
      return true;
  return false;
}

bool MachineState::exitsWithoutInput() const {
  for (const auto& t: trans)
    if (t.inputEmpty())
      return true;
  return false;
}

bool MachineState::waits() const {
  return !exitsWithoutInput();
}

bool MachineState::continues() const {
  return !exitsWithInput() && !terminates();
}

bool MachineState::exitsWithIO() const {
  for (const auto& t: trans)
    if (!t.inputEmpty() || !t.outputEmpty())
      return true;
  return false;
}

bool MachineState::exitsWithoutIO() const {
  for (const auto& t: trans)
    if (t.inputEmpty() && t.outputEmpty())
      return true;
  return false;
}

bool MachineState::isSilent() const {
  return !exitsWithIO();
}

bool MachineState::isLoud() const {
  return exitsWithIO() && !exitsWithoutIO();
}

StateIndex Machine::nStates() const {
  return state.size();
}

size_t Machine::nTransitions() const {
  size_t n = 0;
  for (auto& ms: state)
    n += ms.trans.size();
  return n;
}

size_t Machine::nConditionedTransitions() const {
  map<pair<InputSymbol,OutputSymbol>,size_t> count;
  size_t nullCount = 0;
  for (const auto& ms: state)
    for (const auto& t: ms.trans)
      if (t.isSilent())
	++nullCount;
      else
	++count[make_pair (t.in, t.out)];
  size_t maxCount = 0;
  for (const auto& p_c: count)
    maxCount = max (maxCount, p_c.second);
  return maxCount + nullCount;
}

StateIndex Machine::startState() const {
  Assert (nStates() > 0, "Machine has no states");
  return 0;
}

StateIndex Machine::endState() const {
  Assert (nStates() > 0, "Machine has no states");
  return nStates() - 1;
}

string Machine::stateNameJson (StateIndex s) const {
  if (state[s].name.is_null())
    return to_string(s);
  return state[s].name.dump();
}

vguard<InputSymbol> Machine::inputAlphabet() const {
  set<InputSymbol> alph;
  for (const auto& ms: state)
    for (const auto& t: ms.trans)
      if (!t.inputEmpty())
	alph.insert (t.in);
  return vguard<InputSymbol> (alph.begin(), alph.end());
}

vguard<OutputSymbol> Machine::outputAlphabet() const {
  set<OutputSymbol> alph;
  for (const auto& ms: state)
    for (const auto& t: ms.trans)
      if (!t.outputEmpty())
	alph.insert (t.out);
  return vguard<OutputSymbol> (alph.begin(), alph.end());
}

set<string> Machine::params() const {
  set<string> p;
  for (const auto& ms: state)
    for (const auto& t: ms.trans) {
      const auto tp = WeightAlgebra::params (t.weight, funcs.defs);
      p.insert (tp.begin(), tp.end());
    }
  return p;
}

void Machine::writeJson (ostream& out, bool memoizeRepeatedExpressions, bool showParams, bool useStateIDs) const {
  ExprMemos memo;
  ExprRefCounts counts = WeightAlgebra::zeroRefCounts();
  vguard<WeightExpr> common;
  map<string,string> name2def;
  vguard<string> names;
  if (memoizeRepeatedExpressions) {
    set<string> params;
    set<WeightExpr> visited;
    ParamDefs dummyDefs;
    ProgressLog(plogMemo,6);
    plogMemo.initProgress ("Analyzing transition weights to find repeated sub-expressions");
    for (StateIndex s = 0; s < nStates(); ++s) {
      plogMemo.logProgress (s / (double) nStates(), "finished %lu/%lu states", s, nStates());
      for (const auto& t: state[s].trans)
	WeightAlgebra::countRefs (t.weight, counts, params, dummyDefs, NULL);
    }

    auto iter = WeightAlgebra::exprBegin();
    for (ExprIndex n = 0; n < counts.size(); ++iter, ++n) {
      const WeightExpr expr = &*iter;
      if (counts[n] > 1
	  && expr->type != Dbl && expr->type != Int && expr->type != Param && expr->type != Null
	  && !WeightAlgebra::isOne (expr))
	common.push_back (expr);
    }

    map<string,string> def2name;
    size_t n = 0;
    for (const auto& expr: common) {
      const string def = WeightAlgebra::toJsonString (expr, &memo);
      if (def2name.count(def))
	memo[expr] = def2name.at(def);
      else {
	string prefix, name;
	do {
	  prefix = prefix + "_";
	} while (params.count (name = prefix + to_string(++n)));
	memo[expr] = name;
	name2def[name] = def;
	def2name[def] = name;
	names.push_back (name);
      }
    }
    LogThisAt(6,"Memoized " << names.size() << " duplicate expressions" << endl);
  }

  bool gotAllIDs = true;
  for (const auto& ms: state)
    if (ms.name.is_null()) {
      gotAllIDs = false;
      break;
    }

  vguard<json> uniqueName;
  if (useStateIDs) {
    uniqueName.reserve (nStates());
    set<string> seenStateID;
    for (StateIndex s = 0; s < nStates(); ++s) {
      const MachineState& ms = state[s];
      json id = ms.name;
      int n = 1;
      while (seenStateID.count (id.dump()))
	id = json::array ({{ ms.name, ++n }});
      seenStateID.insert (id.dump());
      uniqueName.push_back (id);
    }
  }
  
  out << "{\"state\":" << endl << " [";
  for (StateIndex s = 0; s < nStates(); ++s) {
    const MachineState& ms = state[s];
    out << (s ? "  " : "") << "{";
    if (!useStateIDs)
      out << "\"n\":" << s;
    if (useStateIDs || !ms.name.is_null()) {
      if (!useStateIDs)
	out << "," << endl << "   ";
      out << "\"id\":" << (useStateIDs ? uniqueName[s] : ms.name);
    }
    if (ms.trans.size()) {
      out << "," << endl << "   \"trans\":[";
      size_t nt = 0;
      for (const auto& t: ms.trans) {
	if (nt++)
	  out << "," << endl << "            ";
	out << "{\"to\":";
	if (useStateIDs)
	  out << uniqueName[t.dest];
	else
	  out << t.dest;
	if (!t.inputEmpty()) out << ",\"in\":\"" << escaped_str(t.in) << "\"";
	if (!t.outputEmpty()) out << ",\"out\":\"" << escaped_str(t.out) << "\"";
	if (!WeightAlgebra::isOne (t.weight)) {
	  out << ",\"weight\":";
	  WeightAlgebra::toJsonStream (out, t.weight, &memo);
	}
	out << "}";
      }
      out << "]";
    }
    out << "}";
    if (s < nStates() - 1)
      out << "," << endl;
  }
  out << endl << " ]";
  if (names.size() || funcs.defs.size()) {
    out << "," << endl << " \"defs\":";
    size_t count = 0;
    for (size_t n = 0; n < names.size(); ++n)
      out << ((count++) ? ",\n  " : "\n {")
	  << "\"" << names[n]
	  << "\":" << name2def[names[n]];
    for (const auto& def: funcs.defs) {
      out << ((count++) ? ",\n  " : "\n {")
	  << "\"" << def.first
	  << "\":";
      WeightAlgebra::toJsonStream (out, def.second, &memo);
    }
    out << "}";
  }
  if (showParams) {
    const map<string,string> paramConstraint = cons.byParam();
    vguard<string> unconsParams;
    for (const auto& p: params())
      if (!paramConstraint.count (p))
	unconsParams.push_back (p);
    if (unconsParams.size()) {
      out << "," << endl << " \"params\": [";
      size_t np = 0;
      for (auto& p: unconsParams)
	out << (np++ ? "," : "") << "\"" << escaped_str(p) << "\"";
      out << "]";
    }
  }
  if (!cons.empty()) {
    out << "," << endl << " \"cons\":" << endl;
    cons.writeJson (out);
  } else
    out << endl;
  out << "}" << endl;
}

void Machine::readJson (const json& pj) {
  MachineSchema::validateOrDie ("machine", pj);

  // This JSON notation for machine manipulation is untested, unused, and should probably go.... IH, 2/22/2019
  
  // Check for composite, concatenated, tranposed, reversed, etc etc transducers
  if (pj.count("compose")) {
    const auto arg = pj["compose"];
    *this = Machine::compose (JsonReader<Machine>::fromJson (arg[0]),
			      JsonReader<Machine>::fromJson (arg[1]),
			      true, true, Machine::BreakSilentCycles);

  } else if (pj.count("compose-sum")) {
    const auto arg = pj["compose-sum"];
    *this = Machine::compose (JsonReader<Machine>::fromJson (arg[0]),
			      JsonReader<Machine>::fromJson (arg[1]),
			      true, true, Machine::SumSilentCycles);

  } else if (pj.count("compose-unsort")) {
    const auto arg = pj["compose-unsort"];
    *this = Machine::compose (JsonReader<Machine>::fromJson (arg[0]),
			      JsonReader<Machine>::fromJson (arg[1]),
			      true, true, Machine::LeaveSilentCycles);

  } else if (pj.count("concat")) {
    const auto arg = pj["concat"];
    *this = Machine::concatenate (JsonReader<Machine>::fromJson (arg[0]),
				  JsonReader<Machine>::fromJson (arg[1]));
    
  } else if (pj.count("intersect")) {
    const auto arg = pj["intersect"];
    *this = Machine::intersect (JsonReader<Machine>::fromJson (arg[0]),
				JsonReader<Machine>::fromJson (arg[1]),
				Machine::BreakSilentCycles);
    
  } else if (pj.count("intersect-sum")) {
    const auto arg = pj["intersect-sum"];
    *this = Machine::intersect (JsonReader<Machine>::fromJson (arg[0]),
				JsonReader<Machine>::fromJson (arg[1]),
				Machine::SumSilentCycles);

  } else if (pj.count("intersect-unsort")) {
    const auto arg = pj["intersect-unsort"];
    *this = Machine::intersect (JsonReader<Machine>::fromJson (arg[0]),
				JsonReader<Machine>::fromJson (arg[1]),
				Machine::LeaveSilentCycles);

  } else if (pj.count("union")) {
    const auto arg = pj["union"];
    *this = Machine::takeUnion (JsonReader<Machine>::fromJson (arg[0]),
				JsonReader<Machine>::fromJson (arg[1]));

  } else if (pj.count("loop")) {
    const auto arg = pj["loop"];
    *this = Machine::kleeneLoop (JsonReader<Machine>::fromJson (arg[0]),
				 JsonReader<Machine>::fromJson (arg[1]));

  } else if (pj.count("opt")) {  // optional: zero-or-one
    const auto arg = pj["opt"];
    *this = Machine::zeroOrOne (JsonReader<Machine>::fromJson (pj["opt"]));

  } else if (pj.count("star")) {  // Kleene star
    *this = Machine::kleeneStar (JsonReader<Machine>::fromJson (pj["star"]));

  } else if (pj.count("plus")) {  // Kleene plus
    *this = Machine::kleenePlus (JsonReader<Machine>::fromJson (pj["plus"]));
    
  } else if (pj.count("eliminate")) {
    *this = JsonReader<Machine>::fromJson (pj["eliminate"]).eliminateSilentTransitions();

  } else if (pj.count("reverse")) {
    *this = JsonReader<Machine>::fromJson (pj["eliminate"]).reverse();

  } else if (pj.count("revcomp")) {
    // convoluted... a simpler built-in revcomp would be preferable, oh well
    const Machine m = JsonReader<Machine>::fromJson (pj["revcomp"]);
    const vguard<OutputSymbol> outAlph = m.outputAlphabet();
    const set<OutputSymbol> outAlphSet (outAlph.begin(), outAlph.end());
    *this = Machine::compose (m.reverse(),
			      MachinePresets::makePreset ((outAlphSet.count(string("U")) || outAlphSet.count(string("u")))
							  ? "comprna"
							  : "compdna"));

  } else if (pj.count("transpose")) {
    *this = JsonReader<Machine>::fromJson (pj["eliminate"]).transpose();

  } else {

    // Basic transducer with the following properties:
    // (mandatory) state: list of states with transitions
    //  (optional)  defs: function definitions
    //  (optional)  cons: parameter-fitting constraints
    if (pj.count("defs"))
      funcs.readJson (pj.at("defs"));
    if (pj.count("cons"))
      cons.readJson (pj.at("cons"));

    json jstate = pj.at("state");
    Assert (jstate.is_array(), "state is not an array");
    map<string,StateIndex> id2n;
    set<string> dupIds;
    for (const json& js : jstate) {
      MachineState ms;
      if (js.count("n")) {
	const StateIndex n = js.at("n").get<StateIndex>();
	Require ((StateIndex) state.size() == n, "StateIndex n=%ld out of sequence", n);
      }
      if (js.count("id")) {
	const StateName id = js.at("id");
	Assert (!id.is_number(), "id can't be a number");
	const string idStr = id.dump();
	if (id2n.count (idStr)) {
	  dupIds.insert (idStr);
	  Warn ("Duplicate state ID: %s", idStr.c_str());
	} else
	  id2n[idStr] = state.size();
	ms.name = id;
      }
      state.push_back (ms);
    }

    vguard<MachineState>::iterator msiter = state.begin();
    for (const json& js : jstate) {
      MachineState& ms = *msiter++;
      if (js.count ("trans")) {
	const json& jtrans = js.at("trans");
	Assert (jtrans.is_array(), "trans is not an array");
	for (const json& jt : jtrans) {
	  MachineTransition t;
	  const json& dest = jt.at("to");
	  if (dest.is_number())
	    t.dest = dest.get<StateIndex>();
	  else {
	    const string dstr = dest.dump();
	    Require (id2n.count(dstr), "No such state in \"to\": %s", dstr.c_str());
	    Require (!dupIds.count(dstr), "Ambiguous destination state ID in \"to\": %s", dstr.c_str());
	    t.dest = id2n.at (dstr);
	  }
	  if (jt.count("in"))
	    t.in = jt.at("in").get<string>();
	  if (jt.count("out"))
	    t.out = jt.at("out").get<string>();
	  t.weight = (jt.count("weight") ? WeightAlgebra::fromJson (jt.at("weight")) : WeightAlgebra::one());
	  ms.trans.push_back (t);
	}
      }
    }

    for (const auto& ms: state)
      for (const auto& t: ms.trans)
	Assert (t.dest < state.size(), "State %ld does not exist", t.dest);
  }
}

void Machine::writeDot (ostream& out, const char* emptyLabelText) const {
  out << "digraph G {\n";
  for (StateIndex s = 0; s < nStates(); ++s) {
    const auto& n = state[s].name;
    out << " " << s << " [label=\""
	<< escaped_str (n.is_string() ? n.get<string>() : n.dump())
	<< "\"];" << endl;
  }
  out << endl;
  for (StateIndex s = 0; s < nStates(); ++s) {
    const MachineState& ms = state[s];
    for (const auto& t: ms.trans)
      out << " " << s << " -> " << t.dest << " [headlabel=\""
	  << escaped_str (t.in.empty() ? emptyLabelText : t.in.c_str())
	  << "/"
	  << escaped_str (t.out.empty() ? emptyLabelText : t.out.c_str())
	  << "\""
	  << (WeightAlgebra::isOne(t.weight) ? string() : (string(",taillabel=\"") + WeightAlgebra::toString (t.weight, ParamDefs()) + "\""))
	  << "];" << endl;
    out << endl;
  }
  out << "}" << endl;
}

Machine Machine::projectOutputToInput() const {
  Assert (inputEmpty(), "Attempt to project output->input for transducer whose input is nonempty");
  Machine m (*this);
  for (auto& ms: m.state)
    for (auto& t: ms.trans)
      t.in = t.out;
  return m;
}

 Machine Machine::projectInputToOutput() const {
  Assert (outputEmpty(), "Attempt to project input->output for transducer whose output is nonempty");
  Machine m (*this);
  for (auto& ms: m.state)
    for (auto& t: ms.trans)
      t.out = t.in;
  return m;
}

Machine Machine::silenceOutput() const {
  Machine m (*this);
  for (auto& ms: m.state)
    for (auto& t: ms.trans)
      t.out.clear();
  return m;
}

Machine Machine::silenceInput() const {
  Machine m (*this);
  for (auto& ms: m.state)
    for (auto& t: ms.trans)
      t.in.clear();
  return m;
}

Machine Machine::weightInputs (const map<InputSymbol,WeightExpr>& w) const {
  Test (!inputEmpty(), "Redundant call to weightInputs(): input alphabet is empty");
  Machine m (*this);
  for (auto& ms: m.state)
    for (auto& t: ms.trans)
      if (!t.inputEmpty())
	t.weight = WeightAlgebra::multiply (t.weight, w.at (t.in));
  return m;
}

Machine Machine::weightOutputs (const map<OutputSymbol,WeightExpr>& w) const {
  Test (!outputEmpty(), "Redundant call to weightOutputs(): output alphabet is empty");
  Machine m (*this);
  for (auto& ms: m.state)
    for (auto& t: ms.trans)
      if (!t.outputEmpty())
	t.weight = WeightAlgebra::multiply (t.weight, w.at (t.out));
  return m;
}

Machine Machine::weightInputs (const string& macro) const {
  return weightInputs (WeightAlgebra::makeSymbolExprs (inputAlphabet(), macro));
}

Machine Machine::weightOutputs (const string& macro) const {
  return weightOutputs (WeightAlgebra::makeSymbolExprs (outputAlphabet(), macro));
}

Machine Machine::weightInputsGeometrically (const string& gp) const {
  return Machine::concatenate (weightInputs(gp), Machine::singleTransition (WeightAlgebra::negate (WeightAlgebra::fromJson (json::parse (gp)))));
}

Machine Machine::weightOutputsGeometrically (const string& gp) const {
  return Machine::concatenate (weightOutputs(gp), Machine::singleTransition (WeightAlgebra::negate (WeightAlgebra::fromJson (json::parse (gp)))));
}

Machine Machine::normalizeJointly() const {
  Machine m = *this;
  for (auto& ms: m.state) {
    WeightExpr norm = WeightAlgebra::zero();
    for (auto& t: ms.trans)
      norm = WeightAlgebra::add (norm, t.weight);
    for (auto& t: ms.trans)
      t.weight = WeightAlgebra::divide (t.weight, norm);
  }
  return m;
}

Machine Machine::normalizeConditionally() const {
  Machine m = *this;
  auto alph = m.inputAlphabet();
  alph.push_back (string());
  for (auto& ms: m.state) {
    for (const auto& inSym: alph) {
      WeightExpr norm = WeightAlgebra::zero();
      for (auto& t: ms.trans)
	if (t.in == inSym)
	  norm = WeightAlgebra::add (norm, t.weight);
      for (auto& t: ms.trans)
	if (t.in == inSym)
	  t.weight = WeightAlgebra::divide (t.weight, norm);
    }
  }
  return m;
}

Machine Machine::pointwiseReciprocal() const {
  Machine m (*this);
  for (auto& ms: m.state)
    for (auto& t: ms.trans)
      t.weight = WeightAlgebra::reciprocal (t.weight);
  return m;
}

bool Machine::inputEmpty() const {
  return inputAlphabet().empty();
}

bool Machine::outputEmpty() const{
  return outputAlphabet().empty();
}

bool Machine::isErgodicMachine() const {
  const auto acc = accessibleStates();
  return acc.size() == nStates() && acc.count (nStates() - 1);
}

bool Machine::isWaitingMachine() const {
  for (const auto& ms: state)
    if (!ms.waits() && !ms.continues())
      return false;
  return true;
}

size_t Machine::nBackTransitions() const {
  size_t n = 0;
  for (StateIndex s = 1; s < nStates(); ++s)
    for (const auto& t: state[s].trans)
      if (t.dest <= s)
	++n;
  return n;
}

size_t Machine::nSilentBackTransitions() const {
  size_t n = 0;
  for (StateIndex s = 1; s < nStates(); ++s)
    for (const auto& t: state[s].trans)
      if (t.isSilent() && t.dest <= s)
	++n;
  return n;
}

size_t Machine::nEmptyOutputBackTransitions() const {
  size_t n = 0;
  for (StateIndex s = 1; s < nStates(); ++s)
    for (const auto& t: state[s].trans)
      if (t.outputEmpty() && t.dest <= s)
	++n;
  return n;
}

bool Machine::isAdvancingMachine() const {
  for (StateIndex s = 1; s < nStates(); ++s)
    for (const auto& t: state[s].trans)
      if (t.isSilent() && t.dest <= s)
	return false;
  return true;
}

bool Machine::isDecodingMachine() const {
  for (StateIndex s = 1; s < nStates(); ++s)
    for (const auto& t: state[s].trans)
      if (t.outputEmpty() && t.dest <= s)
	return false;
  return true;
}

bool Machine::isToposortedMachine (bool excludeSelfLoops) const {
  for (StateIndex s = 1; s < nStates(); ++s)
    for (const auto& t: state[s].trans)
      if (excludeSelfLoops ? (t.dest <= s) : (t.dest < s))
	return false;
  return true;
}

inline StateIndex ij2compState (StateIndex i, StateIndex j, StateIndex jStates) {
  return i * jStates + j;
}

inline StateIndex compState2i (StateIndex comp, StateIndex jStates) {
  return comp / jStates;
}

inline StateIndex compState2j (StateIndex comp, StateIndex jStates) {
  return comp % jStates;
}

Machine Machine::compose (const Machine& first, const Machine& origSecond, bool assignStateNames, bool collapseDegenerateTransitions, SilentCycleStrategy cycleStrategy) {
  LogThisAt(3,"Composing " << first.nStates() << "-state transducer with " << origSecond.nStates() << "-state transducer" << endl);
  const Machine second = origSecond.isWaitingMachine() ? origSecond : origSecond.waitingMachine();
  Assert (second.isWaitingMachine(), "Attempt to compose transducers A*B where B is not a waiting machine");

  const StateIndex iStates = first.nStates(), jStates = second.nStates();
  assignStateNames = assignStateNames && !first.stateNamesAreAllNull() && !second.stateNamesAreAllNull();

  // first, a quick optimization hack to filter out inaccessible states
  LogThisAt(6,"Finding accessible states" << endl);
  vguard<bool> keep (iStates * jStates, false);
  vguard<StateIndex> toVisit, dest, keptState;
  keptState.reserve (iStates * jStates);
  toVisit.push_back(0);
  keep[0] = true;
  ProgressLog(plogAcc,6);
  plogAcc.initProgress ("Performing depth-first search of state space (max %lu states)", iStates*jStates);
  while (!toVisit.empty()) {
    plogAcc.logProgress (keptState.size() / (double) (iStates*jStates), "visited %lu states", keptState.size());
    const StateIndex c = toVisit.back();
    toVisit.pop_back();
    keptState.push_back(c);
    const StateIndex i = compState2i(c,jStates), j = compState2j(c,jStates);
    const MachineState& msi = first.state[i];
    const MachineState& msj = second.state[j];
    dest.clear();
    if (msj.waits() || msj.terminates()) {
      for (const auto& it: msi.trans)
	if (it.outputEmpty())
	  dest.push_back (ij2compState(it.dest,j,jStates));
	else
	  for (const auto& jt: msj.trans)
	    if (it.out == jt.in)
	      dest.push_back (ij2compState(it.dest,jt.dest,jStates));
    } else
      for (const auto& jt: msj.trans)
	dest.push_back (ij2compState(i,jt.dest,jStates));
    for (const StateIndex d: dest)
      if (!keep[d]) {
	keep[d] = true;
	toVisit.push_back(d);
      }
  }
  Assert (keep[iStates*jStates-1], "End state of composed machine is not accessible");

  LogThisAt(7,"Sorting & indexing " << keptState.size() << " states" << endl);
  sort (keptState.begin(), keptState.end());
  vguard<StateIndex> comp2kept (iStates * jStates);
  for (StateIndex k = 0; k < keptState.size(); ++k)
    comp2kept[keptState[k]] = k;

  // now do the composition for real
  Machine compMachine;
  compMachine.import (first, second);
  vguard<MachineState>& comp = compMachine.state;
  LogThisAt(7,"Initializing composite machine states" << endl);
  comp.resize (keptState.size());

  if (assignStateNames) {
    ProgressLog(plogName,6);
    plogName.initProgress ("Constructing namespace (%lu states)", keptState.size());
    for (StateIndex k = 0; k < keptState.size(); ++k) {
      const StateIndex c = keptState[k];
      const StateIndex i = compState2i(c,jStates), j = compState2j(c,jStates);
      const MachineState& msi = first.state[i];
      const MachineState& msj = second.state[j];
      plogName.logProgress (k / (double) keptState.size(), "state %ld/%ld", k, keptState.size());
      MachineState& ms = comp[k];
      ms.name = StateName ({first.state[i].name, second.state[j].name});
    }
  }

  ProgressLog(plogTrans,6);
  plogTrans.initProgress ("Computing transition weights (%lu states)", keptState.size());

  TransAccumulator ta;
  for (StateIndex k = 0; k < keptState.size(); ++k) {
    const StateIndex c = keptState[k];
    const StateIndex i = compState2i(c,jStates), j = compState2j(c,jStates);
    const MachineState& msi = first.state[i];
    const MachineState& msj = second.state[j];
    plogTrans.logProgress (k / (double) keptState.size(), "state %ld/%ld", k, keptState.size());
    MachineState& ms = comp[k];
    if (collapseDegenerateTransitions)
      ta.clear();
    else
      ta.transList = &ms.trans;
    if (msj.waits() || msj.terminates()) {
      for (const auto& it: msi.trans)
	if (it.outputEmpty()) {
	  const StateIndex d = ij2compState(it.dest,j,jStates);
	  if (keep[d])
	    ta.accumulate (it.in, string(), comp2kept[d], it.weight);
	} else
	  for (const auto& jt: msj.trans)
	    if (it.out == jt.in) {
	      const StateIndex d = ij2compState(it.dest,jt.dest,jStates);
	      if (keep[d])
		ta.accumulate (it.in, jt.out, comp2kept[d], WeightAlgebra::multiply (it.weight, jt.weight));
	    }
    } else
      for (const auto& jt: msj.trans) {
	const StateIndex d = ij2compState(i,jt.dest,jStates);
	if (keep[d])
	  ta.accumulate (string(), jt.out, comp2kept[d], jt.weight);
      }
    if (collapseDegenerateTransitions)
      ms.trans = ta.transitions();
  }

  LogThisAt(3,"Transducer composition yielded " << compMachine.nStates() << "-state machine" << endl);
  return compMachine.ergodicMachine().advanceSort().processCycles(cycleStrategy).ergodicMachine();
}

Machine Machine::intersect (const Machine& first, const Machine& origSecond, SilentCycleStrategy cycleStrategy) {
  LogThisAt(3,"Intersecting " << first.nStates() << "-state transducer with " << origSecond.nStates() << "-state transducer" << endl);
  Assert (first.outputAlphabet().empty() && origSecond.outputAlphabet().empty(), "Attempt to intersect transducers A&B with nonempty output alphabets");
  const Machine second = origSecond.isWaitingMachine() ? origSecond : origSecond.waitingMachine();
  Assert (second.isWaitingMachine(), "Attempt to intersect transducers A&B where B is not a waiting machine");

  Machine interMachine;
  interMachine.import (first, second);
  vguard<MachineState>& inter = interMachine.state;
  inter = vguard<MachineState> (first.nStates() * second.nStates());

  auto interState = [&](StateIndex i,StateIndex j) -> StateIndex {
    return i * second.nStates() + j;
  };

  const bool assignStateNames = !first.stateNamesAreAllNull() && !second.stateNamesAreAllNull();

  for (StateIndex i = 0; i < first.nStates(); ++i)
    for (StateIndex j = 0; j < second.nStates(); ++j) {
      MachineState& ms = inter[interState(i,j)];
      if (assignStateNames)
	ms.name = StateName ({first.state[i].name, second.state[j].name});
    }

  for (StateIndex i = 0; i < first.nStates(); ++i)
    for (StateIndex j = 0; j < second.nStates(); ++j) {
      MachineState& ms = inter[interState(i,j)];
      const MachineState& msi = first.state[i];
      const MachineState& msj = second.state[j];
      if (msj.waits() || msj.terminates()) {
	for (const auto& it: msi.trans)
	  if (it.inputEmpty())
	    ms.trans.push_back (MachineTransition (it.in, string(), interState(it.dest,j), it.weight));
	  else
	    for (const auto& jt: msj.trans)
	      if (it.in == jt.in)
		ms.trans.push_back (MachineTransition (it.in, string(), interState(it.dest,jt.dest), WeightAlgebra::multiply (it.weight, jt.weight)));
      } else
	for (const auto& jt: msj.trans)
	  ms.trans.push_back (MachineTransition (string(), string(), interState(i,jt.dest), jt.weight));
    }

  LogThisAt(3,"Transducer intersection yielded " << interMachine.nStates() << "-state machine" << endl);
  return interMachine.ergodicMachine().advanceSort().processCycles(cycleStrategy).ergodicMachine();
}

set<StateIndex> Machine::accessibleStates() const {
  vguard<bool> reachableFromStart (nStates(), false);
  deque<StateIndex> fwdQueue;
  fwdQueue.push_back (startState());
  reachableFromStart[fwdQueue.front()] = true;
  while (fwdQueue.size()) {
    const StateIndex c = fwdQueue.front();
    fwdQueue.pop_front();
    for (const auto& t: state[c].trans)
      if (!reachableFromStart[t.dest]) {
	reachableFromStart[t.dest] = true;
	fwdQueue.push_back (t.dest);
      }
  }

  vguard<bool> endReachableFrom (nStates(), false);
  vguard<vguard<StateIndex> > sources (nStates());
  deque<StateIndex> backQueue;
  for (StateIndex s = 0; s < nStates(); ++s)
    for (const auto& t: state[s].trans)
      sources[t.dest].push_back (s);
  backQueue.push_back (endState());
  endReachableFrom[backQueue.front()] = true;
  while (backQueue.size()) {
    const StateIndex c = backQueue.front();
    backQueue.pop_front();
    for (StateIndex src: sources[c])
      if (!endReachableFrom[src]) {
	endReachableFrom[src] = true;
	backQueue.push_back (src);
      }
  }

  set<StateIndex> as;
  for (StateIndex s = 0; s < nStates(); ++s)
    if (reachableFromStart[s] && endReachableFrom[s])
      as.insert (s);

  return as;
}

Machine Machine::ergodicMachine() const {
  Machine em;
  if (isErgodicMachine()) {
    em = *this;
    LogThisAt(5,"Machine is ergodic; no transformation necessary" << endl);
  } else {
    em.import (*this);
    vguard<bool> keep (nStates(), false);
    for (StateIndex s : accessibleStates())
      keep[s] = true;
    Assert (keep[nStates()-1], "End state is not accessible");
    
    map<StateIndex,StateIndex> nullEquiv;
    for (StateIndex s = 0; s < nStates(); ++s)
      if (keep[s]) {
	StateIndex d = s;
	while (state[d].trans.size() == 1 && state[d].trans.front().isSilent() && WeightAlgebra::isOne(state[d].trans.front().weight))
	  d = state[d].trans.front().dest;
	if (d != s)
	  nullEquiv[s] = d;
      }
    vguard<StateIndex> old2new (nStates());
    StateIndex ns = 0;
    for (StateIndex oldIdx = 0; oldIdx < nStates(); ++oldIdx)
      if (keep[oldIdx] && !nullEquiv.count(oldIdx))
	old2new[oldIdx] = ns++;
    for (StateIndex oldIdx = 0; oldIdx < nStates(); ++oldIdx)
      if (keep[oldIdx] && nullEquiv.count(oldIdx))
	old2new[oldIdx] = old2new[nullEquiv.at(oldIdx)];

    Assert (ns > 0, "Machine has no accessible states");

    em.state.reserve (ns);
    for (StateIndex oldIdx = 0; oldIdx < nStates(); ++oldIdx)
      if (keep[oldIdx] && !nullEquiv.count(oldIdx)) {
	em.state.push_back (MachineState());
	em.state.back().name = state[oldIdx].name;
	for (auto& t: state[oldIdx].trans)
	  if (keep[t.dest])
	    em.state.back().trans.push_back (MachineTransition (t.in, t.out, old2new.at(t.dest), t.weight));
      }

    Assert (em.isErgodicMachine(), "failed to create ergodic machine");
    LogThisAt(5,"Trimmed " << nStates() << "-state transducer into " << em.nStates() << "-state ergodic machine" << endl);
    LogThisAt(7,MachineLoader::toJsonString(em) << endl);
  }

  return em;
}

Machine Machine::waitingMachine (const char* waitTag, const char* continueTag) const {
  Machine wm;
  if (isWaitingMachine()) {
    wm = *this;
    LogThisAt(5,"Machine is already a waiting machine; no transformation necessary" << endl);
  } else {
    wm.import (*this);
    vguard<MachineState> newState (state);
    vguard<StateIndex> old2new (nStates()), new2old;
    for (StateIndex s = 0; s < nStates(); ++s) {
      const MachineState& ms = state[s];
      old2new[s] = new2old.size();
      new2old.push_back (s);
      if (!ms.waits() && !ms.continues()) {
	MachineState c, w;
	if (continueTag)
	  c.name[continueTag] = ms.name;
	else
	  c.name = ms.name;
	w.name[waitTag] = ms.name;
	for (const auto& t: ms.trans)
	  if (t.inputEmpty())
	    c.trans.push_back(t);
	  else
	    w.trans.push_back(t);
	c.trans.push_back (MachineTransition (string(), string(), newState.size(), WeightAlgebra::one()));
	old2new.push_back (new2old.size());
	new2old.push_back (newState.size());
	swap (newState[s], c);
	newState.push_back (w);
      }
    }
    for (StateIndex s: new2old) {
      MachineState& ms = newState[s];
      for (auto& t: ms.trans)
	t.dest = old2new[t.dest];
      wm.state.push_back (ms);
    }
    Assert (wm.isWaitingMachine(), "failed to create waiting machine");
    LogThisAt(5,"Converted " << nStates() << "-state transducer into " << wm.nStates() << "-state waiting machine" << endl);
    LogThisAt(7,MachineLoader::toJsonString(wm) << endl);
  }
  return wm;
}

// fwdTrans[i][jMin] = set of effective transitions { (i,j): i <= jMin <= j }
typedef map<StateIndex,map<StateIndex,TransList> > FwdTransMap;

// updateFwdTrans(machine,fwdTrans,i,newMin) populates fwdTrans[i][newMin]
void updateFwdTrans (const Machine& machine, FwdTransMap& fwdTrans, size_t& nElim, StateIndex i, StateIndex newMin) {
  if (!(fwdTrans.count(i) && fwdTrans.at(i).count(newMin))) {
    TransList oldTrans;
    if (newMin > i) {
      updateFwdTrans (machine, fwdTrans, nElim, i, newMin - 1);
      oldTrans = fwdTrans[i][newMin-1];
    } else if (newMin == i)
      oldTrans = machine.state[newMin].trans;

    TransList newFwdTrans;
    StateIndex newMinDest = machine.nStates();
    for (const auto& t_ij: oldTrans) {
      if (t_ij.isLoud())
	newFwdTrans.push_back(t_ij);
      else {
	const StateIndex j = t_ij.dest;
	if (j >= newMin) {
	  newMinDest = min (newMinDest, j);
	  newFwdTrans.push_back(t_ij);
	} else {
	  if (i != j)
	    updateFwdTrans (machine, fwdTrans, nElim, j, newMin);
	  for (const auto& t_jk: i == j ? oldTrans : fwdTrans[j][newMin]) {
	    const StateIndex k = t_jk.dest;
	    Assert (t_jk.isLoud() || (k>j && (k > i || (k == i && i == newMin))), "oops: cycle. i=%d j=%d k=%d", i, j, k);
	    newMinDest = min (newMinDest, k);
	    newFwdTrans.push_back (MachineTransition (t_jk.in, t_jk.out, k, WeightAlgebra::multiply (t_ij.weight, t_jk.weight)));
	  }
	  if (i > j)
	    ++nElim;
	}
      }
    }
    fwdTrans[i][newMin] = newFwdTrans;
  }
}

Machine Machine::processCycles (SilentCycleStrategy cycleStrategy) const {
  return (cycleStrategy == LeaveSilentCycles
	  ? *this
	  : (cycleStrategy == SumSilentCycles
	     ? advancingMachine()
	     : dropSilentBackTransitions()));
}

Machine Machine::dropSilentBackTransitions() const {
  Machine am;
  if (isAdvancingMachine()) {
    am = *this;
    LogThisAt(5,"Machine is already an advancing machine; no transformation necessary" << endl);
  } else {
    am.import (*this);
    if (nStates()) {
      am.state.reserve (nStates());
      for (StateIndex s = 0; s < nStates(); ++s) {
	const MachineState& ms = state[s];
	am.state.push_back (MachineState());
	MachineState& ams = am.state.back();
	ams.name = ms.name;

	for (const auto& t: ms.trans)
	  if (!(t.isSilent() && t.dest <= s))
	    ams.trans.push_back (t);
	  else
	    LogThisAt(6,"Dropping silent transition from #" << s << " to #" << t.dest << ": " << ms.name << "  -->  " << state[t.dest].name << endl);
      }

      Assert (am.isAdvancingMachine(), "failed to create advancing machine");
      LogThisAt(5,"Converted " << nTransitions() << "-transition transducer into " << am.nTransitions() << "-transition advancing machine by dropping silent back-transitions" << endl);
      LogThisAt(7,MachineLoader::toJsonString(am) << endl);
    }
  }
  return am;
}

Machine Machine::advancingMachine() const {
  Machine am;
  if (isAdvancingMachine()) {
    am = *this;
    LogThisAt(5,"Machine is already an advancing machine; no transformation necessary" << endl);
  } else {
    am.import (*this);
    if (nStates()) {
      am.state.reserve (nStates());

      FwdTransMap fwdTrans;
      size_t nElim = 0;

      const size_t totalElim = nSilentBackTransitions();

      ProgressLog(plogElim,6);
      plogElim.initProgress ("Eliminating backward silent transitions", totalElim);

      for (StateIndex s = 0; s < nStates(); ++s) {
	plogElim.logProgress (nElim / (double) totalElim, "%ld/%ld", nElim, totalElim);

	const MachineState& ms = state[s];
	am.state.push_back (MachineState());
	MachineState& ams = am.state.back();
	ams.name = ms.name;

	// recursive call to updateFwdTrans
	updateFwdTrans (*this, fwdTrans, nElim, s, s);

	// aggregate all transitions that go to the same place
	TransAccumulator ta;
	for (const auto& t: fwdTrans[s][s])
	  ta.accumulate (t.in, t.out, t.dest, t.weight);
	const auto et = ta.transitions();
	// factor out self-loops
	WeightExpr exitSelf = WeightAlgebra::one();
	for (const auto& t: et)
	  if (t.isSilent() && t.dest == s)
	    exitSelf = WeightAlgebra::geometricSum (t.weight);
	  else
	    ams.trans.push_back (t);
	if (!WeightAlgebra::isOne (exitSelf))
	  for (auto& t: ams.trans)
	    t.weight = WeightAlgebra::multiply (exitSelf, t.weight);
	fwdTrans[s][s] = ams.trans;
      }
      
      Assert (am.isAdvancingMachine(), "failed to create advancing machine");
      LogThisAt(5,"Converted " << nTransitions() << "-transition transducer into " << am.nTransitions() << "-transition advancing machine" << endl);
      LogThisAt(7,MachineLoader::toJsonString(am) << endl);
    }
  }
  return am;
}

Machine Machine::decodeSort() const {
  return advanceSort (&Machine::nEmptyOutputBackTransitions, &MachineTransition::outputEmpty, "non-outputting");
}

Machine Machine::encodeSort() const {
  return transpose().decodeSort().transpose();
}

bool isMachineTransition (const MachineTransition* mt) { return true; }
Machine Machine::toposort() const {
  return advanceSort (&Machine::nBackTransitions, &isMachineTransition, "general");
}

Machine Machine::advanceSort (function<size_t(const Machine*)> countBackTransitions,
			      function<bool(const MachineTransition*)> mustAdvance,
			      const char* sortType) const
{
  Machine result;
  const size_t nSilentBackBefore = countBackTransitions (this);
  if (nSilentBackBefore) {
    vguard<vguard<StateIndex> > silentIncoming (nStates()), silentOutgoing (nStates());
    vguard<int> nSilentIncoming (nStates()), nSilentOutgoing (nStates());
    for (StateIndex s = 1; s + 1 < nStates(); ++s) {
      const MachineState& ms = state[s];
      for (const auto& trans: ms.trans)
	if (mustAdvance(&trans) && trans.dest != s && trans.dest != endState() && trans.dest != startState()) {
	  silentOutgoing[s].push_back (trans.dest);
	  silentIncoming[trans.dest].push_back (s);
	  ++nSilentOutgoing[s];
	  ++nSilentIncoming[trans.dest];
	}
    }
    auto compareStates = [&] (StateIndex a, StateIndex b) {
      const int aIncoming = nSilentIncoming[a], aDiff = aIncoming - (int) nSilentOutgoing[a];
      const int bIncoming = nSilentIncoming[b], bDiff = bIncoming - (int) nSilentOutgoing[b];
      return (aIncoming == bIncoming
	      ? (aDiff == bDiff
		 ? (a < b)
		 : (aDiff < bDiff))
	      : (aIncoming < bIncoming));
    };
    
    vguard<StateIndex> order;
    vguard<bool> inOrder (nStates(), false);
    set<StateIndex, function<bool (StateIndex, StateIndex)> > queue (compareStates);
    auto removeState = [&] (StateIndex s) -> bool {
      auto iter = queue.find (s);
      const bool found = iter != queue.end() && *iter == s;
      if (found)
	queue.erase (iter);
      return found;
    };
    auto insertState = [&] (StateIndex s) {
      queue.insert (s);
    };
    auto addToOrder = [&] (StateIndex s) {
      order.push_back (s);
      inOrder[s] = true;
      for (const auto& next: silentOutgoing[s]) {
	const bool found = removeState (next);
	--nSilentIncoming[next];
	if (found)
	  insertState (next);
      }
      for (const auto& prev: silentIncoming[s]) {
	const bool found = removeState (prev);
	--nSilentOutgoing[prev];
	if (found)
	  insertState (prev);
      }
    };

    addToOrder (startState());
    if (nStates() > 1) {
      for (StateIndex s = 1; s + 1 < nStates(); ++s)
	queue.insert (s);
      ProgressLog(plogSort,6);
      plogSort.initProgress ("Advance-sorting %lu states", nStates() - 1);
      while (!queue.empty()) {
	plogSort.logProgress ((nStates() - queue.size()) / (double) nStates(), "sorted %lu states", nStates() - queue.size());
	StateIndex next = *queue.begin();
	queue.erase (queue.begin());
	addToOrder (next);
      }
      addToOrder (endState());
    }

    vguard<StateIndex> old2new (nStates());
    bool orderChanged = false;
    for (StateIndex n = 0; n < nStates(); ++n) {
      orderChanged = orderChanged || order[n] != n;
      old2new[order[n]] = n;
    }

    if (!orderChanged) {
      result = *this;
      LogThisAt(5,"Sorting left machine unchanged with " << nSilentBackBefore << " backward " << sortType << " transitions" << endl);
    } else {
      result.import (*this);
      result.state.reserve (nStates());
      for (const auto s: order) {
	result.state.push_back (state[s]);
	for (auto& trans: result.state.back().trans)
	  trans.dest = old2new[trans.dest];
      }
    }

    const size_t nSilentBackAfter = countBackTransitions (&result);
    if (nSilentBackAfter >= nSilentBackBefore) {
      if (orderChanged) {
	if (nSilentBackAfter > nSilentBackBefore)
	  LogThisAt(5,"Sorting increased number of " << sortType << " transitions from " << nSilentBackBefore << " to " << nSilentBackAfter << "; restoring original order" << endl);
	else
	  LogThisAt(5,"Sorting left number of backward " << sortType << " transitions unchanged at " << nSilentBackBefore << "; restoring original order" << endl);
	result = *this;
      }
    } else
      LogThisAt(5,"Sorting reduced number of backward " << sortType << " transitions from " << nSilentBackBefore << " to " << nSilentBackAfter << endl);

    if (nSilentBackAfter && !hasNullPaddingStates()) {
      LogThisAt(5,"Trying to sort again with \"dummy\" null start & end states..." << endl);
      const Machine withDummy = padWithNullStates();
      Assert (withDummy.hasNullPaddingStates(), "Dummy machine does not look like a dummy, triggering infinite dummification loop");
      const Machine sortedWithDummy = withDummy.advanceSort (countBackTransitions, mustAdvance);
      const size_t nSilentBackDummy = countBackTransitions (&sortedWithDummy);
      LogThisAt(5,"Padding with \"dummy\" null states " << (nSilentBackDummy < nSilentBackAfter ? (nSilentBackDummy ? "is better, though not perfect" : "worked!") : "failed") << endl);
      if (nSilentBackDummy < nSilentBackAfter)
	result = sortedWithDummy;
    }
    LogThisAt(7,"Sorted machine:" << endl << MachineLoader::toJsonString(result) << endl);
  } else {
    LogThisAt(5,"Machine has no backward " << sortType << " transitions; sort unnecessary" << endl);
    result = *this;
  }
  
  // show silent backward transitions
#define SilentBackwardLogLevel 9
  if (countBackTransitions (&result) > 0 && LoggingThisAt(SilentBackwardLogLevel)) {
    LogThisAt(SilentBackwardLogLevel,"Backward " << sortType << " transitions:" << endl);
    for (StateIndex s = 1; s < nStates(); ++s)
      for (const auto& t: result.state[s].trans)
	if (mustAdvance(&t) && t.dest <= s)
	  LogThisAt(SilentBackwardLogLevel,"[" << s << "," << result.state[s].name << endl << "," << t.dest << "," << result.state[t.dest].name << "]" << endl);
  }

  return result;
}

Machine Machine::padWithNullStates() const {
  bool hasNullStart = !state.empty() && state[0].trans.size() == 1 && state[0].exitsWithoutIO();
  const StateIndex ssi = startState();
  for (StateIndex s = 0; hasNullStart && s < nStates(); ++s)
    for (const auto& t: state[s].trans)
      if (t.dest == ssi) {
	hasNullStart = false;
	break;
      }
  const Machine dummy = Machine::null();
  Machine result = hasNullStart ? *this : Machine::concatenate (dummy, *this);
  return result.hasNullPaddingStates() ? result : Machine::concatenate (result, dummy);
}

bool Machine::hasNullPaddingStates() const {
  if (state.empty())
    return false;
  if (!(state[0].trans.size() == 1 && state[0].exitsWithoutIO()))
    return false;
  const StateIndex ssi = startState();
  const StateIndex esi = endState();
  if (!state[esi].trans.empty())
    return false;
  size_t nullToEnd = 0;
  for (const auto& ms: state)
    for (const auto& t: ms.trans) {
      if (t.dest == ssi)
	return false;
      if (t.dest == esi) {
	if (!t.isSilent())
	  return false;
	++nullToEnd;
      }
    }
  return nullToEnd == 1;
}

bool Machine::isAligningMachine() const {
  for (StateIndex s = 0; s < nStates(); ++s) {
    const MachineState& ms = state[s];
    map<StateIndex,map<InputSymbol,set<OutputSymbol> > > t;
    for (const auto& trans: ms.trans) {
      if (t[trans.dest][trans.in].count(trans.out))
	return false;
      t[trans.dest][trans.in].insert(trans.out);
    }
  }
  return true;
}

Machine Machine::eliminateRedundantStates() const {
  const Machine rm = isAdvancingMachine() ? *this : advanceSort();
  LogThisAt(3,"Eliminating redundant states from " << rm.nStates() << "-state transducer" << endl);
  vguard<StateIndex> eventualDest (rm.nStates());
  vguard<WeightExpr> exitMultiplier (rm.nStates(), WeightAlgebra::one());
  for (StateIndex s = rm.nStates(); s > 0; ) {
    --s;
    StateIndex t = s;
    WeightExpr mul = WeightAlgebra::one();
    while (t != rm.startState() && t != rm.endState() && rm.state[t].trans.size() == 1) {
      const auto& trans = rm.state[t].trans.front();
      if (!trans.isSilent())
	break;
      mul = WeightAlgebra::multiply (mul, trans.weight);
      t = trans.dest;
    }
    exitMultiplier[s] = mul;
    eventualDest[s] = t;
  }
  vguard<StateIndex> newStateIndex (rm.nStates()), oldStateIndex;
  oldStateIndex.reserve (rm.nStates());
  for (StateIndex s = 0; s < rm.nStates(); ++s)
    if (eventualDest[s] == s) {
      newStateIndex[s] = oldStateIndex.size();
      oldStateIndex.push_back (s);
    }
  for (StateIndex s = 0; s < rm.nStates(); ++s)
    if (eventualDest[s] != s)
      newStateIndex[s] = newStateIndex[eventualDest[s]];
  const StateIndex newStates = oldStateIndex.size();
  if (newStates == rm.nStates()) {
    LogThisAt(5,"No redundant states to eliminate" << endl);
    return rm;
  }
  Machine em;
  em.import (*this);
  em.state = vguard<MachineState> (newStates);
  for (StateIndex s = 0; s < newStates; ++s) {
    em.state[s] = rm.state[oldStateIndex[s]];
    for (auto& mt: em.state[s].trans) {
      mt.weight = WeightAlgebra::multiply (mt.weight, exitMultiplier[mt.dest]);
      mt.dest = newStateIndex[mt.dest];
    }
  }
  LogThisAt(5,"Eliminating redundant states turned a " << rm.nStates() << "-state machine into a " << em.nStates() << "-state machine" << endl);
  return em;
}

Machine Machine::eliminateSilentTransitions (SilentCycleStrategy cycleStrategy) const {
  if (!isAdvancingMachine())
    return processCycles(cycleStrategy).eliminateSilentTransitions();
  LogThisAt(3,"Eliminating silent transitions from " << nStates() << "-state transducer" << endl);
  Machine em;
  em.import (*this);
  if (nStates()) {
    em.state.resize (nStates());
    // Silent transitions from i->j are prepended to loud transitions from j->k.
    // In case there are no loud transitions from j->k (or if j is the end state), then the set of "unaccounted-for" outgoing silent transitions from i is stored,
    // and is then appended to loud transitions h->i (this has to be done in a second pass, because it can be the case that h>i).
    vguard<TransList> silentTrans (nStates());
    for (long long s = nStates() - 1; s >= 0; --s) {
      const MachineState& ms = state[s];
      MachineState& ems = em.state[s];
      ems.name = ms.name;
      TransAccumulator silent, loud;
      WeightExpr selfLoop = WeightAlgebra::zero();
      for (const auto& t: ms.trans)
	if (t.isSilent()) {
	  if (t.dest == s)
	    selfLoop = WeightAlgebra::add (selfLoop, t.weight);
	  else if (state[t.dest].terminates() || t.dest == nStates() - 1)
	    silent.accumulate(t);
	  else {
	    for (const auto& t2: silentTrans[t.dest])
	      silent.accumulate (t.in, t.out, t2.dest, WeightAlgebra::multiply(t.weight,t2.weight));
	    for (const auto& t2: em.state[t.dest].trans)
	      loud.accumulate (t2.in, t2.out, t2.dest, WeightAlgebra::multiply(t.weight,t2.weight));
	  }
	} else
	  loud.accumulate(t);
      ems.trans = loud.transitions();
      silentTrans[s] = silent.transitions();
      if (!WeightAlgebra::isZero (selfLoop)) {
	const WeightExpr selfExit = WeightAlgebra::geometricSum (selfLoop);
	for (auto& t: silentTrans[s])
	  t.weight = WeightAlgebra::multiply (selfExit, t.weight);
      }
    }
    for (MachineState& ems: em.state) {
      TransAccumulator loud;
      for (const auto& t: ems.trans) {
	loud.accumulate(t);
	for (const auto& t2: silentTrans[t.dest])
	  loud.accumulate (t.in, t.out, t2.dest, WeightAlgebra::multiply(t.weight,t2.weight));
      }
      ems.trans = loud.transitions();
    }
    em.state[0].trans.insert (em.state[0].trans.end(), silentTrans[0].begin(), silentTrans[0].end());
  }
  const Machine elimMachine = em.ergodicMachine();
  LogThisAt(3,"Elimination of silent transitions from " << nStates() << "-state, " << nTransitions() << "-transition machine yielded " << elimMachine.nStates() << "-state, " << elimMachine.nTransitions() << "-transition machine" << endl);
  return elimMachine;
}

Machine Machine::generator (const vguard<OutputSymbol>& seq, const string& name) {
  Machine m;
  m.state.resize (seq.size() + 1);
  for (SeqIdx pos = 0; pos <= seq.size(); ++pos)
    m.state[pos].name = json::array ({string(name), pos});
  for (SeqIdx pos = 0; pos < seq.size(); ++pos)
    m.state[pos].trans.push_back (MachineTransition (string(), seq[pos], pos + 1, WeightAlgebra::one()));
  return m;
}

Machine Machine::recognizer (const vguard<InputSymbol>& seq, const string& name) {
  Machine m;
  m.state.resize (seq.size() + 1);
  for (SeqIdx pos = 0; pos <= seq.size(); ++pos)
    m.state[pos].name = json::array ({string(name), pos});
  for (SeqIdx pos = 0; pos < seq.size(); ++pos)
    m.state[pos].trans.push_back (MachineTransition (seq[pos], string(), pos + 1, WeightAlgebra::one()));
  return m;
}

Machine Machine::echo (const vguard<OutputSymbol>& seq, const string& name) {
  return generator(seq,name).projectOutputToInput();
}

Machine Machine::wildGenerator (const vguard<OutputSymbol>& symbols) {
  Machine m;
  m.state.resize (1);
  m.state[0].name = json (symbols);
  for (const auto& sym: symbols)
    m.state[0].trans.push_back (MachineTransition (string(), sym, 0, WeightAlgebra::one()));
  return m;
}

Machine Machine::wildRecognizer (const vguard<InputSymbol>& symbols) {
  Machine m;
  m.state.resize (1);
  m.state[0].name = json (symbols);
  for (const auto& sym: symbols)
    m.state[0].trans.push_back (MachineTransition (sym, string(), 0, WeightAlgebra::one()));
  return m;
}

Machine Machine::wildEcho (const vguard<InputSymbol>& symbols) {
  Machine m;
  m.state.resize (1);
  m.state[0].name = json (symbols);
  for (const auto& sym: symbols)
    m.state[0].trans.push_back (MachineTransition (sym, sym, 0, WeightAlgebra::one()));
  return m;
}

Machine Machine::wildSingleGenerator (const vguard<OutputSymbol>& symbols) {
  Machine m;
  m.state.resize (2);
  m.state[0].name = json (symbols);
  m.state[1].name = MachineEndTag;
  for (const auto& sym: symbols)
    m.state[0].trans.push_back (MachineTransition (string(), sym, 1, WeightAlgebra::one()));
  return m;
}

Machine Machine::wildSingleRecognizer (const vguard<InputSymbol>& symbols) {
  Machine m;
  m.state.resize (2);
  m.state[0].name = json (symbols);
  m.state[1].name = MachineEndTag;
  for (const auto& sym: symbols)
    m.state[0].trans.push_back (MachineTransition (sym, string(), 1, WeightAlgebra::one()));
  return m;
}

Machine Machine::wildSingleEcho (const vguard<InputSymbol>& symbols) {
  Machine m;
  m.state.resize (2);
  m.state[0].name = json (symbols);
  m.state[1].name = MachineEndTag;
  for (const auto& sym: symbols)
    m.state[0].trans.push_back (MachineTransition (sym, sym, 1, WeightAlgebra::one()));
  return m;
}

Machine Machine::concatenate (const Machine& left, const Machine& right, const char* leftTag, const char* rightTag) {
  Assert (left.nStates() && right.nStates(), "Attempt to concatenate transducer with uninitialized transducer");
  Machine m (left);
  m.import (left, right);
  for (auto& ms: m.state)
    if (!ms.name.is_null())
      ms.name = json::array ({leftTag, ms.name});
  m.state.insert (m.state.end(), right.state.begin(), right.state.end());
  for (StateIndex s = left.state.size(); s < m.nStates(); ++s) {
    MachineState& ms = m.state[s];
    if (!ms.name.is_null())
      ms.name = json::array ({rightTag, m.state[s].name});
    for (auto& t: ms.trans)
      t.dest += left.state.size();
  }
  m.state[left.endState()].trans.push_back (MachineTransition (string(), string(), right.startState() + left.state.size(), WeightAlgebra::one()));
  return m;
}

Machine Machine::takeUnion (const Machine& first, const Machine& second) {
  return takeUnion (first, second, WeightAlgebra::one(), WeightAlgebra::one());
}

Machine Machine::takeUnion (const Machine& first, const Machine& second, const WeightExpr& pFirst) {
  return takeUnion (first, second, pFirst, WeightAlgebra::negate(pFirst));
}

Machine Machine::takeUnion (const Machine& first, const Machine& second, const WeightExpr& pFirst, const WeightExpr& pSecond) {
  Assert (first.nStates() && second.nStates(), "Attempt to find union of transducer with uninitialized transducer");
  Machine m;
  m.import (first, second);
  m.state.reserve (first.nStates() + second.nStates() + 2);
  m.state.push_back (MachineState());
  m.state.insert (m.state.end(), first.state.begin(), first.state.end());
  m.state.insert (m.state.end(), second.state.begin(), second.state.end());
  m.state.push_back (MachineState());
  for (StateIndex s = 0; s < first.nStates(); ++s) {
    MachineState& ms = m.state[s+1];
    if (!ms.name.is_null())
      ms.name = json::array ({"union-1", ms.name});
    for (auto& t: ms.trans)
      ++t.dest;
  }
  for (StateIndex s = 0; s < second.nStates(); ++s) {
    MachineState& ms = m.state[s+1+first.nStates()];
    if (!ms.name.is_null())
      ms.name = json::array ({"union-2", ms.name});
    for (auto& t: ms.trans)
      t.dest += 1 + first.nStates();
  }
  m.state[0].trans.push_back (MachineTransition (string(), string(), 1, pFirst));
  m.state[0].trans.push_back (MachineTransition (string(), string(), 1 + first.nStates(), pSecond));
  m.state[1 + first.endState()].trans.push_back (MachineTransition (string(), string(), m.endState(), WeightAlgebra::one()));
  m.state[1 + first.nStates() + second.endState()].trans.push_back (MachineTransition (string(), string(), m.endState(), WeightAlgebra::one()));
  return m;
}

Machine Machine::zeroOrOne (const Machine& q) {
  Assert (q.nStates(), "Attempt to quantify uninitialized transducer");
  Machine m (q);
  if (!m.state.back().terminates()) {
    for (auto& ms: m.state)
      if (!ms.name.is_null())
	ms.name = json::array ({"quant-main", ms.name});
    m.state.back().trans.push_back (MachineTransition (string(), string(), m.endState() + 1, WeightAlgebra::one()));
    m.state.push_back (MachineState());
    if (!q.stateNamesAreAllNull())
      m.state.back().name = json::array ({"quant-end"});
  }
  m.state[m.startState()].trans.push_back (MachineTransition (string(), string(), m.endState(), WeightAlgebra::one()));
  return m;
}

Machine Machine::kleenePlus (const Machine& k) {
  Assert (k.nStates(), "Attempt to form Kleene closure of uninitialized transducer");
  Machine m (k);
  m.state.insert (m.state.begin(), MachineState());
  if (!k.stateNamesAreAllNull())
    m.state.front().name = "kleene-plus";
  for (auto& ms: m.state)
    for (auto& t: ms.trans)
      ++t.dest;
  m.state[m.startState()].trans.push_back (MachineTransition (string(), string(), m.startState() + 1, WeightAlgebra::one()));
  m.state[m.endState()].trans.push_back (MachineTransition (string(), string(), m.startState() + 1, WeightAlgebra::one()));
  return m;
}

Machine Machine::kleeneStar (const Machine& k) {
  return zeroOrOne (kleenePlus (k));
}

Machine Machine::kleeneLoop (const Machine& main, const Machine& loop) {
  Assert (main.nStates(), "Attempt to form Kleene closure of uninitialized transducer");
  Assert (loop.nStates(), "Attempt to form Kleene closure with uninitialized loop transducer");
  const bool assignStateNames = !main.stateNamesAreAllNull() && !loop.stateNamesAreAllNull();
  Machine m (main);
  m.state.reserve (main.nStates() + loop.nStates() + 1);
  for (auto& ms: m.state)
    if (assignStateNames && !ms.name.is_null())
      ms.name = json::array ({"loop-main", ms.name});
  m.state.insert (m.state.end(), loop.state.begin(), loop.state.end());
  for (StateIndex s = main.nStates(); s < m.nStates(); ++s) {
    MachineState& ms = m.state[s];
    if (assignStateNames && !ms.name.is_null())
      ms.name = json::array ({"loop-continue", m.state[s].name});
    for (auto& t: ms.trans)
      t.dest += main.nStates();
  }
  m.state.push_back (MachineState());
  if (assignStateNames)
    m.state.back().name = json::array ({"loop-end"});
  m.state[main.endState()].trans.push_back (MachineTransition (string(), string(), main.nStates() + loop.startState(), WeightAlgebra::one()));
  m.state[main.endState()].trans.push_back (MachineTransition (string(), string(), m.endState(), WeightAlgebra::one()));
  m.state[main.nStates() + loop.endState()].trans.push_back (MachineTransition (string(), string(), m.startState(), WeightAlgebra::one()));
  return m;
}

Machine Machine::kleeneCount (const Machine& m, const string& countParam) {
  Machine result = kleeneStar (concatenate (singleTransition (WeightAlgebra::param (countParam)), m));
  result.funcs.defs[countParam] = WeightAlgebra::one();
  return result;
}

Machine Machine::reverse() const {
  Machine m;
  m.import (*this);
  m.state.resize (nStates());
  for (StateIndex s = 0; s < nStates(); ++s) {
    const StateIndex r = nStates() - 1 - s;
    const MachineState& ms = state[s];
    m.state[r].name = ms.name;
    for (const auto& t: ms.trans)
      m.state[nStates() - 1 - t.dest].trans.push_back (MachineTransition (t.in, t.out, r, t.weight));
  }
  return m;
}

Machine Machine::transpose() const {
  Machine m (*this);
  for (auto& ms: m.state)
    for (auto& t: ms.trans)
      swap (t.in, t.out);
  return m;
}

Machine Machine::null() {
  Machine n;
  n.state.push_back (MachineState());
  return n;
}

Machine Machine::singleTransition (const WeightExpr& weight) {
  Machine n;
  n.state.push_back (MachineState());
  n.state.push_back (MachineState());
  n.state[0].trans.push_back (MachineTransition (string(), string(), 1, weight));
  n.state[0].name = json("trans-start");
  n.state[1].name = json("trans-end");
  return n;
}

TransAccumulator::TransAccumulator() : transList (NULL)
{ }

void TransAccumulator::clear() {
  t.clear();
}

void TransAccumulator::accumulate (const MachineTransition& t) {
  accumulate (t.in, t.out, t.dest, t.weight);
}

void TransAccumulator::accumulate (InputSymbol in, OutputSymbol out, StateIndex dest, WeightExpr w) {
  if (transList)
    transList->push_back (MachineTransition (in, out, dest, w));
  else {
    if (t[dest][in].count(out))
      t[dest][in][out] = WeightAlgebra::add(w,t[dest][in][out]);
    else
      t[dest][in][out] = w;
  }
}

TransList TransAccumulator::transitions() const {
  TransList trans;
  for (const auto& dest_map: t)
    for (const auto& in_map: dest_map.second)
      for (const auto& out_weight: in_map.second)
	trans.push_back (MachineTransition (in_map.first, out_weight.first, dest_map.first, out_weight.second));
  return trans;
}

MachinePath::MachinePath() {}
MachinePath::MachinePath (const MachineTransition& mt) { trans.push_back (mt); }

MachinePath MachinePath::concatenate (const MachinePath& m) const {
  MachinePath result (*this);
  result.trans.insert (result.trans.end(), m.trans.begin(), m.trans.end());
  return result;
}

vguard<InputSymbol> MachinePath::inputSequence() const {
  return SeqPair::getInput (alignment());
}

vguard<OutputSymbol> MachinePath::outputSequence() const {
  return SeqPair::getOutput (alignment());
}

MachinePath::AlignPath MachinePath::alignment() const {
  return SeqPair::getAlignment (*this);
}

void MachinePath::writeJson (ostream& out, const Machine& m) const {
  out << "{\"start\":" << m.startState();
  if (!m.state[m.startState()].name.is_null())
    out << ",\"id\":" << m.state[m.startState()].name;
  out << ",\"trans\":[";
  size_t n = 0;
  for (const auto& t: trans) {
    out << (n++ ? "," : "")
	<< "{\"to\":" << t.dest;
    if (!m.state[t.dest].name.is_null())
      out << ",\"id\":" << m.state[t.dest].name;
    if (!t.inputEmpty())
      out << ",\"in\":\"" << escaped_str(t.in) << "\"";
    if (!t.outputEmpty())
      out << ",\"out\":\"" << escaped_str(t.out) << "\"";
    out << "}";
  }
  out << "]}";
}

void MachinePath::clear() { trans.clear(); }

MachineBoundPath::MachineBoundPath (const MachinePath& mp, const Machine& m)
  : MachinePath (mp), machine (m)
{ }

void MachineBoundPath::writeJson (ostream& out) const {
  MachinePath::writeJson (out, machine);
}

void Machine::import (const Machine& m, bool overwrite) {
  funcs = ParamAssign (funcs.combine (m.funcs, overwrite));
  cons = cons.combine (m.cons);
}

void Machine::import (const Machine& m1, const Machine& m2, bool overwrite) {
  import (m1, overwrite);
  import (m2, overwrite);
}

Params Machine::getParamDefs (bool assignDefaultValuesToMissingParams) const {
  Params p = funcs;
  if (assignDefaultValuesToMissingParams)
    p = cons.defaultParams().combine (p, true);
  return p;
}

bool Machine::stateNamesAreAllNull() const {
  for (const auto& ms: state)
    if (!ms.name.is_null())
      return false;
  return true;
}

Machine Machine::downsample (double maxProportionOfTransitionsToKeep, double minPostProbOfSelectedTransitions) const {
  Assert (isToposortedMachine(true), "Machine must be acyclic & topologically sorted before downsampling can take place");

  Machine null (*this);
  vguard<vguard<bool> > transAllowed;
  transAllowed.reserve (null.nStates());
  for (auto& ms: null.state) {
    for (auto& mt: ms.trans)
      mt.in = mt.out = string();
    transAllowed.push_back (vguard<bool> (ms.trans.size()));
  }

  const size_t nTransNull = null.nTransitions();
  LogThisAt(3,"Eliminating low-probability transitions from " << nTransNull << "-transition machine" << endl);
  
  const SeqPair emptySeqPair;
  const EvaluatedMachine eval (null, getParamDefs (true));
  const ForwardMatrix fwd (eval, emptySeqPair);
  const BackwardMatrix back (eval, emptySeqPair);

  size_t nTrans = 0;
  DPMatrix<IdentityIndexMapper>::TraceTerminator stopTrace = [&] (Envelope::InputIndex inPos, Envelope::OutputIndex outPos, StateIndex s, EvaluatedMachineState::TransIndex ti) {
    if (transAllowed[s][ti])
      return true;
    LogThisAt(8,"Adding transition #" << ti << " from state #" << s << endl);
    transAllowed[s][ti] = true;
    ++nTrans;
    return false;
  };

  BackwardMatrix::PostTransQueue queue = back.postTransQueue (fwd);
  const size_t nTransTarget = null.nTransitions() * maxProportionOfTransitionsToKeep;

  ProgressLog(plogTrace,6);
  plogTrace.initProgress ("Discarding lowest-probability transitions (threshold %g, max %lu)", minPostProbOfSelectedTransitions, nTransTarget);
  while (!queue.empty() && (nTrans == 0 || nTrans < nTransTarget)) {
    const BackwardMatrix::PostTrans pt = queue.top();
    plogTrace.logProgress (nTrans / (double) nTransTarget, "kept %lu transitions; next has weight %g", nTrans, pt.weight);
    if (pt.weight < minPostProbOfSelectedTransitions && nTrans > 0)
      break;
    queue.pop();
    const MachineTransition& mt = state[pt.src].getTransition (pt.transIndex);
    LogThisAt(7,"Tracing from " << pt.src << " -[" << mt.in << "/" << mt.out << "]-> " << mt.dest << " (transition #" << pt.transIndex << ", post.prob. " << pt.weight << "); " << nTrans << "/" << nTransTarget << " transitions" << endl);
    back.traceFrom (null, fwd, pt.inPos, pt.outPos, pt.src, pt.transIndex, stopTrace);
  }

  return subgraph (transAllowed);
}

Machine Machine::stochasticDownsample (mt19937& rng, double maxProportionOfTransitionsToKeep, int maxNumberOfPathsToSample) const {
  Assert (isToposortedMachine(true), "Machine must be acyclic & topologically sorted before stochastic downsampling can take place");

  Machine null (*this);
  vguard<vguard<bool> > transAllowed;
  transAllowed.reserve (null.nStates());
  for (auto& ms: null.state) {
    for (auto& mt: ms.trans)
      mt.in = mt.out = string();
    transAllowed.push_back (vguard<bool> (ms.trans.size()));
  }

  const size_t nTransNull = null.nTransitions();
  const size_t nTransTarget = null.nTransitions() * maxProportionOfTransitionsToKeep;
  LogThisAt(3,"Sampling transitions from " << nTransNull << "-transition machine" << endl);
  
  const SeqPair emptySeqPair;
  const EvaluatedMachine eval (null, getParamDefs (true));
  const ForwardMatrix fwd (eval, emptySeqPair);

  size_t nTrans = 0;
  MachinePath mp;
  DPMatrix<IdentityIndexMapper>::TraceTerminator neverStopTrace = [&] (Envelope::InputIndex inPos, Envelope::OutputIndex outPos, StateIndex s, EvaluatedMachineState::TransIndex ti) {
    if (!transAllowed[s][ti]) {
      LogThisAt(8,"Adding transition #" << ti << " from state #" << s << endl);
      transAllowed[s][ti] = true;
      ++nTrans;
    }
    mp.trans.push_front (null.state[s].getTransition (ti));
    return false;
  };

  ProgressLog(plogTrace,6);
  plogTrace.initProgress ("Sampling %d paths (max %lu transitions)", maxNumberOfPathsToSample, nTransTarget);
  ForwardMatrix::TransSelector selectRandomTrans = fwd.randomTransSelector (rng);
  for (size_t nPath = 0; nPath < maxNumberOfPathsToSample && nTrans < nTransTarget; ++nPath) {
    plogTrace.logProgress (max (nPath / (double) maxNumberOfPathsToSample, nTrans / (double) nTransTarget), "sampled %d paths, %lu transitions", nPath, nTrans);
    mp.clear();
    fwd.traceBack (null, fwd.inLen, fwd.outLen, null.endState(), neverStopTrace, selectRandomTrans);
    LogThisAt(7,JsonWriter<MachineBoundPath>::toJsonString (MachineBoundPath (mp, null)) << endl);
  }

  return subgraph (transAllowed);
}

Machine Machine::subgraph (const vguard<vguard<bool> >& transAllowed) const {
  Machine result (*this);
  for (StateIndex s = 0; s < nStates(); ++s) {
    const MachineState& ms = state[s];
    MachineState& rs = result.state[s];
    rs.trans.clear();
    EvaluatedMachineState::TransIndex ti;
    TransList::const_iterator iter;
    for (ti = 0, iter = ms.trans.begin(); iter != ms.trans.end(); ++ti, ++iter)
      if (transAllowed[s][ti])
	rs.trans.push_back (*iter);
  }
  LogThisAt(5,"Subgraph of " << nTransitions() << "-transition machine has " << result.nTransitions() << " transitions" << endl);
  return result.ergodicMachine().eliminateRedundantStates();
}

Machine Machine::stripNames() const {
  Machine m (*this);
  for (auto& ms: m.state)
    ms.name = nullptr;
  return m;
}
