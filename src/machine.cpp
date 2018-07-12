#include <iomanip>
#include <fstream>
#include <set>
#include <json.hpp>

#include "machine.h"
#include "fastseq.h"
#include "logger.h"
#include "schema.h"
#include "params.h"

using json = nlohmann::json;

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
      const auto tp = WeightAlgebra::params (t.weight, defs.defs);
      p.insert (tp.begin(), tp.end());
    }
  return p;
}

bool cmpRefCounts (const RefCount* a, const RefCount* b) {
  return a->order < b->order;
}

void Machine::writeJson (ostream& out, bool memoizeRepeatedExpressions, bool showParams) const {
  ExprMemos memo;
  ExprRefCounts counts;
  vguard<const RefCount*> common;
  map<string,string> name2def;
  vguard<string> names;
  if (memoizeRepeatedExpressions) {
    set<string> params;
    ParamDefs dummyDefs;
    for (StateIndex s = 0; s < nStates(); ++s)
      for (const auto& t: state[s].trans) {
	WeightAlgebra::countRefs (t.weight, counts);
	const auto tp = WeightAlgebra::params (t.weight, dummyDefs);
	params.insert (tp.begin(), tp.end());
      }

    for (const auto& t_c: counts) {
      const WeightExpr expr = t_c.second.expr;
      if (t_c.second.refs.size() > 1
	  && expr->type != Dbl && expr->type != Int && expr->type != Param && expr->type != Null
	  && !WeightAlgebra::isOne (expr))
	common.push_back (&t_c.second);
    }
    sort (common.begin(), common.end(), cmpRefCounts);

    map<string,string> def2name;
    size_t n = 0;
    for (const auto& c: common) {
      const string def = WeightAlgebra::toJsonString (c->expr, &memo);
      if (def2name.count(def))
	memo[c->expr] = def2name.at(def);
      else {
	string prefix, name;
	do {
	  prefix = prefix + "_";
	} while (params.count (name = prefix + to_string(++n)));
	memo[c->expr] = name;
	name2def[name] = def;
	def2name[def] = name;
	names.push_back (name);
      }
    }
  }

  out << "{\"state\":" << endl << " [";
  for (StateIndex s = 0; s < nStates(); ++s) {
    const MachineState& ms = state[s];
    out << (s ? "  " : "") << "{\"n\":" << s;
    if (!ms.name.is_null())
      out << "," << endl << "   \"id\":" << ms.name;
    if (ms.trans.size()) {
      out << "," << endl << "   \"trans\":[";
      size_t nt = 0;
      for (const auto& t: ms.trans) {
	if (nt++)
	  out << "," << endl << "            ";
	out << "{\"to\":" << t.dest;
	if (!t.inputEmpty()) out << ",\"in\":\"" << t.in << "\"";
	if (!t.outputEmpty()) out << ",\"out\":\"" << t.out << "\"";
	if (!WeightAlgebra::isOne (t.weight))
	  out << ",\"weight\":" << WeightAlgebra::toJsonString (t.weight, &memo);
	out << "}";
      }
      out << "]";
    }
    out << "}";
    if (s < nStates() - 1)
      out << "," << endl;
  }
  out << endl << " ]";
  if (names.size() || defs.defs.size()) {
    out << "," << endl << " \"defs\":";
    size_t count = 0;
    for (size_t n = 0; n < names.size(); ++n)
      out << ((count++) ? ",\n  " : "\n {")
	  << "\"" << names[n]
	  << "\":" << name2def[names[n]];
    for (const auto& def: defs.defs)
      out << ((count++) ? ",\n  " : "\n {")
	  << "\"" << def.first
	  << "\":" << WeightAlgebra::toJsonString (def.second, &memo);
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
    out << "," << endl << " \"cons\": ";
    cons.writeJson (out);
  } else
    out << endl;
  out << "}" << endl;
}

void Machine::readJson (const json& pj) {
  MachineSchema::validateOrDie ("machine", pj);

  if (pj.count("defs"))
    defs.readJson (pj.at("defs"));
  if (pj.count("cons"))
    cons.readJson (pj.at("cons"));
  // commented out code auto-binds all defined names to their values, no longer necessarily since we're explicitly preserving the names
  /*
  ParamFuncs funcs;
  if (pj.count("defs"))
    funcs.readJson (pj.at("defs"));
  for (auto& p_d: funcs.defs)
    defs.defs[p_d.first] = WeightAlgebra::bind (p_d.second, funcs.defs);
  */
  
  json jstate = pj.at("state");
  Assert (jstate.is_array(), "state is not an array");
  map<string,StateIndex> id2n;
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
      Require (!id2n.count(idStr), "Duplicate state %s", idStr.c_str());
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
	  t.dest = id2n.at (dstr);
	}
	if (jt.count("in"))
	  t.in = jt.at("in").get<string>();
	if (jt.count("out"))
	  t.out = jt.at("out").get<string>();
	t.weight = (jt.count("weight") ? WeightAlgebra::fromJson (jt.at("weight"), &defs.defs) : WeightAlgebra::one());
	ms.trans.push_back (t);
      }
    }
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
	  << "\",taillabel=\""
	  << WeightAlgebra::toString (t.weight, ParamDefs())
	  << "\"];" << endl;
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

bool Machine::inputEmpty() const {
  return inputAlphabet().empty();
}

bool Machine::outputEmpty() const{
  return outputAlphabet().empty();
}

bool Machine::isErgodicMachine() const {
  return accessibleStates().size() == nStates();
}

bool Machine::isWaitingMachine() const {
  for (const auto& ms: state)
    if (!ms.waits() && !ms.continues())
      return false;
  return true;
}

size_t Machine::nSilentBackTransitions() const {
  size_t n = 0;
  for (StateIndex s = 1; s < nStates(); ++s)
    for (const auto& t: state[s].trans)
      if (t.isSilent() && t.dest <= s)
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

inline StateIndex ij2compState (StateIndex i, StateIndex j, StateIndex jStates) {
  return i * jStates + j;
}

inline StateIndex compState2i (StateIndex comp, StateIndex jStates) {
  return comp / jStates;
}

inline StateIndex compState2j (StateIndex comp, StateIndex jStates) {
  return comp % jStates;
}

Machine Machine::compose (const Machine& first, const Machine& origSecond, bool assignCompositeStateNames, bool collapseDegenerateTransitions) {
  LogThisAt(3,"Composing " << first.nStates() << "-state transducer with " << origSecond.nStates() << "-state transducer" << endl);
  const Machine second = origSecond.isWaitingMachine() ? origSecond : origSecond.waitingMachine();
  Assert (second.isWaitingMachine(), "Attempt to compose transducers A*B where B is not a waiting machine");

  const StateIndex iStates = first.nStates(), jStates = second.nStates();

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

  if (assignCompositeStateNames) {
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
  return compMachine.ergodicMachine().advanceSort().advancingMachine().ergodicMachine();
}

Machine Machine::intersect (const Machine& first, const Machine& origSecond) {
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
  for (StateIndex i = 0; i < first.nStates(); ++i)
    for (StateIndex j = 0; j < second.nStates(); ++j) {
      MachineState& ms = inter[interState(i,j)];
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
  return interMachine.ergodicMachine().advanceSort().advancingMachine().ergodicMachine();
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

Machine Machine::waitingMachine() const {
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
	c.name = ms.name;
	w.name[MachineWaitTag] = ms.name;
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

Machine Machine::advancingMachine() const {
  Machine am;
  if (isAdvancingMachine()) {
    am = *this;
    LogThisAt(5,"Machine is already an advancing machine; no transformation necessary" << endl);
  } else {
    am.import (*this);
    if (nStates()) {
      am.state.reserve (nStates());

      // fwdTrans[i][jMin] = set of effective transitions { (i,j): i <= jMin <= j }
      map<StateIndex,map<StateIndex,TransList> > fwdTrans;
      size_t nElim = 0;

      // updateFwdTrans(i,newMin) calculates fwdTrans[i][newMin]
      function<void(StateIndex,StateIndex)> updateFwdTrans = [&](StateIndex i, StateIndex newMin) {
	if (!(fwdTrans.count(i) && fwdTrans.at(i).count(newMin))) {
	  TransList oldTrans;
	  if (newMin > i) {
	    updateFwdTrans (i, newMin - 1);
	    oldTrans = fwdTrans[i][newMin-1];
	  } else if (newMin == i)
	    oldTrans = state[newMin].trans;

	  TransList newFwdTrans;
	  StateIndex newMinDest = nStates();
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
		  updateFwdTrans (j, newMin);
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
      };

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
	updateFwdTrans(s,s);

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

Machine Machine::advanceSort() const {
  Machine result;
  const size_t nSilentBackBefore = nSilentBackTransitions();
  if (nSilentBackBefore) {
    vguard<set<StateIndex> > silentIncoming (nStates()), silentOutgoing (nStates());
    for (StateIndex s = 0; s < nStates(); ++s) {
      const MachineState& ms = state[s];
      for (const auto& trans: ms.trans)
	if (trans.isSilent() && trans.dest != s && trans.dest != endState()) {
	  silentOutgoing[s].insert (trans.dest);
	  silentIncoming[trans.dest].insert (s);
	}
    }
    vguard<StateIndex> order;
    vguard<bool> inOrder (nStates(), false);
    auto addToOrder = [&] (StateIndex s) {
      order.push_back (s);
      inOrder[s] = true;
      for (const auto& next: silentOutgoing[s])
	if (next != s)
	  silentIncoming[next].erase(s);
      for (const auto& prev: silentIncoming[s])
	if (prev != s)
	  silentOutgoing[prev].erase(s);
    };
    addToOrder (startState());
    if (nStates() > 1) {
      list<StateIndex> queue;
      for (StateIndex s = 1; s + 1 < nStates(); ++s)
	queue.push_back (s);
      while (queue.size()) {
	// find lowest-numbered state with no incoming (silent) transitions, or (failing that) largest difference between incoming & outgoing
	// if more than one state has no incoming transitions, the tiebreaker is the largest incoming-outgoing difference
	list<StateIndex>::iterator next = queue.end();
	int nextIncoming, nextDiff;
	for (auto iter = queue.begin(); iter != queue.end(); ++iter) {
	  const int sIncoming = (int) silentIncoming[*iter].size(), sDiff = sIncoming - (int) silentOutgoing[*iter].size();
	  if (next == queue.end() || (sIncoming == 0 ? (nextIncoming > 0 || sDiff < nextDiff) : (nextIncoming > 0 && sDiff < nextDiff))) {
	    next = iter;
	    nextIncoming = sIncoming;
	    nextDiff = sDiff;
	  }
	}
	addToOrder (*next);
	queue.erase (next);
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
      LogThisAt(5,"Sorting left machine unchanged with " << nSilentBackBefore << " backward silent transitions" << endl);
    } else {
      result.import (*this);
      result.state.reserve (nStates());
      for (const auto s: order) {
	result.state.push_back (state[s]);
	for (auto& trans: result.state.back().trans)
	  trans.dest = old2new[trans.dest];
      }
    
      const size_t nSilentBackAfter = result.nSilentBackTransitions();
      Assert (nSilentBackAfter <= nSilentBackBefore, "Sorting increased number of silent backward transitions from %u to %u", nSilentBackBefore, nSilentBackAfter);
      if (nSilentBackAfter == nSilentBackBefore) {
	result = *this;
	LogThisAt(5,"Sorting left number of backward silent transitions unchanged at " << nSilentBackBefore << "; restoring original order" << endl);
      } else
	LogThisAt(5,"Sorting reduced number of backward silent transitions from " << nSilentBackBefore << " to " << nSilentBackAfter << endl);
      LogThisAt(7,"Sorted machine:" << endl << MachineLoader::toJsonString(result) << endl);
    }
  } else {
    LogThisAt(5,"Machine has no backward silent transitions; sort unnecessary" << endl);
    result = *this;
  }
  
  // show silent backward transitions
#define SilentBackwardLogLevel 9
  if (result.nSilentBackTransitions() > 0 && LoggingThisAt(SilentBackwardLogLevel)) {
    LogThisAt(SilentBackwardLogLevel,"Silent backward transitions:" << endl);
    for (StateIndex s = 1; s < nStates(); ++s)
      for (const auto& t: state[s].trans)
	if (t.isSilent() && t.dest <= s)
	  LogThisAt(SilentBackwardLogLevel,"[" << s << "," << state[s].name << endl << "," << t.dest << "," << state[t.dest].name << "]" << endl);
  }

  return result;
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

Machine Machine::eliminateSilentTransitions() const {
  if (!isAdvancingMachine())
    return advancingMachine().eliminateSilentTransitions();
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
      for (const auto& t: ms.trans)
	if (t.isSilent()) {
	  if (state[t.dest].terminates() || t.dest == nStates() - 1)
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

Machine Machine::generator (const string& name, const vguard<OutputSymbol>& seq) {
  Machine m;
  m.state.resize (seq.size() + 1);
  for (SeqIdx pos = 0; pos <= seq.size(); ++pos)
    m.state[pos].name = json::array ({string(name), pos});
  for (SeqIdx pos = 0; pos < seq.size(); ++pos)
    m.state[pos].trans.push_back (MachineTransition (string(), seq[pos], pos + 1, WeightAlgebra::one()));
  return m;
}

Machine Machine::acceptor (const string& name, const vguard<InputSymbol>& seq) {
  Machine m;
  m.state.resize (seq.size() + 1);
  for (SeqIdx pos = 0; pos <= seq.size(); ++pos)
    m.state[pos].name = json::array ({string(name), pos});
  for (SeqIdx pos = 0; pos < seq.size(); ++pos)
    m.state[pos].trans.push_back (MachineTransition (seq[pos], string(), pos + 1, WeightAlgebra::one()));
  return m;
}

Machine Machine::concatenate (const Machine& left, const Machine& right) {
  Assert (left.nStates() && right.nStates(), "Attempt to concatenate transducer with uninitialized transducer");
  Machine m (left);
  m.import (left, right);
  for (auto& ms: m.state)
    if (!ms.name.is_null())
      ms.name = json::array ({"concat-l", ms.name});
  m.state.insert (m.state.end(), right.state.begin(), right.state.end());
  for (StateIndex s = left.state.size(); s < m.nStates(); ++s) {
    MachineState& ms = m.state[s];
    if (!ms.name.is_null())
      ms.name = json::array ({"concat-r", m.state[s].name});
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
    m.state.push_back (MachineState());
    m.state.back().name = json::array ({"quant-end"});
  }
  m.state[m.startState()].trans.push_back (MachineTransition (string(), string(), m.endState(), WeightAlgebra::one()));
  return m;
}

Machine Machine::kleenePlus (const Machine& k) {
  Assert (k.nStates(), "Attempt to form Kleene closure of uninitialized transducer");
  Machine m (k);
  m.state[k.endState()].trans.push_back (MachineTransition (string(), string(), m.startState(), WeightAlgebra::one()));
  return m;
}

Machine Machine::kleeneStar (const Machine& k) {
  return zeroOrOne (kleenePlus (k));
}

Machine Machine::kleeneLoop (const Machine& main, const Machine& loop) {
  Assert (main.nStates(), "Attempt to form Kleene closure of uninitialized transducer");
  Assert (loop.nStates(), "Attempt to form Kleene closure with uninitialized loop transducer");
  Machine m (main);
  m.state.reserve (main.nStates() + loop.nStates() + 1);
  for (auto& ms: m.state)
    if (!ms.name.is_null())
      ms.name = json::array ({"loop-main", ms.name});
  m.state.insert (m.state.end(), loop.state.begin(), loop.state.end());
  for (StateIndex s = main.nStates(); s < m.nStates(); ++s) {
    MachineState& ms = m.state[s];
    if (!ms.name.is_null())
      ms.name = json::array ({"loop-continue", m.state[s].name});
    for (auto& t: ms.trans)
      t.dest += main.nStates();
  }
  m.state.push_back (MachineState());
  m.state.back().name = json::array ({"loop-end"});
  m.state[main.endState()].trans.push_back (MachineTransition (string(), string(), main.nStates() + loop.startState(), WeightAlgebra::one()));
  m.state[main.endState()].trans.push_back (MachineTransition (string(), string(), m.endState(), WeightAlgebra::one()));
  m.state[main.nStates() + loop.endState()].trans.push_back (MachineTransition (string(), string(), m.startState(), WeightAlgebra::one()));
  return m;
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

Machine Machine::flipInOut() const {
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

void MachinePath::writeJson (ostream& out) const {
  out << "{\"start\":0,\"trans\":[";
  size_t n = 0;
  for (const auto& t: trans) {
    out << (n++ ? "," : "")
	<< "{\"to\":" << t.dest;
    if (!t.inputEmpty())
      out << ",\"in\":\"" << t.in << "\"";
    if (!t.outputEmpty())
      out << ",\"out\":\"" << t.out;
    out << "\"}";
  }
  out << "]}";
}

void Machine::import (const Machine& m) {
  defs = ParamFuncs (defs.combine (m.defs));
  cons = cons.combine (m.cons);
}

void Machine::import (const Machine& m1, const Machine& m2) {
  import (m1);
  import (m2);
}
