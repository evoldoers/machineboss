#include <iomanip>
#include <fstream>
#include <set>
#include "trans.h"
#include "logger.h"
#include "json.hpp"

using json = nlohmann::json;

TransWeight WeightAlgebra::multiply (const TransWeight& l, const TransWeight& r) {
  TransWeight w;
  if (l.is_boolean() && l.get<bool>())
    w = r;
  else if (r.is_boolean() && r.get<bool>())
    w = l;
  else if (l.is_number_integer() && r.is_number_integer())
    w = l.get<int>() * r.get<int>();
  else if (l.is_number() && r.is_number())
    w = l.get<double>() * r.get<double>();
  else {
    TransWeight& m = w["*"];
    if (l.is_object() && l.count("*")) {
      m = l.at("*");
      if (r.is_object() && r.count("*"))
	m.insert (m.end(), r.at("*").begin(), r.at("*").end());
      else
	m.push_back (r);
    } else {
      if (r.is_object() && r.count("*")) {
	m = r.at("*");
	m.insert (m.begin(), l);
      } else
	m = TransWeight ({l, r});
    }
  }
  return w;
}

TransWeight WeightAlgebra::add (const TransWeight& l, const TransWeight& r) {
  TransWeight w;
  if (l.is_number_integer() && r.is_number_integer())
    w = l.get<int>() + r.get<int>();
  else if (l.is_number() && r.is_number())
    w = l.get<double>() + r.get<double>();
  else {
    TransWeight& s = w["+"];
    if (l.is_object() && l.count("+")) {
      s = l.at("+");
      if (r.is_object() && r.count("+"))
	s.insert (s.end(), r.at("+").begin(), r.at("+").end());
      else
	s.push_back (r);
    } else {
      if (r.is_object() && r.count("+")) {
	s = r.at("+");
	s.push_back (l);
      } else
	s = TransWeight ({l, r});
    }
  }
  return w;
}

MachineTransition::MachineTransition()
{ }

MachineTransition::MachineTransition (InputSymbol in, OutputSymbol out, State dest, TransWeight weight)
  : in (in),
  out (out),
  dest (dest),
  weight (weight)
{ }

bool MachineTransition::inputEmpty() const {
  return in == MachineNull;
}

bool MachineTransition::outputEmpty() const {
  return out == MachineNull;
}

bool MachineTransition::isSilent() const {
  return in == MachineNull && out == MachineNull;
}

bool MachineTransition::isLoud() const {
  return in != MachineNull || out != MachineNull;
}

MachineState::MachineState()
{ }

const MachineTransition* MachineState::transFor (InputSymbol in) const {
  for (const auto& t: trans)
    if (t.in == in)
      return &t;
  return NULL;
}

bool MachineState::terminates() const {
  return trans.empty();
}

bool MachineState::exitsWithInput() const {
  for (const auto& t: trans)
    if (t.in)
      return true;
  return false;
}

bool MachineState::exitsWithoutInput() const {
  for (const auto& t: trans)
    if (!t.in)
      return true;
  return false;
}

bool MachineState::waits() const {
  return !exitsWithoutInput();
}

bool MachineState::continues() const {
  return !exitsWithInput() && !terminates();
}

bool MachineState::isDeterministic() const {
  return trans.size() == 1 && trans.front().in == 0;
}

const MachineTransition& MachineState::next() const {
  Assert (isDeterministic(), "Called next() method on a non-deterministic state");
  return trans.front();
}

bool MachineState::exitsWithIO() const {
  for (const auto& t: trans)
    if (t.in || t.out)
      return true;
  return false;
}

bool MachineState::exitsWithoutIO() const {
  for (const auto& t: trans)
    if (!t.in && !t.out)
      return true;
  return false;
}

bool MachineState::isSilent() const {
  return !exitsWithIO();
}

bool MachineState::isLoud() const {
  return exitsWithIO() && !exitsWithoutIO();
}

State Machine::nStates() const {
  return state.size();
}

State Machine::startState() const {
  Assert (nStates() > 0, "Machine has no states");
  return 0;
}

State Machine::endState() const {
  Assert (nStates() > 0, "Machine has no states");
  return nStates() - 1;
}

Machine::Machine()
{ }

string Machine::inputAlphabet() const {
  set<char> alph;
  for (const auto& ms: state)
    for (const auto& t: ms.trans)
      if (!t.inputEmpty())
	alph.insert (t.in);
  return string (alph.begin(), alph.end());
}

string Machine::outputAlphabet() const {
  set<char> alph;
  for (const auto& ms: state)
    for (const auto& t: ms.trans)
      if (!t.outputEmpty())
	alph.insert (t.out);
  return string (alph.begin(), alph.end());
}

void Machine::writeJson (ostream& out) const {
  out << "{\"state\":" << endl;
  for (State s = 0; s < nStates(); ++s) {
    const MachineState& ms = state[s];
    out << (s == 0 ? " [" : "  ") << "{\"n\":" << s;
    if (!ms.name.is_null())
      out << "," << endl << "   \"id\":" << ms.name;
    if (ms.trans.size()) {
      out << "," << endl << "   \"trans\":[";
      for (size_t nt = 0; nt < ms.trans.size(); ++nt) {
	const MachineTransition& t = ms.trans[nt];
	if (nt > 0)
	  out << "," << endl << "            ";
	out << "{\"to\":" << t.dest;
	if (t.in) out << ",\"in\":\"" << t.in << "\"";
	if (t.out) out << ",\"out\":\"" << t.out << "\"";
	if (!(t.weight.is_boolean() && t.weight.get<bool>()))
	  out << ",\"weight\":" << t.weight;
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

string Machine::toJsonString() const {
  stringstream ss;
  writeJson (ss);
  return ss.str();
}

void Machine::readJson (istream& in) {
  state.clear();
  json pj;
  in >> pj;
  json jstate = pj.at("state");
  Assert (jstate.is_array(), "state is not an array");
  map<string,State> id2n;
  for (const json& js : jstate) {
    MachineState ms;
    if (js.count("n")) {
      const State n = js.at("n").get<State>();
      Require ((State) state.size() == n, "State n=%ld out of sequence", n);
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
	t.in = t.out = 0;
	const json& dest = jt.at("to");
	t.dest = dest.is_number()
	  ? dest.get<State>()
	  : id2n.at (dest.dump());
	if (jt.count("in")) {
	  const string tin = jt.at("in").get<string>();
	  Assert (tin.size() == 1, "Invalid input character: %s", tin.c_str());
	  t.in = tin[0];
	}
	if (jt.count("out")) {
	  const string tout = jt.at("out").get<string>();
	  Assert (tout.size() == 1, "Invalid output character: %s", tout.c_str());
	  t.out = tout[0];
	}
	t.weight = jt.count("weight") ? jt.at("weight") : TransWeight(true);
	ms.trans.push_back (t);
      }
    }
  }
}

Machine Machine::fromJson (istream& in) {
  Machine machine;
  machine.readJson (in);
  return machine;
}

Machine Machine::fromFile (const char* filename) {
  ifstream infile (filename);
  if (!infile)
    Fail ("File not found: %s", filename);
  return fromJson (infile);
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

bool Machine::isPunctuatedMachine() const {
  for (const auto& ms: state)
    if (!ms.isSilent() && !ms.isLoud())
      return false;
  return true;
}

bool Machine::isAdvancingMachine() const {
  for (State s = 1; s < nStates(); ++s)
    for (const auto& t: state[s].trans)
      if (t.isSilent() && t.dest <= s)
	return false;
  return true;
}

Machine Machine::compose (const Machine& first, const Machine& origSecond) {
  LogThisAt(3,"Composing " << first.nStates() << "-state transducer with " << origSecond.nStates() << "-state transducer" << endl);
  const Machine second = origSecond.isWaitingMachine() ? origSecond : origSecond.waitingMachine();
  Assert (second.isWaitingMachine(), "Attempt to compose transducers A*B where B is not a waiting machine");

  Machine compMachine;
  vguard<MachineState>& comp = compMachine.state;
  comp = vguard<MachineState> (first.nStates() * second.nStates());

  auto compState = [&](State i,State j) -> State {
    return i * second.nStates() + j;
  };
  for (State i = 0; i < first.nStates(); ++i)
    for (State j = 0; j < second.nStates(); ++j) {
      MachineState& ms = comp[compState(i,j)];
      ms.name = StateName ({first.state[i].name, second.state[j].name});
    }
  for (State i = 0; i < first.nStates(); ++i)
    for (State j = 0; j < second.nStates(); ++j) {
      MachineState& ms = comp[compState(i,j)];
      const MachineState& msi = first.state[i];
      const MachineState& msj = second.state[j];
      map<State,map<InputSymbol,map<OutputSymbol,TransWeight> > > t;
      auto accum = [&] (InputSymbol in, OutputSymbol out, State dest, TransWeight w) {
	LogThisAt(6,"Adding transition from " << ms.name << " to " << comp[dest].name << " with weight " << w << endl);
	if (t[dest][in].count(out))
	  t[dest][in][out] = WeightAlgebra::add(w,t[dest][in][out]);
	else
	  t[dest][in][out] = w;
      };
      if (msj.waits() || msj.terminates()) {
	for (const auto& it: msi.trans)
	  if (it.out == MachineNull)
	    accum (it.in, MachineNull, compState(it.dest,j), it.weight);
	  else
	    for (const auto& jt: msj.trans)
	      if (it.out == jt.in)
		accum (it.in, jt.out, compState(it.dest,jt.dest), WeightAlgebra::multiply (it.weight, jt.weight));
      } else
	for (const auto& jt: msj.trans)
	  accum (MachineNull, jt.out, compState(i,jt.dest), jt.weight);
      for (const auto& dest_map: t)
	for (const auto& in_map: dest_map.second)
	  for (const auto& out_weight: in_map.second)
	    ms.trans.push_back (MachineTransition (in_map.first, out_weight.first, dest_map.first, out_weight.second));
    }

  LogThisAt(8,"Intermediate machine:" << endl << compMachine.toJsonString());

  Machine finalMachine = compMachine.ergodicMachine();
  LogThisAt(3,"Transducer composition yielded " << finalMachine.nStates() << "-state machine; " << plural (compMachine.nStates() - finalMachine.nStates(), "more state was", "more states were") << " unreachable" << endl);

  return finalMachine;
}

set<State> Machine::accessibleStates() const {
  vguard<bool> reachableFromStart (nStates(), false);
  deque<State> fwdQueue;
  fwdQueue.push_back (startState());
  reachableFromStart[fwdQueue.front()] = true;
  while (fwdQueue.size()) {
    const State c = fwdQueue.front();
    fwdQueue.pop_front();
    for (const auto& t: state[c].trans)
      if (!reachableFromStart[t.dest]) {
	reachableFromStart[t.dest] = true;
	fwdQueue.push_back (t.dest);
      }
  }

  vguard<bool> endReachableFrom (nStates(), false);
  vguard<vguard<State> > sources (nStates());
  deque<State> backQueue;
  for (State s = 0; s < nStates(); ++s)
    for (const auto& t: state[s].trans)
      sources[t.dest].push_back (s);
  backQueue.push_back (endState());
  endReachableFrom[backQueue.front()] = true;
  while (backQueue.size()) {
    const State c = backQueue.front();
    backQueue.pop_front();
    for (State src: sources[c])
      if (!endReachableFrom[src]) {
	endReachableFrom[src] = true;
	backQueue.push_back (src);
      }
  }

  set<State> as;
  for (State s = 0; s < nStates(); ++s)
    if (reachableFromStart[s] && endReachableFrom[s])
      as.insert (s);

  return as;
}
  
Machine Machine::ergodicMachine() const {
  vguard<bool> keep (nStates(), false);
  for (State s : accessibleStates())
    keep[s] = true;

  map<State,State> nullEquiv;
  for (State s = 0; s < nStates(); ++s)
    if (keep[s]) {
      State d = s;
      while (state[d].trans.size() == 1 && state[d].trans.front().isSilent())
	d = state[d].trans.front().dest;
      if (d != s)
	nullEquiv[s] = d;
    }
  vguard<State> old2new (nStates());
  State ns = 0;
  for (State oldIdx = 0; oldIdx < nStates(); ++oldIdx)
    if (keep[oldIdx] && !nullEquiv.count(oldIdx))
      old2new[oldIdx] = ns++;
  for (State oldIdx = 0; oldIdx < nStates(); ++oldIdx)
    if (keep[oldIdx] && nullEquiv.count(oldIdx))
      old2new[oldIdx] = old2new[nullEquiv.at(oldIdx)];

  Machine em;
  em.state.reserve (ns);
  for (State oldIdx = 0; oldIdx < nStates(); ++oldIdx)
    if (keep[oldIdx] && !nullEquiv.count(oldIdx)) {
      em.state.push_back (MachineState());
      em.state.back().name = state[oldIdx].name;
      for (auto& t: state[oldIdx].trans)
	if (keep[t.dest])
	  em.state.back().trans.push_back (MachineTransition (t.in, t.out, old2new.at(t.dest), t.weight));
    }

  Assert (em.isErgodicMachine(), "failed to create ergodic machine");
  LogThisAt(5,"Converted " << nStates() << "-state transducer into " << em.nStates() << "-state ergodic machine" << endl);
  LogThisAt(7,em.toJsonString() << endl);

  return em;
}

Machine Machine::waitingMachine() const {
  vguard<MachineState> newState (state);
  vguard<State> old2new (nStates()), new2old;
  for (State s = 0; s < nStates(); ++s) {
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
      c.trans.push_back (MachineTransition (MachineNull, MachineNull, newState.size(), TransWeight(true)));
      old2new.push_back (new2old.size());
      new2old.push_back (newState.size());
      swap (newState[s], c);
      newState.push_back (w);
    }
  }
  Machine wm;
  for (State s: new2old) {
    MachineState& ms = newState[s];
    for (auto& t: ms.trans)
      t.dest = old2new[t.dest];
    wm.state.push_back (ms);
  }
  Assert (wm.isWaitingMachine(), "failed to create waiting machine");
  LogThisAt(5,"Converted " << nStates() << "-state transducer into " << wm.nStates() << "-state waiting machine" << endl);
  LogThisAt(7,wm.toJsonString() << endl);
  return wm;
}

Machine Machine::punctuatedMachine() const {
  vguard<MachineState> newState (state);
  vguard<State> old2new (nStates()), new2old;
  for (State s = 0; s < nStates(); ++s) {
    const MachineState& ms = state[s];
    old2new[s] = new2old.size();
    new2old.push_back (s);
    if (!ms.isSilent() && !ms.isLoud()) {
      MachineState loud, silent;
      loud.name = ms.name;
      silent.name[MachineSilentTag] = ms.name;
      for (const auto& t: ms.trans)
	if (t.isSilent())
	  silent.trans.push_back(t);
	else
	  loud.trans.push_back(t);
      silent.trans.push_back (MachineTransition (MachineNull, MachineNull, newState.size(), TransWeight(true)));
      old2new.push_back (new2old.size());
      new2old.push_back (newState.size());
      swap (newState[s], loud);
      newState.push_back (silent);
    }
  }
  Machine pm;
  for (State s: new2old) {
    MachineState& ms = newState[s];
    for (auto& t: ms.trans)
      t.dest = old2new[t.dest];
    pm.state.push_back (ms);
  }
  Assert (pm.isPunctuatedMachine(), "failed to create punctuated machine");
  LogThisAt(5,"Converted " << nStates() << "-state transducer into " << pm.nStates() << "-state punctuated machine" << endl);
  LogThisAt(7,pm.toJsonString() << endl);
  return pm;
}

Machine Machine::advancingMachine() const {
  Machine am;
  am.state.reserve (nStates());
  typedef map<State,MachineTransition> DestMap;
  map<State,DestMap> effDest;
  map<State,State> pos;
  for (State s = 0; s < nStates(); ++s) {
    const MachineState& ms = state[s];
    am.state.push_back (MachineState());
    MachineState& ams = am.state.back();
    ams.name = ms.name;
    DestMap dm;
    for (const auto& t : ms.trans)
      if (t.isLoud() || t.dest > s)
	ams.trans.push_back(t);
      else {
	// WRITE ME
      }
  }
  return am;
}

