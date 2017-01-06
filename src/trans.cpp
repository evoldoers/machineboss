#include <iomanip>
#include <fstream>
#include "trans.h"
#include "logger.h"
#include "json.hpp"

using json = nlohmann::json;

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

bool MachineTransition::isNull() const {
  return in == MachineNull && out == MachineNull;
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

bool MachineState::exitsWithInput (const char* symbols) const {
  for (const auto& t: trans)
    if (t.in && strchr (symbols, t.in) != NULL)
      return true;
  return false;
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
  return exitsWithInput() && !exitsWithoutInput();
}

bool MachineState::jumps() const {
  return !exitsWithInput() && exitsWithoutInput();
}

bool MachineState::emitsOutput() const {
  for (const auto& t: trans)
    if (t.out)
      return true;
  return false;
}

bool MachineState::isDeterministic() const {
  return trans.size() == 1 && trans.front().in == 0;
}

const MachineTransition& MachineState::next() const {
  Assert (isDeterministic(), "Called next() method on a non-deterministic state");
  return trans.front();
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

void Machine::writeDot (ostream& out) const {
  out << "digraph G {" << endl;
  for (State s = 0; s < nStates(); ++s) {
    const MachineState& ms = state[s];
    out << " " << s << " [label=\"" << ms.name << "\"];" << endl;
  }
  out << endl;
  for (State s = 0; s < nStates(); ++s) {
    const MachineState& ms = state[s];
    for (const auto& t: ms.trans) {
      out << " " << s << " -> " << t.dest << " [label=\"";
      if (!t.inputEmpty())
	out << t.in;
      out << "/";
      if (!t.outputEmpty())
	out << t.out;
      out << "\"];" << endl;
    }
    out << endl;
  }
  out << "}" << endl;
}

void Machine::write (ostream& out) const {
  const size_t iw = stateIndexWidth();
  const size_t nw = stateNameWidth();
  for (State s = 0; s < nStates(); ++s) {
    const MachineState& ms = state[s];
    out << setw(iw+1) << left << stateIndex(s)
	<< setw(nw+1) << left << ms.name;
    for (const auto& t: ms.trans) {
      out << " ";
      if (!t.inputEmpty())
	out << t.in;
      out << "/";
      if (!t.outputEmpty())
	out << t.out;
      out << "->" << stateIndex(t.dest)
	  << " " << t.weight;
    }
    out << endl;
  }
}

string Machine::stateIndex (State s) {
  return string("#") + to_string(s);
}

size_t Machine::stateNameWidth() const {
  size_t w = 0;
  for (const auto& ms: state)
    w = max (w, ms.name.size());
  return w;
}

size_t Machine::stateIndexWidth() const {
  size_t w = 0;
  for (State s = 0; s < nStates(); ++s)
    w = max (w, stateIndex(s).size());
  return w;
}

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
  out << "{\"state\": [" << endl;
  for (State s = 0; s < nStates(); ++s) {
    const MachineState& ms = state[s];
    out << " {\"n\":" << s << ",";
    if (ms.name.size())
	out << "\"id\":\"" << ms.name << "\",";
    out << "\"trans\":[";
    for (size_t nt = 0; nt < ms.trans.size(); ++nt) {
      const MachineTransition& t = ms.trans[nt];
      if (nt > 0) out << ",";
      out << "{";
      if (t.in) out << "\"in\":\"" << t.in << "\",";
      if (t.out) out << "\"out\":\"" << t.out << "\",";
      out << "\"to\":" << t.dest
	  << ",\"weight\":" << t.weight
	  << "}";
    }
    out << "]}";
    if (s < nStates() - 1)
      out << ",";
    out << endl;
  }
  out << "]}" << endl;
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
      const string id = js.at("id").get<string>();
      Require (!id2n.count(id), "Duplicate state %s", id.c_str());
      id2n[id] = state.size();
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
	  : id2n.at (dest.get<string>());
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
	t.weight = jt.count("weight") ? jt.at("weight").get<double>() : 1.;
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

bool Machine::isWaitingMachine() const {
  for (const auto& ms: state)
    if (!ms.waits() && !ms.jumps() && !ms.terminates())
      return false;
  return true;
}

Machine Machine::compose (const Machine& first, const Machine& origSecond) {
  LogThisAt(3,"Composing " << first.nStates() << "-state transducer with " << origSecond.nStates() << "-state transducer" << endl);
  const Machine second = origSecond.isWaitingMachine() ? origSecond : origSecond.waitingMachine();
  Assert (second.isWaitingMachine(), "Attempt to compose transducers A*B where B is not a waiting machine");

  Machine tmpMachine;
  vguard<MachineState>& comp = tmpMachine.state;
  comp = vguard<MachineState> (first.nStates() * second.nStates());

  auto compState = [&](State i,State j) -> State {
    return i * second.nStates() + j;
  };
  auto compStateName = [&](State i,State j) -> string {
    return string("(") + first.state[i].name + "," + second.state[j].name + ")";
  };
  for (State i = 0; i < first.nStates(); ++i)
    for (State j = 0; j < second.nStates(); ++j) {
      MachineState& ms = comp[compState(i,j)];
      const MachineState& msi = first.state[i];
      const MachineState& msj = second.state[j];
      ms.name = compStateName(i,j);
      if (msj.waits() || msj.terminates()) {
	for (const auto& it: msi.trans)
	  if (it.out == MachineNull) {
	    ms.trans.push_back (MachineTransition (it.in, MachineNull, compState(it.dest,j), it.weight));
	    LogThisAt(6,"Adding transition from " << ms.name << " to " << compStateName(it.dest,j) << endl);
	  } else
	    for (const auto& jt: msj.trans)
	      if (it.out == jt.in) {
		ms.trans.push_back (MachineTransition (it.in, jt.out, compState(it.dest,jt.dest), it.weight * jt.weight));
		LogThisAt(6,"Adding transition from " << ms.name << " to " << compStateName(it.dest,jt.dest) << endl);
	      }
      } else
	for (const auto& jt: msj.trans) {
	  ms.trans.push_back (MachineTransition (MachineNull, jt.out, compState(i,jt.dest), jt.weight));
	  LogThisAt(6,"Adding transition from " << ms.name << " to " << compStateName(i,jt.dest) << endl);
	}
    }

  LogThisAt(8,"Intermediate machine:" << endl << tmpMachine.toJsonString());

  vguard<bool> reachableFromStart (comp.size(), false);
  deque<State> queue;
  queue.push_back (compState(first.startState(),second.startState()));
  reachableFromStart[queue.front()] = true;
  while (queue.size()) {
    const State c = queue.front();
    queue.pop_front();
    for (const auto& t: comp[c].trans)
      if (!reachableFromStart[t.dest]) {
	reachableFromStart[t.dest] = true;
	queue.push_back (t.dest);
      }
  }

  vguard<bool> endReachableFrom (comp.size(), false);
  vguard<vguard<State> > sources (comp.size());
  for (State s = 0; s < comp.size(); ++s)
    for (const auto& t: comp[s].trans)
      sources[t.dest].push_back (s);
  queue.push_back (compState(first.nStates()-1,second.nStates()-1));
  endReachableFrom[queue.front()] = true;
  while (queue.size()) {
    const State c = queue.front();
    queue.pop_front();
    for (State src: sources[c])
      if (!endReachableFrom[src]) {
	endReachableFrom[src] = true;
	queue.push_back (src);
      }
  }

  map<State,State> nullEquiv;
  for (State s = 0; s < comp.size(); ++s)
    if (reachableFromStart[s] && endReachableFrom[s]) {
      State d = s;
      while (comp[d].trans.size() == 1 && comp[d].trans.front().isNull())
	d = comp[d].trans.front().dest;
      if (d != s)
	nullEquiv[s] = d;
    }
  vguard<State> old2new (comp.size());
  State nStates = 0;
  for (State oldIdx = 0; oldIdx < comp.size(); ++oldIdx)
    if (reachableFromStart[oldIdx] && endReachableFrom[oldIdx] && !nullEquiv.count(oldIdx))
      old2new[oldIdx] = nStates++;
  for (State oldIdx = 0; oldIdx < comp.size(); ++oldIdx)
    if (reachableFromStart[oldIdx] && endReachableFrom[oldIdx] && nullEquiv.count(oldIdx))
      old2new[oldIdx] = old2new[nullEquiv.at(oldIdx)];
  for (State oldIdx = 0; oldIdx < comp.size(); ++oldIdx)
    if (reachableFromStart[oldIdx] && endReachableFrom[oldIdx])
      for (auto& t: comp[oldIdx].trans)
	t.dest = old2new[t.dest];
  LogThisAt(3,"Transducer composition yielded " << nStates << "-state machine; " << plural (comp.size() - nStates, "more state was", "more states were") << " unreachable" << endl);
  Machine compMachine;
  compMachine.state.reserve (nStates);
  for (State oldIdx = 0; oldIdx < comp.size(); ++oldIdx)
    if (reachableFromStart[oldIdx] && endReachableFrom[oldIdx] && !nullEquiv.count(oldIdx))
      compMachine.state.push_back (comp[oldIdx]);
  return compMachine;
}

vguard<State> Machine::decoderToposort (const string& inputAlphabet) const {
  LogThisAt(5,"Toposorting transducer for decoder" << endl);
  deque<State> S;
  vguard<State> L;
  vguard<int> nParents (nStates());
  vguard<vguard<State> > children (nStates());
  int edges = 0;
  for (State s = 0; s < nStates(); ++s)
    for (const auto& t: state[s].trans)
      if (t.outputEmpty() && (t.inputEmpty() || inputAlphabet.find(t.in) != string::npos)) {
	++nParents[t.dest];
	++edges;
	children[s].push_back (t.dest);
      }
  for (State s = 0; s < nStates(); ++s)
    if (nParents[s] == 0)
      S.push_back (s);
  while (S.size()) {
    const State n = S.front();
    S.pop_front();
    L.push_back (n);
    for (auto m : children[n]) {
      --edges;
      if (--nParents[m] == 0)
	S.push_back (m);
    }
  }
  if (edges > 0)
    throw std::domain_error ("Transducer is cyclic, can't toposort");
  return L;
}

Machine Machine::waitingMachine() const {
  vguard<MachineState> newState (state);
  vguard<State> old2new (nStates()), new2old;
  for (State s = 0; s < nStates(); ++s) {
    const MachineState& ms = state[s];
    old2new[s] = new2old.size();
    new2old.push_back (s);
    if (!ms.waits() && !ms.jumps() && !ms.terminates()) {
      MachineState j, w;
      j.name = ms.name + ";j";
      w.name = ms.name + ";w";
      for (const auto& t: ms.trans)
	if (t.inputEmpty())
	  j.trans.push_back(t);
	else
	  w.trans.push_back(t);
      j.trans.push_back (MachineTransition (MachineNull, MachineNull, newState.size(), 1.));
      old2new.push_back (new2old.size());
      new2old.push_back (newState.size());
      swap (newState[s], j);
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
  LogThisAt(5,"Converted " << nStates() << "-state transducer into " << wm.nStates() << "-state waiting machine" << endl);
  LogThisAt(7,wm.toJsonString() << endl);
  return wm;
}
