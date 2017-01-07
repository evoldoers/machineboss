#ifndef TRANSDUCER_INCLUDED
#define TRANSDUCER_INCLUDED

#include <string>
#include <map>
#include <list>
#include "vguard.h"
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

typedef unsigned long long StateIndex;

#define MachineNull '\0'
#define MachineWaitTag "wait"
#define MachineSilentTag "silent"

typedef char OutputSymbol;
typedef char InputSymbol;
typedef json TransWeight;
typedef json StateName;

struct WeightAlgebra {
  static TransWeight multiply (const TransWeight& l, const TransWeight& r);
  static TransWeight add (const TransWeight& l, const TransWeight& r);
};

struct MachineTransition {
  InputSymbol in;
  OutputSymbol out;
  StateIndex dest;
  TransWeight weight;
  MachineTransition();
  MachineTransition (InputSymbol, OutputSymbol, StateIndex, TransWeight);
  bool inputEmpty() const;
  bool outputEmpty() const;
  bool isSilent() const;  // inputEmpty() && outputEmpty()
  bool isLoud() const;  // !isSilent()
};
typedef list<MachineTransition> TransList;

struct MachineState {
  StateName name;
  TransList trans;
  MachineState();
  const MachineTransition* transFor (InputSymbol in) const;
  bool exitsWithInput() const;  // true if this has an input transition
  bool exitsWithoutInput() const;  // true if this has a non-input transition
  bool exitsWithIO() const;  // true if this has any transitions with input and/or output
  bool exitsWithoutIO() const;  // true if this has any transitions without input or output
  bool terminates() const;  // true if this has no outgoing transitions. Note that the end state is not required to have this property
  bool waits() const;  // !exitsWithoutInput()
  bool continues() const;  // !exitsWithInput() && !terminates()
  bool isDeterministic() const;  // true if this has only one transition and it is non-input
  bool isSilent() const;  // !exitsWithIO()
  bool isLoud() const;  // exitsWithIO() && !exitsWithoutIO()
  const MachineTransition& next() const;  // throws an exception if !isDeterministic()
};

struct Machine {
  vguard<MachineState> state;
  
  Machine();
  StateIndex nStates() const;
  StateIndex startState() const;
  StateIndex endState() const;

  void writeJson (ostream& out) const;
  string toJsonString() const;
  void readJson (istream& in);
  static Machine fromJson (istream& in);
  static Machine fromFile (const char* filename);

  string inputAlphabet() const;
  string outputAlphabet() const;

  set<StateIndex> accessibleStates() const;
  
  static Machine compose (const Machine& first, const Machine& second);

  bool isErgodicMachine() const;  // all states accessible
  bool isWaitingMachine() const;  // all states wait or continue
  bool isPunctuatedMachine() const;  // all states silent or loud
  bool isAdvancingMachine() const;  // no silent i->j transitions where j<i

  Machine ergodicMachine() const;  // remove unreachable states
  Machine waitingMachine() const;  // convert to waiting machine
  Machine punctuatedMachine() const;  // convert to punctuated machine
  Machine advancingMachine() const;  // convert to advancing machine
};

struct TransAccumulator {
  map<StateIndex,map<InputSymbol,map<OutputSymbol,TransWeight> > > t;
  void accumulate (InputSymbol in, OutputSymbol out, StateIndex dest, TransWeight w);
  TransList transitions() const;
};

#endif /* TRANSDUCER_INCLUDED */
