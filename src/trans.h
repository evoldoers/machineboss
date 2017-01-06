#ifndef TRANSDUCER_INCLUDED
#define TRANSDUCER_INCLUDED

#include <string>
#include <map>
#include "vguard.h"
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

typedef unsigned long long State;

#define MachineNull '\0'
#define MachineWaitTag "&"
#define MachineSilentTag "$"

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
  State dest;
  TransWeight weight;
  MachineTransition();
  MachineTransition (InputSymbol, OutputSymbol, State, TransWeight);
  bool inputEmpty() const;
  bool outputEmpty() const;
  bool isSilent() const;  // inputEmpty() && outputEmpty()
};

struct MachineState {
  StateName name;
  vguard<MachineTransition> trans;
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
  State nStates() const;
  State startState() const;
  State endState() const;
  
  static Machine compose (const Machine& first, const Machine& second);
  
  void write (ostream& out) const;
  void writeDot (ostream& out) const;
  void writeJson (ostream& out) const;
  string toJsonString() const;
  void readJson (istream& in);
  static Machine fromJson (istream& in);
  static Machine fromFile (const char* filename);
  
  static string stateIndex (State s);
  size_t stateNameWidth() const;
  size_t stateIndexWidth() const;

  string inputAlphabet() const;
  string outputAlphabet() const;

  map<InputSymbol,double> expectedBasesPerInputSymbol (const char* symbols = "01") const;

  bool isWaitingMachine() const;  // all states wait or continue
  bool isPunctuatedMachine() const;  // all states silent or loud
  bool isAcyclicMachine() const;  // no 

  Machine ergodicMachine() const;  // remove unreachable states
  Machine waitingMachine() const;  // convert to waiting machine
  Machine punctuatedMachine() const;  // convert to punctuated machine
};

#endif /* TRANSDUCER_INCLUDED */
