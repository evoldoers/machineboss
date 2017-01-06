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

typedef char OutputSymbol;
typedef char InputSymbol;
typedef json TransWeight;

struct MachineTransition {
  InputSymbol in;
  OutputSymbol out;
  State dest;
  TransWeight weight;
  MachineTransition();
  MachineTransition (InputSymbol, OutputSymbol, State, TransWeight);
  bool inputEmpty() const;
  bool outputEmpty() const;
  bool isNull() const;
  static TransWeight multiply (const TransWeight& l, const TransWeight& r);
};

struct MachineState {
  string name;
  vguard<MachineTransition> trans;
  MachineState();
  const MachineTransition* transFor (InputSymbol in) const;
  bool terminates() const;  // true if this has no outgoing transitions. Note that the end state is not required to have this property
  bool exitsWithInput (const char* symbols) const;  // true if this has an input transition for the specified symbols
  bool exitsWithInput() const;  // true if this has an input transition
  bool exitsWithoutInput() const;  // true if this has a non-input transition
  bool emitsOutput() const;  // true if this has an output transition
  bool isDeterministic() const;  // true if this has only one transition and it is non-input
  bool waits() const;  // exitsWithInput() && !exitsWithoutInput()
  bool jumps() const;  // !exitsWithInput() && exitsWithoutInput()
  const MachineTransition& next() const;
};

struct Machine {
  vguard<MachineState> state;
  
  Machine();
  State nStates() const;
  State startState() const;
  State endState() const;
  
  bool isWaitingMachine() const;

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

  Machine waitingMachine() const;  // convert to waiting machine
  vguard<State> decoderToposort (const string& inputAlphabet) const;  // topological sort by non-output transitions
};

#endif /* TRANSDUCER_INCLUDED */
