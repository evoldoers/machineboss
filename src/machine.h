#ifndef MACHINE_INCLUDED
#define MACHINE_INCLUDED

#include <string>
#include <map>
#include <set>
#include <list>
#include "jsonio.h"
#include "weight.h"
#include "vguard.h"
#include "params.h"
#include "constraints.h"

using namespace std;
using json = nlohmann::json;

typedef unsigned long long StateIndex;

#define MachineWaitTag     "wait"
#define MachineContinueTag NULL
#define MachineCatLeftTag  "concat-l"
#define MachineCatRightTag "concat-r"

typedef string OutputSymbol;
typedef string InputSymbol;
typedef json StateName;

struct MachineTransition {
  InputSymbol in;
  OutputSymbol out;
  StateIndex dest;
  WeightExpr weight;
  MachineTransition();
  MachineTransition (InputSymbol, OutputSymbol, StateIndex, WeightExpr);
  MachineTransition (InputSymbol, OutputSymbol, StateIndex, double);
  bool inputEmpty() const;
  bool outputEmpty() const;
  bool isSilent() const;  // inputEmpty() && outputEmpty()
  bool isLoud() const;  // !isSilent()
};
typedef list<MachineTransition> TransList;

struct MachinePath {
  TransList trans;
  void writeJson (ostream&) const;
};

struct MachineState {
  StateName name;
  TransList trans;
  MachineState();
  MachineTransition getTransition (size_t) const;  // gets the n'th element from trans
  bool exitsWithInput() const;  // true if this has an input transition
  bool exitsWithoutInput() const;  // true if this has a non-input transition
  bool exitsWithIO() const;  // true if this has any transitions with input and/or output
  bool exitsWithoutIO() const;  // true if this has any transitions without input or output
  bool terminates() const;  // true if this has no outgoing transitions. Note that the end state is not required to have this property
  bool waits() const;  // !exitsWithoutInput()                     ["input" or "end" state: machine only leaves this state if it receives input]
  bool continues() const;  // !exitsWithInput() && !terminates()   ["insert" state: can't accept input, and has at least one outgoing transition]
  bool isSilent() const;  // !exitsWithIO()                        ["null" state]
  bool isLoud() const;  // exitsWithIO() && !exitsWithoutIO()      ["emit" state]
};

struct Machine {
  ParamFuncs defs;
  Constraints cons;
  vguard<MachineState> state;

  void writeJson (ostream& out, bool memoizeRepeatedExpressions = false, bool showParams = false) const;
  void readJson (const json& json);
  void writeDot (ostream& out, const char* emptyLabelText = "&epsilon;") const;

  StateIndex nStates() const;
  size_t nTransitions() const;
  StateIndex startState() const;
  StateIndex endState() const;

  string stateNameJson (StateIndex) const;

  vguard<InputSymbol> inputAlphabet() const;  // alphabetically sorted
  vguard<OutputSymbol> outputAlphabet() const;  // alphabetically sorted

  set<StateIndex> accessibleStates() const;
  set<string> params() const;

  static Machine null();
  static Machine singleTransition (const WeightExpr& weight);

  static Machine compose (const Machine& first, const Machine& second, bool assignCompositeStateNames = true, bool collapseDegenerateTransitions = true);
  static Machine intersect (const Machine& first, const Machine& second);
  static Machine concatenate (const Machine& left, const Machine& right, const char* leftTag = MachineCatLeftTag, const char* rightTag = MachineCatRightTag);
  static Machine generator (const string& name, const vguard<OutputSymbol>& seq);
  static Machine acceptor (const string& name, const vguard<InputSymbol>& seq);

  static Machine takeUnion (const Machine& first, const Machine& second);
  static Machine takeUnion (const Machine& first, const Machine& second, const WeightExpr& pFirst);
  static Machine takeUnion (const Machine& first, const Machine& second, const WeightExpr& pFirst, const WeightExpr& pSecond);

  static Machine zeroOrOne (const Machine&);
  static Machine kleeneStar (const Machine&);
  static Machine kleenePlus (const Machine&);
  static Machine kleeneLoop (const Machine&, const Machine&);

  Machine reverse() const;
  Machine transpose() const;

  bool inputEmpty() const;
  bool outputEmpty() const;

  bool isErgodicMachine() const;  // all states accessible
  bool isWaitingMachine() const;  // all states wait or continue
  bool isAdvancingMachine() const;  // no silent i->j transitions where j<i
  bool isAligningMachine() const;  // at most i->j transition with given input & output labels

  Machine projectOutputToInput() const;  // copies all output labels to input labels. Requires inputEmpty()

  Machine ergodicMachine() const;  // remove unreachable states
  Machine waitingMachine (const char* waitTag = MachineWaitTag, const char* continueTag = MachineContinueTag) const;  // convert to waiting machine
  Machine advancingMachine() const;  // convert to advancing machine

  Machine eliminateSilentTransitions() const;

  size_t nSilentBackTransitions() const;
  Machine advanceSort() const;  // attempt to minimize number of silent i->j transitions where j<i

  // helpers to import defs & constraints from other machine(s)
  void import (const Machine& m);
  void import (const Machine& m1, const Machine& m2);
};

typedef JsonLoader<Machine> MachineLoader;

#endif /* MACHINE_INCLUDED */
