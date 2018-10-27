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

#define MachineWaitTag       "wait"
#define MachineContinueTag   NULL
#define MachineCatLeftTag    "concat-l"
#define MachineCatRightTag   "concat-r"
#define MachineDefaultSeqTag "seq"
#define MachineEndTag        "end"
#define MachineParamPrefix   "p"

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
  typedef enum SilentCycleStrategy { LeaveSilentCycles = 0, BreakSilentCycles = 1, SumSilentCycles = 2 } SilentCycleStrategy;

  ParamFuncs funcs;
  Constraints cons;
  vguard<MachineState> state;

  void writeJson (ostream& out, bool memoizeRepeatedExpressions = false, bool showParams = false, bool useStateIDs = false) const;
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

  Params getParamDefs (bool assignDefaultValuesToMissingParams = false) const;
  
  static Machine null();
  static Machine singleTransition (const WeightExpr& weight);

  static Machine compose (const Machine& first, const Machine& second, bool assignCompositeStateNames = true, bool collapseDegenerateTransitions = true, SilentCycleStrategy cycleStrategy = SumSilentCycles);
  static Machine intersect (const Machine& first, const Machine& second, SilentCycleStrategy cycleStrategy = SumSilentCycles);
  static Machine concatenate (const Machine& left, const Machine& right, const char* leftTag = MachineCatLeftTag, const char* rightTag = MachineCatRightTag);

  static Machine generator (const vguard<OutputSymbol>& seq, const string& name = string(MachineDefaultSeqTag));
  static Machine acceptor (const vguard<InputSymbol>& seq, const string& name = string(MachineDefaultSeqTag));
  static Machine echo (const vguard<InputSymbol>& seq, const string& name = string(MachineDefaultSeqTag));

  static Machine wildGenerator (const vguard<OutputSymbol>& symbols);
  static Machine wildAcceptor (const vguard<InputSymbol>& symbols);
  static Machine wildEcho (const vguard<InputSymbol>& symbols);

  static Machine wildSingleGenerator (const vguard<OutputSymbol>& symbols);
  static Machine wildSingleAcceptor (const vguard<InputSymbol>& symbols);
  static Machine wildSingleEcho (const vguard<InputSymbol>& symbols);

  static Machine takeUnion (const Machine& first, const Machine& second);
  static Machine takeUnion (const Machine& first, const Machine& second, const WeightExpr& pFirst);
  static Machine takeUnion (const Machine& first, const Machine& second, const WeightExpr& pFirst, const WeightExpr& pSecond);

  static Machine zeroOrOne (const Machine&);
  static Machine kleeneStar (const Machine&);
  static Machine kleenePlus (const Machine&);
  static Machine kleeneLoop (const Machine&, const Machine&);
  static Machine kleeneCount (const Machine&, const string& countParam);

  Machine reverse() const;
  Machine transpose() const;

  bool inputEmpty() const;
  bool outputEmpty() const;

  bool isErgodicMachine() const;  // all states accessible
  bool isWaitingMachine() const;  // all states wait or continue
  bool isAdvancingMachine() const;  // no silent i->j transitions where j<i
  bool isDecodingMachine() const;  // no non-outputting i->j transitions where j<i
  bool isAligningMachine() const;  // at most one i->j transition with given input & output labels

  Machine padWithNullStates() const;  // adds "dummy" null states at start & end
  bool hasNullPaddingStates() const;  // null transitions out of start & into end
  
  Machine projectOutputToInput() const;  // copies all output labels to input labels, turning a generator into an echoer. Requires inputEmpty()

  Machine pointwiseReciprocal() const;
  Machine weightInputs (const char* paramPrefix = MachineParamPrefix) const;
  Machine weightInputs (const map<InputSymbol,WeightExpr>&) const;
  Machine weightOutputs (const char* paramPrefix = MachineParamPrefix) const;
  Machine weightOutputs (const map<OutputSymbol,WeightExpr>&) const;

  Machine weightInputsUniformly() const;
  Machine weightOutputsUniformly() const;

  Machine normalizeJointly() const;  // for each state, sum_{outgoing transitions} p(trans) = 1
  Machine normalizeConditionally() const;  // for each state & each input token, sum_{outgoing transitions} p(trans) = 1

  Machine ergodicMachine() const;  // remove unreachable states
  Machine waitingMachine (const char* waitTag = MachineWaitTag, const char* continueTag = MachineContinueTag) const;  // convert to waiting machine

  size_t nSilentBackTransitions() const;
  size_t nEmptyOutputBackTransitions() const;
  Machine decodeSort() const;  // same as advanceSort(true)
  Machine encodeSort() const;  // same as transpose().advanceSort(true).transpose()
  Machine advanceSort (bool decode = false) const;  // attempt to minimize number of silent i->j transitions where j<i (if decode=true, then s/silent/non-outputting/)
  Machine advancingMachine() const;  // convert to advancing machine by eliminating silent back-transitions

  Machine processCycles (SilentCycleStrategy cycleStrategy = SumSilentCycles) const;  // returns either advancingMachine(), dropSilentBackTransitions(), or clone of self, depending on strategy
  Machine dropSilentBackTransitions() const;
  Machine eliminateSilentTransitions (SilentCycleStrategy cycleStrategy = SumSilentCycles) const;  // eliminates silent transitions, first processing cycles using the selected strategy

  // helpers to import defs & constraints from other machine(s)
  void import (const Machine& m);
  void import (const Machine& m1, const Machine& m2);
};

typedef JsonLoader<Machine> MachineLoader;

struct MachinePath {
  TransList trans;
  void writeJson (ostream&, const Machine&) const;
};

struct MachineBoundPath : MachinePath {
  const Machine& machine;
  MachineBoundPath (const MachinePath&, const Machine&);
  void writeJson (ostream&) const;
};

#endif /* MACHINE_INCLUDED */
