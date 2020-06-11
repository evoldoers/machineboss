#ifndef MACHINE_INCLUDED
#define MACHINE_INCLUDED

#include <string>
#include <map>
#include <set>
#include <list>
#include <random>

#include "jsonio.h"
#include "weight.h"
#include "vguard.h"
#include "params.h"
#include "constraints.h"

namespace MachineBoss {

using namespace std;
using json = nlohmann::json;

typedef unsigned long long StateIndex;

#define MachineWaitTag       "wait"
#define MachineContinueTag   NULL
#define MachineCatLeftTag    "concat-l"
#define MachineCatRightTag   "concat-r"
#define MachineDefaultSeqTag "seq"
#define MachineStartTag      "start"
#define MachineEndTag        "end"

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
  size_t findTransition (const MachineTransition&) const;  // finds the index of a trans element, using in/out/dest only (ignoring weight)
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
  size_t nConditionedTransitions() const;  // number of transitions conditional on given input-output labels (calculated as a max over all such labels, plus null transitions)
  StateIndex startState() const;  // always 0 (method checks that machine has at least one state)
  StateIndex endState() const;  // always final state (method checks that machine has at least one state)

  string stateNameJson (StateIndex) const;

  vguard<InputSymbol> inputAlphabet() const;  // alphabetically sorted
  vguard<OutputSymbol> outputAlphabet() const;  // alphabetically sorted

  set<StateIndex> accessibleStates() const;
  set<string> params() const;

  Params getParamDefs (bool assignDefaultValuesToMissingParams = false) const;

  bool stateNamesAreAllNull() const;

  // Machine operations
  // Note the guarantees in the comments for some methods (wild*, concatenate)
  // These guarantees allow construction of local-alignment machines with predictable properties.
  // For example:
  //   wildGenerator.X.wildGenerator  where "." represents concatenation
  // always yields a single left-flanking state, then X's states, then a single right-flanking state
  static Machine null();  // single state, no transitions: weight is one for empty string, zero for all other strings
  static Machine zero();  // two states, no transitions: weight is zero for all strings
  static Machine singleTransition (const WeightExpr& weight);

  static Machine compose (const Machine& first, const Machine& second, bool assignCompositeStateNames = true, bool collapseDegenerateTransitions = true, SilentCycleStrategy cycleStrategy = SumSilentCycles);
  static Machine intersect (const Machine& first, const Machine& second, SilentCycleStrategy cycleStrategy = SumSilentCycles);
  static Machine concatenate (const Machine& left, const Machine& right, const char* leftTag = MachineCatLeftTag, const char* rightTag = MachineCatRightTag);  // guaranteed: left's states followed by right's states

  static Machine generator (const vguard<OutputSymbol>& seq, const string& name = string(MachineDefaultSeqTag));
  static Machine recognizer (const vguard<InputSymbol>& seq, const string& name = string(MachineDefaultSeqTag));
  static Machine echo (const vguard<InputSymbol>& seq, const string& name = string(MachineDefaultSeqTag));

  static Machine wildGenerator (const vguard<OutputSymbol>& symbols);  // guaranteed: returns a single-state Machine
  static Machine wildRecognizer (const vguard<InputSymbol>& symbols);  // guaranteed: returns a single-state Machine
  static Machine wildEcho (const vguard<InputSymbol>& symbols);  // guaranteed: returns a single-state Machine

  static Machine wildSingleGenerator (const vguard<OutputSymbol>& symbols);
  static Machine wildSingleRecognizer (const vguard<InputSymbol>& symbols);
  static Machine wildSingleEcho (const vguard<InputSymbol>& symbols);

  static Machine takeUnion (const Machine& first, const Machine& second);
  static Machine takeUnion (const Machine& first, const Machine& second, const WeightExpr& pFirst);
  static Machine takeUnion (const Machine& first, const Machine& second, const WeightExpr& pFirst, const WeightExpr& pSecond);

  static Machine zeroOrOne (const Machine&);
  static Machine kleeneStar (const Machine&);
  static Machine kleenePlus (const Machine&);
  static Machine kleeneLoop (const Machine&, const Machine&);
  static Machine kleeneCount (const Machine&, const string& countParam);

  static Machine repeat (const Machine&, int copies);
  
  Machine reverse() const;
  Machine transpose() const;

  bool inputEmpty() const;  // true iff machine is a generator
  bool outputEmpty() const;  // true iff machine is a recognizer

  bool isErgodicMachine() const;  // all states accessible
  bool isWaitingMachine() const;  // all states wait or continue
  bool isToposortedMachine (bool acyclic = false) const;  // no i->j transitions where j<i (if acyclic==true, then i->i transitions must not be present either)
  bool isAdvancingMachine() const;  // no silent i->j transitions where j<=i
  bool isDecodingMachine() const;  // no non-outputting i->j transitions where j<=i
  bool isAligningMachine() const;  // at most one i->j transition with given input & output labels

  Machine padWithNullStates() const;  // adds "dummy" null states at start & end
  bool hasNullPaddingStates() const;  // null transitions out of start & into end
  
  Machine silenceInput() const;  // clears all output labels
  Machine silenceOutput() const;  // clears all output labels
  Machine projectOutputToInput() const;  // copies all output labels to input labels, turning a generator into an echoer. Requires inputEmpty()
  Machine projectInputToOutput() const;  // copies all input labels to output labels, turning a recognizer into an echoer. Requires outputEmpty()

  Machine pointwiseReciprocal() const;
  Machine weightInputs (const map<InputSymbol,WeightExpr>&) const;
  Machine weightOutputs (const map<OutputSymbol,WeightExpr>&) const;
  Machine weightInputs (const string& macro = string(WeightMacroDefaultMacro)) const;
  Machine weightOutputs (const string& macro = string(WeightMacroDefaultMacro)) const;

  Machine weightInputsGeometrically (const string&) const;
  Machine weightOutputsGeometrically (const string&) const;

  Machine normalizeJointly() const;  // for each state, sum_{outgoing transitions} p(trans) = 1
  Machine normalizeConditionally() const;  // for each state & each input token, sum_{outgoing transitions} p(trans) = 1

  Machine ergodicMachine() const;  // remove unreachable states
  Machine waitingMachine (const char* waitTag = MachineWaitTag, const char* continueTag = MachineContinueTag) const;  // convert to waiting machine

  size_t nBackTransitions() const;
  size_t nSilentBackTransitions() const;
  size_t nEmptyOutputBackTransitions() const;
  Machine decodeSort() const;  // does advanceSort() on non-outputting transitions
  Machine encodeSort() const;  // same as transpose().decodeSort().transpose()
  Machine toposort() const;  // does advanceSort() on all transitions
  Machine advancingMachine() const;  // convert to advancing machine by eliminating silent back-transitions

  // advanceSort tries to minimize number of "silent" i->j transitions where j<i
  // Different applications can override definition of "silent", e.g. for decoding s/silent/non-outputting/
  Machine advanceSort (function<size_t(const Machine*)> countBackTransitions = &Machine::nSilentBackTransitions,
		       function<bool(const MachineTransition*)> mustAdvance = &MachineTransition::isSilent,
		       const char* mustAdvanceDescription = "silent") const;

  Machine processCycles (SilentCycleStrategy cycleStrategy = SumSilentCycles) const;  // returns either advancingMachine(), dropSilentBackTransitions(), or clone of self, depending on strategy
  Machine dropSilentBackTransitions() const;
  Machine eliminateSilentTransitions (SilentCycleStrategy cycleStrategy = SumSilentCycles) const;  // eliminates silent transitions, first processing cycles using the selected strategy
  Machine eliminateRedundantStates() const;  // eliminates states which have only one outgoing transition that is silent

  Machine subgraph (const vguard<vguard<bool> >&) const;
  Machine downsample (double maxProportionOfTransitionsToKeep, double minPostProbOfSelectedTransitions = 0.) const;
  Machine stochasticDownsample (mt19937& rng, double maxProportionOfTransitionsToKeep, int maxNumberOfPathsToSample) const;

  Machine stripNames() const;  // some algorithms take a while to construct the namespace... this helps
  
  // helpers to import defs & constraints from other machine(s)
  void import (const Machine& m, bool overwrite = false);
  void import (const Machine& m1, const Machine& m2, bool overwrite = false);
};

typedef JsonLoader<Machine> MachineLoader;

struct MachinePath {
  typedef pair<InputSymbol,OutputSymbol> AlignCol;
  typedef list<AlignCol> AlignPath;
  TransList trans;
  MachinePath();
  MachinePath (const MachineTransition&);
  void clear();
  MachinePath concatenate (const MachinePath&) const;
  void writeJson (ostream&, const Machine&) const;
  vguard<InputSymbol> inputSequence() const;
  vguard<OutputSymbol> outputSequence() const;
  AlignPath alignment() const;
};

struct MachineBoundPath : MachinePath {
  const Machine& machine;
  MachineBoundPath (const MachinePath&, const Machine&);
  void writeJson (ostream&) const;
};
  
}  // end namespace

#endif /* MACHINE_INCLUDED */
