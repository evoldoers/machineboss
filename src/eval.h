#ifndef EVAL_INCLUDED
#define EVAL_INCLUDED

#include <algorithm>
#include "trans.h"
#include "params.h"

template<typename Symbol,typename Token>
struct Tokenizer {
  vguard<Symbol> tok2sym;
  map<Symbol,Token> sym2tok;
  Tokenizer (const vguard<Symbol>& symbols) {
    tok2sym.push_back (string());   // token zero is the empty string
    tok2sym.insert (tok2sym.end(), symbols.begin(), symbols.end());
    for (Token tok = 0; tok < (Token) tok2sym.size(); ++tok)
      sym2tok[tok2sym[tok]] = tok;
  }
  inline Token emptyToken() const { return 0; }
  vguard<Token> tokenize (const vguard<Symbol>& symSeq) const {
    vguard<Token> tokSeq (symSeq.size());
    transform (symSeq.begin(), symSeq.end(), tokSeq.begin(), sym2tok.at);
    return tokSeq;
  }
};

typedef int InputToken;
typedef int OutputToken;

typedef Tokenizer<InputSymbol,InputToken> InputTokenizer;
typedef Tokenizer<OutputSymbol,OutputToken> OutputTokenizer;

struct EvaluatedMachineTransition {
  InputToken in;
  OutputToken out;
  StateIndex src, dest;
  double logWeight;

  EvaluatedMachineTransition (StateIndex src, const MachineTransition&, const Params&, const InputTokenizer&, const OutputTokenizer&);
};

struct EvaluatedMachineState {
  StateName name;
  map<InputToken,multimap<OutputToken,EvaluatedMachineTransition> > incoming, outgoing;
};

struct EvaluatedMachine {
  InputTokenizer inputTokenizer;
  OutputTokenizer outputTokenizer;
  vguard<EvaluatedMachineState> state;
  EvaluatedMachine (const Machine&, const Params&);
  inline StateIndex nStates() const { return state.size(); }
};

#endif /* EVAL_INCLUDED */
