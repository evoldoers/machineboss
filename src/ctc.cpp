#include "ctc.h"
#include "logger.h"

PrefixTree::Node::Node() :
  inTok (0),
  parent (NULL),
  nStates (0),
  outLen (0)
{ }

PrefixTree::Node::Node (const PrefixTree& tree, const Node* parent, InputToken inTok) :
  inTok (inTok),
  parent (parent),
  nStates (tree.nStates),
  outLen (tree.outLen)
{ }

void PrefixTree::Node::fill (const PrefixTree& tree)
{
  cellStorage = vector<double> (nCells(), -numeric_limits<double>::infinity());

  seqCell (0, 0) = 0;
  for (OutputIndex outPos = 0; outPos <= outLen; ++outPos) {
    const OutputToken outTok = outPos ? tree.output[outPos-1] : OutputTokenizer::emptyToken();
    for (StateIndex d = 0; d < nStates; ++d) {
      const EvaluatedMachineState& state = tree.machine.state[d];
      double& ll = seqCell (outPos, d);
      if (parent && state.incoming.count (inTok)) {
	const auto& incoming = state.incoming.at (inTok);
	if (outPos)
	  accumulateSeqCell (ll, incoming, *parent, outTok, outPos - 1);
	accumulateSeqCell (ll, incoming, *parent, OutputTokenizer::emptyToken(), outPos);
      }
      if (state.incoming.count (InputTokenizer::emptyToken())) {
	const auto& incoming = state.incoming.at (InputTokenizer::emptyToken());
	if (outPos)
	  accumulateSeqCell (ll, incoming, *this, outTok, outPos - 1);
	accumulateSeqCell (ll, incoming, *this, OutputTokenizer::emptyToken(), outPos);
      }
      LogThisAt(6,"seqCell("<<outPos<<","<<d<<")="<<ll<<endl);
    }
    // looping over d AND prevState seems inefficient! Could precompute sumInTrans*outTrans for each outTok
    for (StateIndex d = 0; d < nStates; ++d) {
      double& ll = prefixCell (outPos, d);
      ll = seqCell (outPos, d);
      if (outPos) {
	const EvaluatedMachineState& state = tree.machine.state[d];
	for (const auto& i_ostm: state.incoming) {
	  const EvaluatedMachineState::OutStateTransMap& outStateTransMap = i_ostm.second;
	  if (outStateTransMap.count (outTok))
	    for (const auto& st: outStateTransMap.at (outTok)) {
	      const EvaluatedMachineState::Trans& trans = st.second;
	      for (StateIndex prevState = 0; prevState < nStates; ++prevState) {
		const double prevCell = prefixCell (outPos - 1, prevState);
		log_accum_exp (ll, prevCell + tree.sumInTrans[prevState][st.first] + trans.logWeight);
		LogThisAt(7,"prefixCell("<<outPos<<","<<d<<") logsum+= "<<prevCell<<" + "<<tree.sumInTrans[prevState][st.first]<<" + "<<trans.logWeight<<" ("<<prevState<<"->"<<st.first<<"->"<<d<<")"<<endl);
	      }
	    }
	}
      }
      LogThisAt(6,"prefixCell("<<outPos<<","<<d<<")="<<ll<<endl);
    }
  }

  logPrefixProb = -numeric_limits<double>::infinity();
  for (StateIndex prevState = 0; prevState < nStates; ++prevState) {
    const double prevCell = prefixCell (outLen, prevState);
    log_accum_exp (logPrefixProb, prevCell + tree.sumInTrans[prevState][tree.nStates - 1]);
  }
}

double PrefixTree::Node::logSeqProb() const {
  return seqCell (outLen, nStates - 1);
}

PrefixTree::PrefixTree (const EvaluatedMachine& machine, const vguard<OutputSymbol>& outSym) :
  machine (machine),
  sumInTrans (machine.sumInTrans()),
  output (machine.outputTokenizer.tokenize (outSym)),
  outLen (output.size()),
  nStates (machine.nStates()),
  bestSeqNode (NULL),
  bestLogSeqProb (-numeric_limits<double>::infinity())
{
  addNode (NULL, machine.inputTokenizer.emptyToken());
  const InputToken inToks = machine.inputTokenizer.tok2sym.size() - 1;
  while (!nodeQueue.empty()) {
    const auto bp = bestPrefix();
    LogThisAt (5, "Extending " << (bp.size() ? to_string_join(bp,"") : string("<root>")) << endl);
    Node* parent = bestPrefixNode();
    if (parent->logPrefixProb > bestLogSeqProb) {
      nodeQueue.pop();
      parent->child.reserve (inToks);
      for (InputToken inTok = 1; inTok <= inToks; ++inTok)
	parent->child.push_back (addNode (parent, inTok));
    } else
      break;
  }
}

PrefixTree::Node* PrefixTree::addNode (const Node* parent, InputToken inTok) {
  nodeStore.push_back (Node (*this, parent, inTok));
  Node* nodePtr = &nodeStore.back();

  LogThisAt (5, "Adding node " << (parent ? to_string_join (seqTraceback (nodePtr), "") : string("<root>")) << endl);

  nodePtr->fill (*this);
  nodeQueue.push (nodePtr);

  const double logNodeSeqProb = nodePtr->logSeqProb();
  if (logNodeSeqProb > bestLogSeqProb) {
    bestSeqNode = nodePtr;
    bestLogSeqProb = logNodeSeqProb;
  }
  LogThisAt (6, "logP(seq)=" << logNodeSeqProb << " logP(seq*)=" << nodePtr->logPrefixProb << " seq: " << to_string_join (seqTraceback (nodePtr), "") << endl);

  return nodePtr;
}

vguard<InputSymbol> PrefixTree::seqTraceback (const Node* node) const {
  list<InputToken> inSeq;
  while (node && node->inTok) {
    inSeq.push_back (node->inTok);
    node = node->parent;
  }
  return machine.inputTokenizer.detokenize (vguard<InputToken> (inSeq.rbegin(), inSeq.rend()));
}
