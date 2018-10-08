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
  logPrefixProb = -numeric_limits<double>::infinity();

  if (!parent)
    seqCell (0, 0) = 0;

  for (OutputIndex outPos = 0; outPos <= outLen; ++outPos) {
    const OutputToken outTok = outPos ? tree.output[outPos-1] : OutputTokenizer::emptyToken();
    for (StateIndex d = 0; d < nStates; ++d) {
      LogThisAt(9,"d="<<d<<": ");
      const EvaluatedMachineState& state = tree.machine.state[d];
      double& ll = seqCell (outPos, d);
      const EvaluatedMachineState::OutStateTransMap* nonAbsorbing = NULL;
      const EvaluatedMachineState::OutStateTransMap* absorbing = NULL;
      if (parent && state.incoming.count (inTok))
	absorbing = &state.incoming.at (inTok);
      if (state.incoming.count (InputTokenizer::emptyToken()))
	nonAbsorbing = &state.incoming.at (InputTokenizer::emptyToken());
      if (absorbing && outPos)
	accumulateSeqCell (ll, *absorbing, *parent, outTok, outPos - 1);
      if (absorbing)
	accumulateSeqCell (ll, *absorbing, *parent, OutputTokenizer::emptyToken(), outPos);
      prefixCell (outPos, d) = ll;
      if (outPos && nonAbsorbing)
	accumulateSeqCell (ll, *nonAbsorbing, *this, outTok, outPos - 1);
      if (nonAbsorbing)
	accumulateSeqCell (ll, *nonAbsorbing, *this, OutputTokenizer::emptyToken(), outPos);
      LogThisAt(8,"seqCell("<<outPos<<","<<d<<")="<<ll<<endl);
    }
    // looping over d AND prevState seems inefficient! Could precompute sumInTrans*outTrans for each outTok
    for (StateIndex d = 0; d < nStates; ++d) {
      double& ll = prefixCell (outPos, d);
      if (outPos) {
	const EvaluatedMachineState& state = tree.machine.state[d];
	for (const auto& i_ostm: state.incoming) {
	  const EvaluatedMachineState::OutStateTransMap& outStateTransMap = i_ostm.second;
	  if (outStateTransMap.count (outTok))
	    for (const auto& st: outStateTransMap.at (outTok)) {
	      const EvaluatedMachineState::Trans& trans = st.second;
	      for (StateIndex prevState = 0; prevState < nStates; ++prevState) {
		const double prevCell = prefixCell (outPos - 1, prevState);
		const double logEmitWeight = prevCell + tree.sumInTrans[prevState][st.first] + trans.logWeight;
		log_accum_exp (ll, logEmitWeight);
		LogThisAt(9,"prefixCell("<<outPos<<","<<d<<") logsum+= "<<prevCell<<" + "<<tree.sumInTrans[prevState][st.first]<<" + "<<trans.logWeight<<" ("<<prevState<<"->"<<st.first<<"->"<<d<<")"<<" ... now "<<ll<<endl);
	      }
	    }
	}
      }
      LogThisAt(8,"prefixCell("<<outPos<<","<<d<<")="<<ll<<endl);
    }
  }

  for (StateIndex d = 0; d < nStates; ++d) {
    log_accum_exp (logPrefixProb, prefixCell(outLen,d) + tree.sumInTrans[d][tree.nStates - 1]);
    LogThisAt(9,"logPrefixProb logsum+= "<<prefixCell(outLen,d)<<" + "<<tree.sumInTrans[d][tree.nStates - 1]<<" ("<<d<<"->end)"<<endl);
  }

  if (parent && logPrefixProb > parent->logPrefixProb)
    Warn ("LogP(%s*)=%g rose from LogP(%s*)=%g",
	  to_string_join(tree.seqTraceback(this),"").c_str(), logPrefixProb,
	  to_string_join(tree.seqTraceback(parent),"").c_str(), parent->logPrefixProb);
}

double PrefixTree::Node::logSeqProb() const {
  return seqCell (outLen, nStates - 1);
}

vguard<InputToken> PrefixTree::Node::traceback() const {
  list<InputToken> result;
  for (const Node* node = this; node->inTok; node = node->parent)
    result.push_front (node->inTok);
  return vguard<InputToken> (result.begin(), result.end());
}

PrefixTree::InputIndex PrefixTree::Node::length() const {
  InputIndex len = 0;
  for (const Node* node = this; node->inTok; node = node->parent)
    ++len;
  return len;
}

PrefixTree::Node* PrefixTree::Node::randomChild (mt19937& mt) const {
  uniform_real_distribution<double> distrib (0, 1);
  const double r0 = distrib (mt);
  auto rc = child.begin();
  size_t nc = 0;
  for (double r = r0; rc != child.end() && (r -= exp ((**rc).logPrefixProb - logPrefixProb)) > 0; ++rc)
    ++nc;
  LogThisAt(5,"Randomly sampled child #" << nc << (rc == child.end() ? " (terminating)" : "") << " with probability " << (exp((rc == child.end() ? logSeqProb() : (**rc).logPrefixProb) - logPrefixProb)) << " (r=" << r0 << ")" << endl);
  return rc == child.end() ? NULL : *rc;
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
}

vguard<InputSymbol> PrefixTree::doPrefixSearch() {
  while (!nodeQueue.empty()) {
    Node* parent = bestPrefixNode();
    nodeQueue.pop();
    if (parent->logPrefixProb > bestLogSeqProb)
      extendNode (parent);
    else
      break;
  }

  Assert (bestSeqNode, "No valid sequence found");
  return bestSeq();
}

vguard<InputToken> PrefixTree::sampleTokSeq (mt19937& mt) {
  Node* current = rootNode();
  while (current->logPrefixProb > current->logSeqProb()) {
    extendNode (current);
    Node* next = current->randomChild (mt);
    if (!next)
      break;
    current = next;
  }
  return current->traceback();
}

vguard<InputSymbol> PrefixTree::sampleSeq (mt19937& mt) {
  return machine.inputTokenizer.detokenize (sampleTokSeq (mt));
}

#define BurnSteps 100
#define TargetInitAcceptProb 0.8
vguard<InputSymbol> PrefixTree::doAnnealedSearch (mt19937& mt, int stepsPerTok) {
  const InputToken inToks = machine.inputTokenizer.tok2sym.size() - 1;
  const vguard<InputToken> initSeq = sampleTokSeq (mt);
  const int steps = stepsPerTok * initSeq.size() * inToks;
  LogThisAt(3,"Simulated annealing with initial sequence of length " << initSeq.size() << " at " << stepsPerTok << " steps-per-token with size-" << inToks << " alphabet, total " << steps << " steps" << endl);
  list<InputToken> current (initSeq.begin(), initSeq.end());
  double currentLogSeqProb = logSeqProb (current);
  uniform_int_distribution<InputToken> subDist (1, inToks - 1);
  uniform_int_distribution<InputToken> insDist (1, inToks);
  uniform_real_distribution<double> acceptDist (0, 1);
  const size_t burnSteps = current.size() + BurnSteps;
  vguard<double> burnLog;
  burnLog.reserve (burnSteps);
  double initTemperature = 0;
  int lastBurnStep = 0;
  for (int step = 0; step - lastBurnStep < steps; ++step) {
    const size_t len = current.size();
    const bool burning = burnLog.size() < burnSteps;
    if (burning) {
      lastBurnStep = step;
      if (step > steps && burnLog.empty()) {
	LogThisAt(4,"Failed to find any improvements after " << steps << " attempts; stopping" << endl);
	break;
      }
    }
    const double temperature = burning ? 1. : (initTemperature * (1 - ((step - lastBurnStep) / (double) steps)));
    //  sample type & location of event (substitution, insertion, deletion) with weight (len, len+1, len)
    uniform_int_distribution<int> eventDist (0, 3*len);
    const int r = eventDist (mt);
    const int type = r == 3*len ? (int) 2 : (int) (r / len);
    const int pos = r == 3*len ? (int) len : (int) (r % len);
    auto iter = current.begin();
    for (int n = 0; n < pos; ++n)
      ++iter;
    InputToken oldTok;
    double revFwdProposalRatio;
    switch (type) {
    case 0: // substitution
      {
	const InputToken offset = subDist (mt);
	oldTok = *iter;
	*iter = (((oldTok - 1) + offset) % inToks) + 1;
	revFwdProposalRatio = 1;
      }
      break;
    case 1: // deletion
      oldTok = *iter;
      current.erase (iter++);
      revFwdProposalRatio = 1 / (double) inToks;
      break;
    case 2: // insertion
      {
	const InputToken newTok = insDist (mt);
	iter = current.insert (iter, newTok);
	revFwdProposalRatio = inToks;
      }
      break;
    default:
      break;
    }
    //  calculate logSeqProb (new, old, delta)
    const double newLogSeqProb = logSeqProb (current);
    const double logHastings = min (0., (double) newLogSeqProb - currentLogSeqProb + log (revFwdProposalRatio));
    const double acceptProb = exp (logHastings / temperature);
    const bool accept = (step < burnSteps
			 ? (newLogSeqProb > -numeric_limits<double>::infinity())
			 : (acceptDist(mt) < acceptProb));
    LogThisAt(5,"Simulated annealing step " << (burning?step:(step-lastBurnStep)) << "/" << (burning ? (to_string(burnSteps) + " (burn-in)") : to_string(steps)) << ": T=" << temperature << " " << (type?(type==1?"Delete":"Insert at"):"Substitute") << " " << pos << " of " << to_string_join (machine.inputTokenizer.detokenize(vguard<InputToken> (current.begin(), current.end())),"") << " log(old)=" << currentLogSeqProb << " log(new)=" << newLogSeqProb << " log(revFwdProposalRatio)=" << log(revFwdProposalRatio) << " log(Hastings)=" << logHastings << " P(accept)=" << acceptProb << " " << (accept ? "Accepted" : "Rejected") << endl);
    if (burning && logHastings > -numeric_limits<double>::infinity() && logHastings < numeric_limits<double>::infinity()) {
      burnLog.push_back (logHastings);
      if (burnLog.size() == burnSteps) {
	// expected proportion of accepted moves A = (1/N) sum_n exp(H[n]/T)    (assume all negative H)
	// assume H ~ N(m,v), then
	// A = int_x exp(x/T) exp(-(x-m)^2/v) / sqrt(2*pi*v)
	//   = int_x exp((-x^2 + 2xm - m^2 + vx/T)/v) / sqrt(2*pi*v)
	//   = int_x exp((-x^2 + 2x(m+v/2T) - m^2)/v) / sqrt(2*pi*v)
	//   = int_x exp((-x^2 + 2x(m+v/2T) - (m+v/2T)^2 - (v/2T)^2 + mv/T)/v) / sqrt(2*pi*v)
	//   = exp ((-(v/2T)^2 + mv/T)/v)
 	//   = exp (m/T - v/4T^2)
 	// So
	// log(A) = m/T - v/4T^2
	// 4log(A)T^2 - 4mT + v = 0
	// T = (4m +/- sqrt(16m^2 - 16log(A)v)) / 8log(A)
	//   = (m +/- sqrt(m^2 - log(A)v)) / 2log(A)    ... m<0, can ignore +ve root ...
	//   = (m - sqrt(m^2 - log(A)v)) / 2log(A)      ... check: if v=0, then T = m/log(A) as expected
	double n = 0, sum = 0, sumsq = 0;
	for (double h: burnLog) {
	  sum += h;
	  sumsq += h*h;
	  ++n;
	}
	const double mean = sum/n, variance = sumsq/n - mean*mean;
	const double logInitAcceptProb = log (TargetInitAcceptProb);
	initTemperature = (mean - sqrt(mean*mean - logInitAcceptProb*variance)) / (2*logInitAcceptProb);
	LogThisAt(5,"Log(Hastings) mean " << mean << " variance " << variance << " T0=" << initTemperature << endl);
	LogThisAt(4,"Completed " << burnSteps << "-step burn-in; simulated annealing at initial T=" << initTemperature << " for " << steps << " steps" << endl);
      }
    }
    //  accept/reject
    if (accept)
      currentLogSeqProb = newLogSeqProb;
    else
      switch (type) {
      case 0: *iter = oldTok; break;
      case 1: current.insert (iter, oldTok); break;
      case 2: current.erase (iter); break;
      default: break;
      }
  }
  return bestSeq();
}

double PrefixTree::logSeqProb (const list<InputToken>& input) {
  Node* current = rootNode();
  for (const auto& inTok: input)
    current = addNode (current, inTok);
  return current->logSeqProb();
}

void PrefixTree::extendNode (Node* parent) {
  const InputToken inToks = machine.inputTokenizer.tok2sym.size() - 1;
  LogThisAt (5, "Nodes: " << nodeStore.size() << " Extending " << to_string_join(bestPrefix(),"") << "* (logP " << parent->logPrefixProb << ")" << endl);
  double norm = parent->logSeqProb();
  for (InputToken inTok = 1; inTok <= inToks; ++inTok)
    log_accum_exp (norm, addNode(parent,inTok)->logPrefixProb);
  LogThisAt (6, "log(Sum_x(P(Sx*)) / P(S*)) = " << (norm - parent->logPrefixProb) << endl);
}

PrefixTree::Node* PrefixTree::rootNode() {
  return &nodeStore.front();
}

PrefixTree::Node* PrefixTree::addNode (Node* parent, InputToken inTok) {
  if (parent)
    for (const auto& c: parent->child)
      if (c->inTok == inTok)
	return &*c;
  nodeStore.push_back (Node (*this, parent, inTok));
  Node* nodePtr = &nodeStore.back();
  if (parent)
    parent->child.push_back (nodePtr);
  
  LogThisAt (6, "Adding node " << (parent ? to_string_join (seqTraceback (nodePtr), "") : string("<root>")) << endl);

  nodePtr->fill (*this);
  if (nodePtr->logPrefixProb > bestLogSeqProb)
    nodeQueue.push (nodePtr);

  const double logNodeSeqProb = nodePtr->logSeqProb();
  if (logNodeSeqProb > bestLogSeqProb) {
    bestSeqNode = nodePtr;
    bestLogSeqProb = logNodeSeqProb;
    LogThisAt (4, "Nodes: " << nodeStore.size() << " Best sequence so far: " << to_string_join (bestSeq(), "") << " (" << bestLogSeqProb << ")" << endl);
  }
  LogThisAt (7, "logP(seq)=" << logNodeSeqProb << " logP(seq*)=" << nodePtr->logPrefixProb << " seq: " << to_string_join (seqTraceback (nodePtr), "") << endl);

  return nodePtr;
}

vguard<InputSymbol> PrefixTree::seqTraceback (const Node* node) const {
  return machine.inputTokenizer.detokenize (node->traceback());
}
