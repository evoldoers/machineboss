#include "compiler.h"

static const string xvar ("x"), yvar ("y"), paramvar ("p"), buf0var ("buf0"), buf1var ("buf1"), currentvar ("current"), prevvar ("prev"), tmpvar ("tmp");
static const string xidx ("ix"), yidx ("iy"), xvec ("vx"), yvec ("vy");
static const string xsize ("sx"), ysize ("sy");
static const string tab ("  "), tab2 ("    "), tab3 ("      "), tab4 ("      ");

JavaScriptCompiler::JavaScriptCompiler() {
  funcKeyword = "function";
  vecRefType = "const";
  arrayRefType = "var";
  indexType = "var";
  sizeType = "const";
  sizeMethod = "length";
  weightType = "const";
  tmpType = "var";
  mathLibrary = "Math.";
}

string JavaScriptCompiler::declareArray (const string& arrayName, const string& dim1, const string& dim2, const string& dim3) const {
  return string("var ") + arrayName + " = new Array(" + dim1 + ").fill(0).map (function() { return new Array (" + dim2 + ").fill(0).map (function() { return new Array (" + dim3 + ").fill(0) }) });";
}

string JavaScriptCompiler::binarySoftplus (const string& a, const string& b) const {
  return string("Math.max (") + a + ", " + b + ") + Math.log(1 + Math.exp(-Math.abs(" + a + " - " + b + ")))";
}

string JavaScriptCompiler::mapAccessor (const string& obj, const string& key) const {
  return obj + "[\"" + escaped_str(key) + "\"]";
}

CPlusPlusCompiler::CPlusPlusCompiler() {
  funcKeyword = "double";
  matrixType = "const vector<vector<double> >& ";
  vecRefType = "const vector<double>&";
  paramsType = "const map<string,double>& ";
  arrayRefType = "vector<vector<vector<double> > >&";
  indexType = "size_t";
  sizeType = "const size_t";
  sizeMethod = "size()";
  weightType = "const double";
  tmpType = "double";
}

Compiler::MachineInfo::MachineInfo (const Compiler& c, const Machine& m)
  : compiler (c),
    wm (m.isWaitingMachine() ? m : m.waitingMachine()),
    eval (wm),
    incoming (wm.nStates())
{
  for (const auto& f_d: wm.defs.defs) {
    const auto f = funcIdx.size();
    funcIdx[f_d.first] = f;
  }
  for (StateIndex s = 0; s < wm.nStates(); ++s) {
    TransIndex t = 0;
    for (const auto& trans: wm.state[s].trans) {
      incoming[trans.dest].push_back (StateTransIndex (s, t));
      ++t;
    }
  }
}

void Compiler::MachineInfo::updateCell (ostream& result, const string& indent, const string& currentBuf, bool& touched, StateIndex s, const string& expr) const {
  result << indent << tmpvar << " = ";
  if (!touched)
    result << expr;
  else
    result << compiler.binarySoftplus (tmpvar, expr);
  result << ";" << endl;
  touched = true;
}

void Compiler::MachineInfo::addTransitions (ostream& result, const string& indent, const string& currentBuf, const string& prevBuf, bool withInput, bool withOutput, bool& touched, StateIndex s, bool outputWaiting) const {
  if (outputWaiting) {
    if (withInput && !withOutput) {
      const string expr = currentBuf + "[1][" + xidx + " - 1][" + to_string(s) + "] + " + xvec + "[" + to_string (eval.inputTokenizer.tok2sym.size() - 1) + "]";
      updateCell (result, indent, currentBuf, touched, s, expr);
    }
  } else {
    if (withOutput && !withInput && wm.state[s].waits()) {
      const string expr = prevBuf + "[0][" + xidx + "][" + to_string(s) + "] + " + yvec + "[" + to_string (eval.outputTokenizer.tok2sym.size() - 1) + "]";
      updateCell (result, indent, currentBuf, touched, s, expr);
    }
    for (const auto& s_t: incoming[s]) {
      const auto& trans = wm.state[s_t.first].getTransition(s_t.second);
      if (withInput != trans.inputEmpty() && withOutput != trans.outputEmpty()) {
	string expr = (withOutput ? prevBuf : currentBuf) + "[1][" + xidx + (withInput ? " - 1" : "") + "][" + to_string(s_t.first) + "] + " + transVar(s_t.first,s_t.second);
	if (withInput)
	  expr += " + " + xvec + "[" + to_string (eval.inputTokenizer.sym2tok.at(trans.in) - 1) + "]";
	if (withOutput)
	  expr += " + " + yvec + "[" + to_string (eval.outputTokenizer.sym2tok.at(trans.out) - 1) + "]";
	updateCell (result, indent, currentBuf, touched, s, expr);
      }
    }
  }
}

void Compiler::MachineInfo::storeTransitions (ostream& result, const string& indent, const string& currentBuf, const string& prevBuf, bool withNull, bool withIn, bool withOut, bool withBoth) const {
  for (StateIndex s = 0; s < wm.nStates(); ++s) {
    bool touched = false;
    for (int outputWaiting = 0; outputWaiting < 2; ++outputWaiting) {
      if (withIn)
	addTransitions (result, indent, currentBuf, prevBuf, true, false, touched, s, outputWaiting);
      if (withOut)
	addTransitions (result, indent, currentBuf, prevBuf, false, true, touched, s, outputWaiting);
      if (withBoth)
	addTransitions (result, indent, currentBuf, prevBuf, true, true, touched, s, outputWaiting);
      if (withNull)
	addTransitions (result, indent, currentBuf, prevBuf, false, false, touched, s, outputWaiting);
      if (touched)
	result << indent << currentBuf << "[" << outputWaiting << "][" << xidx << "][" << s << "] = " << tmpvar << ";" << endl;
    }
  }
}

string CPlusPlusCompiler::declareArray (const string& arrayName, const string& dim1, const string& dim2, const string& dim3) const {
  return string("vector<vector<vector<double> > > ") + arrayName + " (" + dim1 + ", vector<vector<double> > (" + dim2 + ", vector<double> (" + dim3 + ")));";
}

string CPlusPlusCompiler::binarySoftplus (const string& a, const string& b) const {
  return string("log_sum_exp (") + a + ", " + b + ")";
}

string CPlusPlusCompiler::mapAccessor (const string& obj, const string& key) const {
  return obj + ".at(string(\"" + escaped_str(key) + "\"))";
}

string Compiler::funcVar (FuncIndex f) { return string("f") + to_string(f+1); }
string Compiler::transVar (StateIndex s, TransIndex t) { return string("t") + to_string(s+1) + "_" + to_string(t+1); }

string Compiler::compileForward (const Machine& m, const char* funcName) const {
  ostringstream out;
  const MachineInfo info (*this, m);
  const Machine& wm (info.wm);
  out << "// generated automatically by bossmachine, do not edit" << endl;
  out << funcKeyword << " " << funcName << " (" << matrixType << xvar << ", " << matrixType << yvar << ", " << paramsType << paramvar << ") {" << endl;
  out << tab << sizeType << " " << xsize << " = " << xvar << "." << sizeMethod << ";" << endl;
  out << tab << sizeType << " " << ysize << " = " << yvar << "." << sizeMethod << ";" << endl;
  // TODO: toposort params to get dependency order correct. Check for circular dependencies (add tests for these)
  for (const auto& f_d: wm.defs.defs)
    out << tab << weightType << " " << funcVar(info.funcIdx.at(f_d.first)) << " = " << expr2string (f_d.second, info.funcIdx) << ";" << endl;
  for (StateIndex s = 0; s < wm.nStates(); ++s) {
    TransIndex t = 0;
    for (const auto& trans: wm.state[s].trans) {
      out << tab << weightType << " " << transVar(s,t) << " = " << expr2string (WeightAlgebra::logOf (trans.weight), info.funcIdx) << ";" << endl;
      ++t;
    }
  }
  // Indexing convention: buf[yWaitFlag][xIndex][state]
  out << tab << declareArray (buf0var, to_string(2), xsize + " + 1", to_string (info.wm.nStates())) << endl;
  out << tab << declareArray (buf1var, to_string(2), xsize + " + 1", to_string (info.wm.nStates())) << endl;
  out << tab << buf0var << "[" << (info.wm.state[0].waits() ? 1 : 0) << "][0][0] = 0;" << endl;
  out << tab << indexType << " " << xidx << " = 0, " << yidx << ";" << endl;
  out << tab << tmpType << " " << tmpvar << ";" << endl;
  info.storeTransitions (out, tab, buf0var, string(), true, false, false, false);
  out << tab << "for (" << xidx << " = 1; " << xidx << " <= " << xsize << "; ++" << xidx << ") {" << endl;
  out << tab2 << vecRefType << " " << xvec << " = " << xvar << "[" << xidx << " - 1];" << endl;
  info.storeTransitions (out, tab2, buf0var, string(), true, true, false, false);
  out << tab << "}" << endl;
  out << tab << "for (" << yidx << " = 1; " << yidx << " <= " << ysize << "; ++" << yidx << ") {" << endl;
  out << tab2 << vecRefType << " " << yvec << " = " << yvar << "[" << yidx << " - 1];" << endl;
  out << tab2 << arrayRefType << " " << currentvar << " = " << yidx << " & 1 ? " << buf1var << " : " << buf0var << ";" << endl;
  out << tab2 << arrayRefType << " " << prevvar << " = " << yidx << " & 1 ? " << buf0var << " : " << buf1var << ";" << endl;
  info.storeTransitions (out, tab2, currentvar, prevvar, true, false, true, false);
  out << tab2 << "for (" << xidx << " = 1; " << xidx << " <= " << xsize << "; ++" << xidx << ") {" << endl;
  out << tab3 << vecRefType << " " << xvec << " = " << xvar << "[" << xidx << " - 1];" << endl;
  info.storeTransitions (out, tab3, currentvar, prevvar, true, true, true, true);
  out << tab2 << "}" << endl;  // end xidx loop
  out << tab << "}" << endl;  // end yidx loop
  out << tab << "return (" << ysize << " & 1 ? " << buf1var << " : " << buf0var << ")[1][" << xsize << "][" << (info.wm.nStates() - 1) << "];" << endl;
  out << "}" << endl;  // end function
  return out.str();
}

string Compiler::expr2string (const WeightExpr& w, const map<string,FuncIndex>& funcIdx, int parentPrecedence) const {
  ostringstream expr;
  const ExprType op = w->type;
  switch (op) {
  case Null: expr << 0; break;
  case Int: expr << w->args.intValue; break;
  case Dbl: expr << w->args.doubleValue; break;
  case Param:
    {
      const string& n (*w->args.param);
      if (funcIdx.count(n))
	expr << funcVar (funcIdx.at (n));
      else
	expr << mapAccessor (paramvar, n);
    }
    break;
  case Log:
  case Exp:
    expr << mathLibrary << (op == Log ? "log" : "exp") << "(" << expr2string(w->args.arg,funcIdx) << ")";
    break;
  case Pow:
    expr << mathLibrary << "pow(" << expr2string(w->args.binary.l,funcIdx) << "," << expr2string(w->args.binary.r,funcIdx) << ")";
    break;
  default:
    // Precedence rules

    // a*b: rank 2
    // a needs () if it's anything except a multiplication or division [parent rank 2]
    // b needs () if it's anything except a multiplication or division [parent rank 2]

    // a/b: rank 2
    // a needs () if it's anything except a multiplication or division [parent rank 2]
    // b needs () if it's anything except a constant/function [parent rank 3]

    // a-b: rank 1
    // a never needs () [parent rank 0]
    // b needs () if it's anything except a multiplication or division [parent rank 2]

    // a+b: rank 1
    // a never needs () [parent rank 0]
    // b never needs () [parent rank 0]

    int p, l, r;
    string opcode;
    if (op == Mul) { p = l = r = 2; opcode = "*"; }
    else if (op == Div) { p = l = 2; r = 3; opcode = "/"; }
    else if (op == Sub) { p = 1; l = 0; r = 2; opcode = "-"; }
    else if (op == Add) { p = 1; l = r = 0; opcode = "+"; }
    expr << (parentPrecedence > p ? "(" : "")
	 << expr2string(w->args.binary.l,funcIdx,l)
	 << opcode
	 << expr2string(w->args.binary.r,funcIdx,r)
	 << (parentPrecedence > p ? ")" : "");
    break;
  }
  return expr.str();
}
