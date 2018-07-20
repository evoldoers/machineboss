#include "compiler.h"

static const string xvar ("x"), yvar ("y"), paramvar ("p"), buf0var ("buf0"), buf1var ("buf1"), currentvar ("current"), prevvar ("prev"), tmpvar ("tmp");
static const string currentcell ("cell"), xcell ("xcell"), ycell ("ycell"), xycell ("xycell");
static const string xidx ("ix"), yidx ("iy"), xvec ("vx"), yvec ("vy");
static const string xsize ("sx"), ysize ("sy");
static const string tab ("  "), tab2 ("    "), tab3 ("      "), tab4 ("      ");

JavaScriptCompiler::JavaScriptCompiler() {
  funcKeyword = "function";
  vecRefType = "const";
  arrayRefType = "var";
  cellRefType = "var";
  constCellRefType = "const";
  indexType = "var";
  sizeType = "const";
  sizeMethod = "length";
  weightType = "const";
  tmpType = "var";
  mathLibrary = "Math.";
}

string JavaScriptCompiler::declareArray (const string& arrayName, const string& dim1, const string& dim2) const {
  return string("var ") + arrayName + " = new Array(" + dim1 + ").fill(0).map (function() { return new Array (" + dim2 + ").fill(0) });";
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
  arrayRefType = "vector<vector<double> >&";
  cellRefType = "vector<double>&";
  constCellRefType = "const vector<double>&";
  indexType = "size_t";
  sizeType = "const size_t";
  sizeMethod = "size()";
  weightType = "const double";
  tmpType = "double";
}

Compiler::MachineInfo::MachineInfo (const Compiler& c, const Machine& m)
  : compiler (c),
    wm (m.advancingMachine().ergodicMachine().waitingMachine()),
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

void Compiler::MachineInfo::updateCell (ostream& result, const string& indent, bool& touched, StateIndex s, const string& expr) const {
  result << indent << tmpvar << " = ";
  if (!touched)
    result << expr;
  else
    result << compiler.binarySoftplus (tmpvar, expr);
  result << ";" << endl;
  touched = true;
}

void Compiler::MachineInfo::addTransitions (ostream& result, const string& indent, bool withInput, bool withOutput, bool& touched, StateIndex s, bool outputWaiting) const {
  if (outputWaiting) {
    if (withInput && !withOutput) {
      const string expr = xcell + "[" + to_string (2*s + 1) + "] + " + xvec + "[" + to_string (eval.inputTokenizer.tok2sym.size() - 1) + "]";
      updateCell (result, indent, touched, s, expr);
    }
  } else {
    if (withOutput && !withInput && wm.state[s].waits()) {
      const string expr = ycell + "[" + to_string(2*s) + "] + " + yvec + "[" + to_string (eval.outputTokenizer.tok2sym.size() - 1) + "]";
      updateCell (result, indent, touched, s, expr);
    }
    for (const auto& s_t: incoming[s]) {
      const auto& trans = wm.state[s_t.first].getTransition(s_t.second);
      if (withInput != trans.inputEmpty() && withOutput != trans.outputEmpty()) {
	string expr = (withOutput ? (withInput ? xycell : ycell) : (withInput ? xcell: currentcell)) + "[" + to_string (2*s_t.first + 1) + "] + " + transVar(s_t.first,s_t.second);
	if (withInput)
	  expr += " + " + xvec + "[" + to_string (eval.inputTokenizer.sym2tok.at(trans.in) - 1) + "]";
	if (withOutput)
	  expr += " + " + yvec + "[" + to_string (eval.outputTokenizer.sym2tok.at(trans.out) - 1) + "]";
	updateCell (result, indent, touched, s, expr);
      }
    }
  }
}

void Compiler::MachineInfo::storeTransitions (ostream& result, const string& indent, bool withNull, bool withIn, bool withOut, bool withBoth) const {
  for (StateIndex s = 0; s < wm.nStates(); ++s) {
    bool touched = false;
    for (int outputWaiting = 0; outputWaiting < 2; ++outputWaiting) {
      if (withIn)
	addTransitions (result, indent, true, false, touched, s, outputWaiting);
      if (withOut)
	addTransitions (result, indent, false, true, touched, s, outputWaiting);
      if (withBoth)
	addTransitions (result, indent, true, true, touched, s, outputWaiting);
      if (withNull)
	addTransitions (result, indent, false, false, touched, s, outputWaiting);
      if (touched)
	result << indent << currentcell << "[" << (2*s + outputWaiting) << "] = " << tmpvar << ";" << endl;
    }
  }
}

string CPlusPlusCompiler::declareArray (const string& arrayName, const string& dim1, const string& dim2) const {
  return string("vector<vector<double> > ") + arrayName + " (" + dim1 + ", vector<double> (" + dim2 + "));";
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
  Assert (m.nStates() > 0, "Can't compile empty machine");
  ostringstream out;
  const MachineInfo info (*this, m);
  const Machine& wm (info.wm);
  out << "// generated automatically by bossmachine, do not edit" << endl;
  out << funcKeyword << " " << funcName << " (" << matrixType << xvar << ", " << matrixType << yvar << ", " << paramsType << paramvar << ") {" << endl;
  out << tab << sizeType << " " << xsize << " = " << xvar << "." << sizeMethod << ";" << endl;
  out << tab << sizeType << " " << ysize << " = " << yvar << "." << sizeMethod << ";" << endl;
  const auto params = WeightAlgebra::toposortParams (wm.defs.defs);
  for (const auto& p: params)
    out << tab << weightType << " " << funcVar(info.funcIdx.at(p)) << " = " << info.expr2string(wm.defs.defs.at(p)) << ";" << endl;
  for (StateIndex s = 0; s < wm.nStates(); ++s) {
    TransIndex t = 0;
    for (const auto& trans: wm.state[s].trans) {
      out << tab << weightType << " " << transVar(s,t) << " = " << expr2string (WeightAlgebra::logOf (trans.weight), info.funcIdx) << ";" << endl;
      ++t;
    }
  }
  // Indexing convention: buf[yWaitFlag][xIndex][state]
  out << tab << declareArray (buf0var, xsize + " + 1", to_string (2*info.wm.nStates())) << endl;
  out << tab << declareArray (buf1var, xsize + " + 1", to_string (2*info.wm.nStates())) << endl;
  out << tab << indexType << " " << xidx << " = 0, " << yidx << ";" << endl;
  out << tab << tmpType << " " << tmpvar << ";" << endl;

  // x=0, y=0
  out << tab << "{" << endl;
  out << tab2 << cellRefType << " " << currentcell << " = " << buf0var << "[0];" << endl;
  out << tab2 << currentcell << "[0] = 0;" << endl;
  out << tab2 << currentcell << "[1] = 0;" << endl;
  info.storeTransitions (out, tab2, true, false, false, false);
  out << tab << "}" << endl;

  // x>0, y=0
  out << tab << "for (" << xidx << " = 1; " << xidx << " <= " << xsize << "; ++" << xidx << ") {" << endl;
  out << tab2 << vecRefType << " " << xvec << " = " << xvar << "[" << xidx << " - 1];" << endl;
  out << tab2 << cellRefType << " " << currentcell << " = " << buf0var << "[" << xidx << "];" << endl;
  out << tab2 << constCellRefType << " " << xcell << " = " << buf0var << "[" << xidx << " - 1];" << endl;
  info.storeTransitions (out, tab2, true, true, false, false);
  out << tab << "}" << endl;

  // y>0
  out << tab << "for (" << yidx << " = 1; " << yidx << " <= " << ysize << "; ++" << yidx << ") {" << endl;
  out << tab2 << vecRefType << " " << yvec << " = " << yvar << "[" << yidx << " - 1];" << endl;
  out << tab2 << arrayRefType << " " << currentvar << " = " << yidx << " & 1 ? " << buf1var << " : " << buf0var << ";" << endl;
  out << tab2 << arrayRefType << " " << prevvar << " = " << yidx << " & 1 ? " << buf0var << " : " << buf1var << ";" << endl;

  // x=0, y>0
  out << tab2 << "{" << endl;
  out << tab3 << cellRefType << " " << currentcell << " = " << currentvar << "[0];" << endl;
  out << tab3 << constCellRefType << " " << ycell << " = " << prevvar << "[0];" << endl;
  info.storeTransitions (out, tab3, true, false, true, false);
  out << tab2 << "}" << endl;

  // x>0, y>0
  out << tab2 << "for (" << xidx << " = 1; " << xidx << " <= " << xsize << "; ++" << xidx << ") {" << endl;
  out << tab3 << vecRefType << " " << xvec << " = " << xvar << "[" << xidx << " - 1];" << endl;
  out << tab3 << cellRefType << " " << currentcell << " = " << currentvar << "[" << xidx << "];" << endl;
  out << tab3 << constCellRefType << " " << xcell << " = " << currentvar << "[" << xidx << " - 1];" << endl;
  out << tab3 << constCellRefType << " " << ycell << " = " << prevvar << "[" << xidx << "];" << endl;
  out << tab3 << constCellRefType << " " << xycell << " = " << prevvar << "[" << xidx << " - 1];" << endl;
  info.storeTransitions (out, tab3, true, true, true, true);

  out << tab2 << "}" << endl;  // end xidx loop
  out << tab << "}" << endl;  // end yidx loop
  out << tab << "return (" << ysize << " & 1 ? " << buf1var << " : " << buf0var << ")[" << xsize << "][" << (2*info.wm.nStates() - 1) << "];" << endl;
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
