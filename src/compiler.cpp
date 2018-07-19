#include "compiler.h"

JavaScriptCompiler::JavaScriptCompiler() {
  funcKeyword = "function";
  indexType = "var";
  sizeType = "const";
  sizeMethod = "length";
  varKeyword = "var";
  constVarKeyword = "const";
  mathLibrary = "Math.";
}

CPlusPlusCompiler::CPlusPlusCompiler() {
  funcKeyword = "double";
  matrixType = "const vector<vector<double> >& ";
  paramsType = "const map<string,double>& ";
  paramsAccessorPrefix = "string(";
  paramsAccessorSuffix = ")";
  indexType = "size_t";
  sizeType = "const size_t";
  sizeMethod = "size()";
  varKeyword = "double";
  constVarKeyword = "const double";
}

static const string xvar ("x"), yvar ("y"), paramvar ("p");
static const string xidx ("ix"), yidx ("iy");
static const string xsize ("sx"), ysize ("sy");
static const string tab ("  ");
string transVar (StateIndex s, size_t t) { return string("t") + to_string(s+1) + "_" + to_string(t+1); }
string funcVar (size_t f) { return string("f") + to_string(f+1); }
string Compiler::compileForward (const Machine& machine, const char* funcName) const {
  ostringstream body;
  map<string,size_t> funcIdx;
  body << funcKeyword << " " << funcName << " (" << matrixType << xvar << ", " << matrixType << yvar << ", " << paramsType << paramvar << ") {" << endl;
  body << tab << sizeType << " " << xsize << " = " << xvar << "." << sizeMethod << ";" << endl;
  body << tab << sizeType << " " << ysize << " = " << yvar << "." << sizeMethod << ";" << endl;
  for (const auto& f_d: machine.defs.defs) {
    const auto f = funcIdx.size();
    funcIdx[f_d.first] = f;
    // UNFINISHED...
    body << tab << constVarKeyword << " " << funcVar(f) << " = " << expr2string(f_d.second,funcIdx) << ";" << endl;
  }
  
  //  body << "  for (" << indexType << " " << xidx << " = 0; " << xidx << " < " ...
  // TODO: write me
  body << "}" << endl;
  return body.str();
}

string Compiler::expr2string (const WeightExpr& w, const map<string,size_t>& funcIdx, int parentPrecedence) const {
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
	expr << paramvar << "[" << paramsAccessorPrefix << '"' << escaped_str(n) << '"' << paramsAccessorSuffix << "]";
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
