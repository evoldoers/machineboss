#ifndef COMPILER_INCLUDED
#define COMPILER_INCLUDED

#include "machine.h"
#include "eval.h"

#define DefaultForwardFunctionName "computeForward"

// dual-purpose C++/JavaScript compiler
struct Compiler {
  typedef size_t TransIndex;
  typedef size_t FuncIndex;
  typedef pair<StateIndex,TransIndex> StateTransIndex;

  // machine analysis for compiler
  struct MachineInfo {
    const Compiler& compiler;
    Machine wm;
    EvaluatedMachine eval;
    map<string,FuncIndex> funcIdx;
    vguard<vguard<StateTransIndex> > incoming;
    MachineInfo (const Compiler&, const Machine&);
    string expr2string (const WeightExpr& w) const { return compiler.expr2string (w, funcIdx); }
    void storeTransitions (ostream&, const string& indent, const string& currentBuf, const string& prevBuf, bool withNull, bool withIn, bool withOut, bool withBoth) const;
    void addTransitions (ostream&, const string& indent, const string& currentBuf, const string& prevBuf, bool withInput, bool withOutput, bool& touched, StateIndex s, bool outputWaiting) const;
    void updateCell (ostream&, const string& indent, const string& currentBuf, bool& touched, StateIndex s, const string& expr) const;
  };

  string funcKeyword;    // "double" for C++, "function" for JS
  string matrixType;     // "const vector<vector<double> >& " for C++
  string vecRefType;     // "const vector<double>&" for C++, "const" for JS
  string paramsType;     // "const map<string,double>& " for C++
  string arrayRefType;   // "double*" for C++, "var" for JS
  string indexType;      // "size_t" for C++, "var" for JS
  string sizeType;       // "const size_t" for C++, "const" for JS
  string sizeMethod;     // "size()" for C++, "length" for JS
  string weightType;     // "const double" for C++, "const" for JS
  string tmpType;        // "double" for C++, "var" for JS
  string mathLibrary;    // "Math." for JS
  
  static string transVar (StateIndex s, TransIndex t);
  static string funcVar (FuncIndex f);

  virtual string mapAccessor (const string& obj, const string& key) const = 0;
  virtual string binarySoftplus (const string&, const string&) const = 0; // library function that implements log(exp(a)+exp(b))
  virtual string declareArray (const string& arrayName, const string& dim1, const string& dim2, const string& dim3) const = 0;
  
  string expr2string (const WeightExpr& w, const map<string,FuncIndex>& funcIdx, int parentPrecedence = 0) const;
  string compileForward (const Machine&, const char* funcName = DefaultForwardFunctionName) const;
};

struct JavaScriptCompiler : Compiler {
  JavaScriptCompiler();
  string declareArray (const string& arrayName, const string& dim1, const string& dim2, const string& dim3) const;
  string binarySoftplus (const string&, const string&) const;
  string mapAccessor (const string& obj, const string& key) const;
};

struct CPlusPlusCompiler : Compiler {
  CPlusPlusCompiler();
  string declareArray (const string& arrayName, const string& dim1, const string& dim2, const string& dim3) const;
  string binarySoftplus (const string&, const string&) const;
  string mapAccessor (const string& obj, const string& key) const;
};

#endif /* COMPILER_INCLUDED */
