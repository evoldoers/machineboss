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
    void storeTransitions (ostream&, const string& indent, bool withNull, bool withIn, bool withOut, bool withBoth, bool start = false) const;
    void addTransitions (vguard<string>& exprs, bool withInput, bool withOutput, StateIndex s, bool outputWaiting) const;
    string bufRowAccessor (const string&, const string&) const;
    string inputRowAccessor (const string&, const string&) const;
    void showCell (ostream&, const string& indent, bool withInput, bool withOutput) const;
  };

  string preamble;         // #include's, helper function declarations or definitions, etc.
  string funcKeyword;      // keywords to declare function (return type for C++, "function" for JS)
  string matrixType;       // type of probability matrix passed into function
  string funcInit;         // create SoftPlus object, etc.
  string vecRefType;       // type of log-probability vector
  string constVecRefType;  // type of constant log-probability vector
  string paramsType;       // type of parameters object passed into function (string-keyed map)
  string arrayRefType;     // reference to DP matrix row
  string cellRefType;      // reference to DP matrix cell
  string constCellRefType; // const reference to DP matrix cell
  string indexType;        // unsigned int
  string sizeType;         // const unsigned int
  string sizeMethod;       // method to call on a container to get its size
  string weightType;       // const double
  string logWeightType;    // const long, or whatever type is used to store logs internally
  string resultType;       // const double
  string mathLibrary;      // prefix/namespace for math functions
  string infinity;         // maximum value representable as a log
  
  static string transVar (StateIndex s, TransIndex t);
  static string funcVar (FuncIndex f);

  virtual string mapAccessor (const string& obj, const string& key) const = 0;
  virtual string constArrayAccessor (const string& obj, const string& key) const = 0;
  virtual string declareArray (const string& arrayName, const string& dim) const = 0;
  virtual string declareArray (const string& arrayName, const string& dim1, const string& dim2) const = 0;
  virtual string deleteArray (const string& arrayName) const = 0;
  virtual string arrayRowAccessor (const string& arrayName, const string& rowIndex, const string& rowSize) const = 0;
  virtual string makeString (const string& arg) const = 0;
  virtual string toString (const string& arg) const = 0;
  virtual string warn (const vguard<string>& args) const = 0;

  virtual string binarySoftplus (const string&, const string&) const = 0; // library function that implements log(exp(a)+exp(b))
  virtual string boundLog (const string&) const = 0; // library function that constraints argument to infinity bounds
  virtual string unaryLog (const string&) const = 0;
  virtual string unaryExp (const string&) const = 0;
  virtual string realLog (const string&) const = 0;
  
  string logSumExpReduce (vguard<string>& exprs, const string& lineIndent, bool topLevel = true) const;
  string valOrInf (const string& arg) const;
  string expr2string (const WeightExpr& w, const map<string,FuncIndex>& funcIdx, int parentPrecedence = 0) const;

  string compileForward (const Machine&, const char* funcName = DefaultForwardFunctionName) const;
};

struct JavaScriptCompiler : Compiler {
  JavaScriptCompiler();
  string declareArray (const string& arrayName, const string& dim1, const string& dim2) const;
  string declareArray (const string& arrayName, const string& dim) const;
  string deleteArray (const string& arrayName) const;
  string arrayRowAccessor (const string& arrayName, const string& rowIndex, const string& rowSize) const;
  string mapAccessor (const string& obj, const string& key) const;
  string constArrayAccessor (const string& obj, const string& key) const;
  string binarySoftplus (const string&, const string&) const;
  string boundLog (const string&) const;
  string unaryLog (const string&) const;
  string unaryExp (const string&) const;
  string realLog (const string&) const;
  string warn (const vguard<string>& args) const;
  string makeString (const string& arg) const;
  string toString (const string& arg) const;
};

struct CPlusPlusCompiler : Compiler {
  CPlusPlusCompiler();
  string declareArray (const string& arrayName, const string& dim1, const string& dim2) const;
  string declareArray (const string& arrayName, const string& dim) const;
  string deleteArray (const string& arrayName) const;
  string arrayRowAccessor (const string& arrayName, const string& rowIndex, const string& rowSize) const;
  string mapAccessor (const string& obj, const string& key) const;
  string constArrayAccessor (const string& obj, const string& key) const;
  string binarySoftplus (const string&, const string&) const;
  string boundLog (const string&) const;
  string unaryLog (const string&) const;
  string unaryExp (const string&) const;
  string realLog (const string&) const;
  string warn (const vguard<string>& args) const;
  string makeString (const string& arg) const;
  string toString (const string& arg) const;
};

#endif /* COMPILER_INCLUDED */
