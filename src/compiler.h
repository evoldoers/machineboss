#ifndef COMPILER_INCLUDED
#define COMPILER_INCLUDED

#include "machine.h"
#include "eval.h"

#define DefaultForwardFunctionName "computeForward"
#define DefaultCodeGenDir          "."
#define DirectorySeparator         "/"

// dual-purpose C++/JavaScript compiler
struct Compiler {
  typedef size_t TransIndex;
  typedef size_t FuncIndex;
  typedef pair<StateIndex,TransIndex> StateTransIndex;

  typedef enum SeqType { Profile = 0, IntVec = 1, String = 2 } SeqType;
  
  // machine analysis for compiler
  struct MachineInfo {
    const Compiler& compiler;
    Machine wm;
    EvaluatedMachine eval;
    map<string,FuncIndex> funcIdx;
    vguard<vguard<StateTransIndex> > incoming;
    MachineInfo (const Compiler&, const Machine&);
    string expr2string (const WeightExpr& w) const { return compiler.expr2string (w, funcIdx); }
    string storeTransitions (ostream*, const char* dir, const char* funcPrefix, bool withNull, bool withIn, bool withOut, bool withBoth, InputToken inTok, OutputToken outTok, SeqType outType, bool start) const;
    void addTransitions (vguard<string>& exprs, bool withInput, bool withOutput, StateIndex s, InputToken inTok, OutputToken outTok, SeqType outType, bool outputWaiting) const;
    void flushTransitions (ostream&, string& lvalue, string& rvalue, const string& indent) const;
    string bufRowAccessor (const string&, const string&, const SeqType) const;
    string inputRowAccessor (const string&, const string&) const;
    void showCell (ostream&, const string& indent, bool withInput, bool withOutput) const;
  };

  // general config
  bool showCells;  // add code that displays DP matrix
  
  // per-language config
  string preamble;         // #include's, helper function declarations or definitions, etc.
  string funcKeyword;      // keywords to declare function returning number (return type for C++, "function" for JS)
  string voidFuncKeyword;  // keywords to declare function returning void ("void" for C++, "function" for JS)
  string matrixArgType;    // type of probability matrix passed into function (for SeqType == Profile)
  string intVecArgType;    // type of integer vector passed into function (for SeqType == Int)
  string stringArgType;    // type of string passed into function (for SeqType == String)
  string softplusArgType;  // type of SoftPlus reference ("const SoftPlus&" for C++)
  string cellArgType;      // type of DP matrix cell pointer passed as argument
  string constCellArgType; // type of DP matrix cell pointer passed as argument
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
  string nullValue;        // NULL (C++) or null (JS)
  string infinity;         // maximum value representable using internal log type
  string realInfinity;     // maximum value representable using floating-point type
  string boolType;         // bool
  string abort;            // throw runtime_error()
  string filenameSuffix;   // .cpp, .js
  string headerSuffix;     // .h, .js
  
  Compiler();
  
  static string transVar (const EvaluatedMachine&, StateIndex s, TransIndex t);
  static string funcVar (FuncIndex f);

  virtual string mapAccessor (const string& obj, const string& key) const = 0;
  virtual string mapContains (const string& obj, const string& key) const = 0;
  virtual string constArrayAccessor (const string& obj, const string& key) const = 0;
  virtual string declareArray (const string& arrayName, const string& dim) const = 0;
  virtual string declareArray (const string& arrayName, const string& dim1, const string& dim2) const = 0;
  virtual string deleteArray (const string& arrayName) const = 0;
  virtual string arrayRowAccessor (const string& arrayName, const string& rowIndex, const string& rowSize) const = 0;
  virtual string makeString (const string& arg) const = 0;
  virtual string toString (const string& arg) const = 0;
  virtual string warn (const vguard<string>& args) const = 0;
  virtual string include (const string& filename) const = 0;
  virtual string declareFunction (const string& proto) const = 0;

  virtual string binarySoftplus (const string&, const string&) const = 0; // library function that implements log(exp(a)+exp(b))
  virtual string boundLog (const string&) const = 0; // library function that constraints argument to infinity bounds
  virtual string unaryLog (const string&) const = 0;
  virtual string unaryExp (const string&) const = 0;
  virtual string realLog (const string&) const = 0;

  virtual string postamble (const vguard<string>& funcNames) const = 0;
  
  static bool isCharAlphabet (const vguard<string>&);

  string logSumExpReduce (vguard<string>& exprs, const string& lineIndent, bool topLevel, bool alreadyBounded) const;
  string valOrInf (const string& arg) const;
  string expr2string (const WeightExpr& w, const map<string,FuncIndex>& funcIdx, int parentPrecedence = 0) const;
  string assertParamDefined (const string& p) const;
  
  string headerFilename (const char* dir, const char* funcName) const;
  string privateHeaderFilename (const char* dir, const char* funcName) const;

  void compileForward (const Machine&, SeqType xType = Profile, SeqType yType = Profile, const char* dir = DefaultCodeGenDir, const char* funcName = DefaultForwardFunctionName) const;
};

struct JavaScriptCompiler : Compiler {
  JavaScriptCompiler();
  string declareArray (const string& arrayName, const string& dim1, const string& dim2) const;
  string declareArray (const string& arrayName, const string& dim) const;
  string deleteArray (const string& arrayName) const;
  string arrayRowAccessor (const string& arrayName, const string& rowIndex, const string& rowSize) const;
  string mapAccessor (const string& obj, const string& key) const;
  string mapContains (const string& obj, const string& key) const;
  string constArrayAccessor (const string& obj, const string& key) const;
  string binarySoftplus (const string&, const string&) const;
  string boundLog (const string&) const;
  string unaryLog (const string&) const;
  string unaryExp (const string&) const;
  string realLog (const string&) const;
  string warn (const vguard<string>& args) const;
  string makeString (const string& arg) const;
  string toString (const string& arg) const;
  string postamble (const vguard<string>& funcNames) const;
  string include (const string& filename) const;
  string declareFunction (const string& proto) const;
};

struct CPlusPlusCompiler : Compiler {
  string cellType;  // "long" (32-bit) or "long long" (64-bit)
  CPlusPlusCompiler (bool is64bit);
  string declareArray (const string& arrayName, const string& dim1, const string& dim2) const;
  string declareArray (const string& arrayName, const string& dim) const;
  string deleteArray (const string& arrayName) const;
  string arrayRowAccessor (const string& arrayName, const string& rowIndex, const string& rowSize) const;
  string mapAccessor (const string& obj, const string& key) const;
  string mapContains (const string& obj, const string& key) const;
  string constArrayAccessor (const string& obj, const string& key) const;
  string binarySoftplus (const string&, const string&) const;
  string boundLog (const string&) const;
  string unaryLog (const string&) const;
  string unaryExp (const string&) const;
  string realLog (const string&) const;
  string warn (const vguard<string>& args) const;
  string makeString (const string& arg) const;
  string toString (const string& arg) const;
  string postamble (const vguard<string>& funcNames) const;
  string include (const string& filename) const;
  string declareFunction (const string& proto) const;
};

#endif /* COMPILER_INCLUDED */
