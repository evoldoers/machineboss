#ifndef COMPILER_INCLUDED
#define COMPILER_INCLUDED

#include "machine.h"

#define DefaultForwardFunctionName "computeForward"

// dual-purpose C++/JavaScript compiler
struct Compiler {
  string funcKeyword;  // "double" for C++, "function" for JS
  string matrixType;   // "const vector<vector<double> >& " for C++
  string paramsType;   // "const map<string,double>& " for C++
  string paramsAccessorPrefix, paramsAccessorSuffix;   // "string(" and ")" for C++
  string indexType;   // "size_t" for C++, "var" for JS
  string sizeType;   // "const size_t" for C++, "const" for JS
  string sizeMethod;   // "size()" for C++, "length" for JS
  string varKeyword;   // "double" for C++, "var" for JS
  string constVarKeyword;   // "const double" for C++, "const" for JS
  string mathLibrary;  // "Math." for JS
  
  string expr2string (const WeightExpr& w, const map<string,size_t>& funcIdx, int parentPrecedence = 0) const;
  string compileForward (const Machine&, const char* funcName = DefaultForwardFunctionName) const;
};

struct JavaScriptCompiler : Compiler {
  JavaScriptCompiler();
};

struct CPlusPlusCompiler : Compiler {
  CPlusPlusCompiler();
};

#endif /* COMPILER_INCLUDED */
