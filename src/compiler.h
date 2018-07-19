#ifndef COMPILER_INCLUDED
#define COMPILER_INCLUDED

#include "machine.h"

#define DefaultForwardFunctionName "computeForward"

// dual-purpose C++/JavaScript compiler
struct Compiler {
  string funcKeyword;  // "double" for C++, "function" for JS
  string matrixType;   // "const vector<vector<double> >& " for C++, "" for JS
  string indexType;   // "size_t" for C++, "var" for JS
  string varKeyword;   // "double" for C++, "var" for JS
  string compileForward (const Machine&, const char* funcName = DefaultForwardFunctionName) const;
};

struct JavaScriptCompiler : Compiler {
  JavaScriptCompiler();
};

struct CPlusPlusCompiler : Compiler {
  CPlusPlusCompiler();
};

#endif /* COMPILER_INCLUDED */
