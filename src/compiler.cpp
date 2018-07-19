#include "compiler.h"

JavaScriptCompiler::JavaScriptCompiler() {
  funcKeyword = "function";
  indexType = "var";
  varKeyword = "var";
}

CPlusPlusCompiler::CPlusPlusCompiler() {
  funcKeyword = "double";
  matrixType = "const vector<vector<double> >& ";
  indexType = "size_t";
  varKeyword = "double";
}

static const string xvar ("x"), yvar ("y");
static const string xidx ("i"), yidx ("j");
string Compiler::compileForward (const Machine& machine, const char* funcName) const {
  ostringstream body;
  body << funcKeyword << " " << funcName << " (" << matrixType << xvar << ", " << matrixType << yvar << ") {" << endl;
  //  body << "  for (" << indexType << " " << xidx << " = 0; " << xidx << " < " ...
  // TODO: write me
  body << "}" << endl;
  return body.str();
}
