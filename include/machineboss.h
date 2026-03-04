#ifndef MACHINEBOSS_API_INCLUDED
#define MACHINEBOSS_API_INCLUDED

/// @file machineboss.h
/// @brief Public API for the Machine Boss WFST library.
///
/// Include this single header to access all public functionality.
/// Link with: -lboss -lgsl -lgslcblas -lboost_regex -lz

// --- Core types ---
#include "machine.h"      // Machine, MachineState, MachineTransition, MachinePath
#include "weight.h"       // WeightExpr, WeightAlgebra
#include "params.h"       // Params, ParamAssign, ParamFuncs
#include "constraints.h"  // Constraints
#include "seqpair.h"      // SeqPair, SeqPairList, Envelope
#include "eval.h"         // EvaluatedMachine, Tokenizer

// --- Algorithms ---
#include "forward.h"      // ForwardMatrix, RollingOutputForwardMatrix
#include "backward.h"     // BackwardMatrix
#include "viterbi.h"      // ViterbiMatrix
#include "counts.h"       // MachineCounts, MachineObjective
#include "fitter.h"       // MachineFitter
#include "beam.h"         // BeamSearchMatrix
#include "ctc.h"          // PrefixTree
#include "compiler.h"     // Compiler, CPlusPlusCompiler, JavaScriptCompiler, WGSLCompiler

// --- Import & construction ---
#include "preset.h"       // MachinePresets
#include "hmmer.h"        // HmmerModel
#include "csv.h"          // CSVProfile
#include "jphmm.h"        // JPHMM
#include "parsers.h"      // RegexParser, parseWeightExpr
#include "fastseq.h"      // FastSeq, readFastSeqs

// --- Convenience API ---
#include "api.h"          // High-level free functions

#endif /* MACHINEBOSS_API_INCLUDED */
