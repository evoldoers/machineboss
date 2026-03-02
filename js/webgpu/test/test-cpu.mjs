/**
 * CPU fallback tests for MachineBoss WebGPU module.
 *
 * Validates Forward, Viterbi, Backward, and posterior computations
 * against reference values from the `boss` CLI.
 *
 * Run: node js/webgpu/test/test-cpu.mjs
 */

import { readFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

import { MachineBoss } from '../machineboss-gpu.mjs';
import { prepareMachine, tokenize } from '../internal/machine-prep.mjs';
import { evaluateWeight } from '../internal/machine-prep.mjs';
import { NEG_INF, logaddexp, makeSemiring, matMul, logIdentity } from '../internal/logmath.mjs';
import { silentClosure } from '../cpu/silent.mjs';
import { forward1D } from '../cpu/forward-1d.mjs';
import { forward2D } from '../cpu/forward-2d.mjs';
import { backward1D } from '../cpu/backward-1d.mjs';
import { backward2D } from '../cpu/backward-2d.mjs';
import { viterbi1D } from '../cpu/viterbi-1d.mjs';
import { viterbi2D } from '../cpu/viterbi-2d.mjs';
import { posteriors1D, posteriors2D } from '../cpu/posteriors.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..', '..', '..');

function loadJSON(path) {
  return JSON.parse(readFileSync(join(ROOT, path), 'utf8'));
}

let passed = 0;
let failed = 0;

function assert(condition, msg) {
  if (!condition) {
    console.error(`  FAIL: ${msg}`);
    failed++;
  } else {
    passed++;
  }
}

function assertClose(actual, expected, tol, msg) {
  if (expected === -Infinity) {
    assert(actual === -Infinity, `${msg}: expected -Infinity, got ${actual}`);
  } else {
    const diff = Math.abs(actual - expected);
    assert(diff < tol, `${msg}: expected ${expected}, got ${actual} (diff=${diff})`);
  }
}

// ============================================================
// Test: Weight expression evaluator
// ============================================================
console.log('Testing weight expression evaluator...');

assert(evaluateWeight(1) === 1, 'constant 1');
assert(evaluateWeight(0.5) === 0.5, 'constant 0.5');
assert(evaluateWeight('p', { p: 0.9 }) === 0.9, 'parameter p');
assert(evaluateWeight({ '*': [2, 3] }) === 6, '2*3');
assert(evaluateWeight({ '+': [2, 3] }) === 5, '2+3');
assert(evaluateWeight({ '-': [5, 2] }) === 3, '5-2');
assert(evaluateWeight({ '/': [6, 2] }) === 3, '6/2');
assertClose(evaluateWeight({ 'log': Math.E }), 1.0, 1e-10, 'log(e)');
assertClose(evaluateWeight({ 'exp': 0 }), 1.0, 1e-10, 'exp(0)');
assert(evaluateWeight({ 'not': 0.3 }) === 0.7, 'not 0.3');
assert(evaluateWeight({ '*': ['no_ins', 'no_del'] }, { no_ins: 0.9, no_del: 0.9 }) === 0.81, 'compound expr');

// ============================================================
// Test: logmath
// ============================================================
console.log('Testing logmath...');

assertClose(logaddexp(Math.log(0.3), Math.log(0.7)), 0, 1e-10, 'logaddexp(log(0.3), log(0.7)) = 0');
assert(logaddexp(NEG_INF, 5) === 5, 'logaddexp(-inf, 5) = 5');
assert(logaddexp(5, NEG_INF) === 5, 'logaddexp(5, -inf) = 5');
assert(logaddexp(NEG_INF, NEG_INF) === NEG_INF, 'logaddexp(-inf, -inf) = -inf');

// ============================================================
// Test: Machine preparation
// ============================================================
console.log('Testing machine preparation...');

const bitnoise = loadJSON('t/machine/bitnoise.json');
const prepared = prepareMachine(bitnoise, { p: 0.9, q: 0.1 });

assert(prepared.nStates === 1, 'bitnoise has 1 state');
assert(prepared.nInputTokens === 3, 'bitnoise: 3 input tokens (null, 0, 1)');
assert(prepared.nOutputTokens === 3, 'bitnoise: 3 output tokens (null, 0, 1)');
assert(prepared.inputAlphabet[0] === '', 'input alphabet[0] = null');
assert(prepared.inputAlphabet[1] === '0', 'input alphabet[1] = "0"');
assert(prepared.inputAlphabet[2] === '1', 'input alphabet[2] = "1"');

// Check transition tensor values
// logTrans[in=1("0"), out=1("0"), 0, 0] = log(p) = log(0.9)
const nOut = prepared.nOutputTokens;
const S = prepared.nStates;
const idx_00 = ((1 * nOut + 1) * S + 0) * S + 0;
assertClose(prepared.logTrans[idx_00], Math.log(0.9), 1e-10, 'logTrans[0->0,0->0] = log(0.9)');

// logTrans[in=1("0"), out=2("1"), 0, 0] = log(q) = log(0.1)
const idx_01 = ((1 * nOut + 2) * S + 0) * S + 0;
assertClose(prepared.logTrans[idx_01], Math.log(0.1), 1e-10, 'logTrans[0->1,0->0] = log(0.1)');

// ============================================================
// Test: Tokenization
// ============================================================
console.log('Testing tokenization...');

const tokens = tokenize('001', prepared.inputAlphabet);
assert(tokens.length === 3, 'tokenize length');
assert(tokens[0] === 1, 'tokenize 0 -> 1');
assert(tokens[1] === 1, 'tokenize 0 -> 1 (2)');
assert(tokens[2] === 2, 'tokenize 1 -> 2');

// ============================================================
// Test: 2D Forward - bitnoise
// ============================================================
console.log('Testing 2D Forward...');

{
  const machine = prepareMachine(bitnoise, { p: 0.9, q: 0.1 });
  const inTok = tokenize('001', machine.inputAlphabet);
  const outTok = tokenize('101', machine.outputAlphabet);
  const ll = await forward2D(machine, inTok, outTok);
  // Reference: boss gives -2.51331
  assertClose(ll, -2.51331, 1e-4, '2D Forward bitnoise 001->101 (p=0.9)');
}

{
  const machine = prepareMachine(bitnoise, { p: 0.01, q: 0.99 });
  const inTok = tokenize('001', machine.inputAlphabet);
  const outTok = tokenize('101', machine.outputAlphabet);
  const ll = await forward2D(machine, inTok, outTok);
  // Reference: boss gives -9.22039
  assertClose(ll, -9.22039, 1e-4, '2D Forward bitnoise 001->101 (p=0.01)');
}

// ============================================================
// Test: 2D Forward - bitecho (identity transducer)
// ============================================================
{
  const bitecho = loadJSON('t/machine/bitecho.json');
  const machine = prepareMachine(bitecho, {});
  const inTok = tokenize('101', machine.inputAlphabet);
  const outTok = tokenize('101', machine.outputAlphabet);
  const ll = await forward2D(machine, inTok, outTok);
  assertClose(ll, 0, 1e-10, '2D Forward bitecho 101->101 = 0');
}

{
  const bitecho = loadJSON('t/machine/bitecho.json');
  const machine = prepareMachine(bitecho, {});
  const inTok = tokenize('101', machine.inputAlphabet);
  const outTok = tokenize('001', machine.outputAlphabet);
  const ll = await forward2D(machine, inTok, outTok);
  assert(ll === -Infinity, '2D Forward bitecho 101->001 = -inf');
}

// ============================================================
// Test: 2D Forward - unitindel (has silent transition to end state)
// ============================================================
{
  const unitindel = loadJSON('t/machine/unitindel.json');
  const machine = prepareMachine(unitindel, { ins: 0.1, no_ins: 0.9, del: 0.1, no_del: 0.9 });
  const inTok = tokenize('xx', machine.inputAlphabet);
  const outTok = tokenize('xxx', machine.outputAlphabet);
  const ll = await forward2D(machine, inTok, outTok);
  // Reference: boss gives -1.6869
  assertClose(ll, -1.6869, 1e-3, '2D Forward unitindel xx->xxx');
}

// ============================================================
// Test: 2D Viterbi
// ============================================================
console.log('Testing 2D Viterbi...');

{
  const machine = prepareMachine(bitnoise, { p: 0.9, q: 0.1 });
  const inTok = tokenize('001', machine.inputAlphabet);
  const outTok = tokenize('101', machine.outputAlphabet);
  const { score } = await viterbi2D(machine, inTok, outTok);
  // For single-state bitnoise, Viterbi = Forward = -2.51331
  assertClose(score, -2.51331, 1e-4, '2D Viterbi bitnoise 001->101');
}

{
  const unitindel = loadJSON('t/machine/unitindel.json');
  const machine = prepareMachine(unitindel, { ins: 0.1, no_ins: 0.9, del: 0.1, no_del: 0.9 });
  const inTok = tokenize('xx', machine.inputAlphabet);
  const outTok = tokenize('xxx', machine.outputAlphabet);
  const { score } = await viterbi2D(machine, inTok, outTok);
  // Reference: boss gives -2.82939
  assertClose(score, -2.82939, 1e-3, '2D Viterbi unitindel xx->xxx');
}

// ============================================================
// Test: Viterbi <= Forward invariant
// ============================================================
console.log('Testing Viterbi <= Forward invariant...');

{
  const unitindel = loadJSON('t/machine/unitindel.json');
  const machine = prepareMachine(unitindel, { ins: 0.1, no_ins: 0.9, del: 0.1, no_del: 0.9 });
  const inTok = tokenize('xx', machine.inputAlphabet);
  const outTok = tokenize('xxx', machine.outputAlphabet);
  const fwd = await forward2D(machine, inTok, outTok);
  const { score: vit } = await viterbi2D(machine, inTok, outTok);
  assert(vit <= fwd + 1e-10, `Viterbi (${vit}) <= Forward (${fwd})`);
}

// ============================================================
// Test: 1D Forward - generator
// ============================================================
console.log('Testing 1D Forward...');

{
  const gen101 = loadJSON('t/expect/generator101.json');
  const machine = prepareMachine(gen101, {});
  const outTok = tokenize('101', machine.outputAlphabet);
  const ll = await forward1D(machine, null, outTok);
  assertClose(ll, 0, 1e-10, '1D Forward generator 101 -> 101 = 0');
}

{
  const gen101 = loadJSON('t/expect/generator101.json');
  const machine = prepareMachine(gen101, {});
  const outTok = tokenize('001', machine.outputAlphabet);
  const ll = await forward1D(machine, null, outTok);
  assert(ll === -Infinity, '1D Forward generator 101 -> 001 = -inf');
}

// ============================================================
// Test: 2D Backward - Backward[start] ≈ Forward ll
// ============================================================
console.log('Testing Backward...');

{
  const machine = prepareMachine(bitnoise, { p: 0.9, q: 0.1 });
  const inTok = tokenize('001', machine.inputAlphabet);
  const outTok = tokenize('101', machine.outputAlphabet);
  const fwd = await forward2D(machine, inTok, outTok);
  const { logLikelihood: bwd } = await backward2D(machine, inTok, outTok);
  assertClose(bwd, fwd, 1e-8, '2D Backward[0,0] ≈ Forward ll for bitnoise');
}

{
  const unitindel = loadJSON('t/machine/unitindel.json');
  const machine = prepareMachine(unitindel, { ins: 0.1, no_ins: 0.9, del: 0.1, no_del: 0.9 });
  const inTok = tokenize('xx', machine.inputAlphabet);
  const outTok = tokenize('xxx', machine.outputAlphabet);
  const fwd = await forward2D(machine, inTok, outTok);
  const { logLikelihood: bwd } = await backward2D(machine, inTok, outTok);
  assertClose(bwd, fwd, 1e-6, '2D Backward[0,0] ≈ Forward ll for unitindel');
}

// ============================================================
// Test: 1D Backward[start] ≈ Forward ll
// ============================================================
{
  const gen101 = loadJSON('t/expect/generator101.json');
  const machine = prepareMachine(gen101, {});
  const outTok = tokenize('101', machine.outputAlphabet);
  const fwd = await forward1D(machine, null, outTok);
  const { logLikelihood: bwd } = await backward1D(machine, null, outTok);
  assertClose(bwd, fwd, 1e-10, '1D Backward[0] ≈ Forward ll for generator');
}

// ============================================================
// Test: Posteriors sum to 1
// ============================================================
console.log('Testing posteriors...');

{
  const machine = prepareMachine(bitnoise, { p: 0.9, q: 0.1 });
  const inTok = tokenize('001', machine.inputAlphabet);
  const outTok = tokenize('101', machine.outputAlphabet);
  const { logLikelihood, posteriors } = await posteriors2D(machine, inTok, outTok);
  assertClose(logLikelihood, -2.51331, 1e-4, 'posteriors2D loglikelihood');

  // For single-state bitnoise, posterior at each filled cell should be 1
  const Li = 3, Lo = 3;
  // At (3,3), state 0
  const postEnd = posteriors[(Li * (Lo + 1) + Lo) * S + 0];
  assertClose(postEnd, 1.0, 1e-6, 'posterior at (Li,Lo,0) = 1 for single-state machine');
}

// ============================================================
// Test: 1D = 2D consistency
// ============================================================
console.log('Testing 1D = 2D consistency...');

{
  // bitnoise as recognizer: input=null, output=tokens
  // For a transducer used in 1D mode with input=null,
  // only output-emit transitions (logTrans[0, outTok, :, :]) apply.
  // bitnoise has no output-only transitions, so 1D with input=null
  // should give -Infinity (bitnoise requires both input and output).
  // Instead test with a dedicated 1D machine.
}

// ============================================================
// Test: MachineBoss API
// ============================================================
console.log('Testing MachineBoss API...');

{
  const mb = await MachineBoss.create(bitnoise, { p: 0.9, q: 0.1 }, { backend: 'cpu' });
  assert(mb.backend === 'cpu', 'backend is cpu');
  assert(mb.nStates === 1, 'nStates = 1');

  const inTok = mb.tokenize('001', 'input');
  const outTok = mb.tokenize('101', 'output');

  const ll = await mb.forward(inTok, outTok);
  assertClose(ll, -2.51331, 1e-4, 'API forward');

  const { score } = await mb.viterbi(inTok, outTok);
  assertClose(score, -2.51331, 1e-4, 'API viterbi');

  const { logLikelihood } = await mb.posteriors(inTok, outTok);
  assertClose(logLikelihood, -2.51331, 1e-4, 'API posteriors loglikelihood');

  mb.destroy();
}

// ============================================================
// Summary
// ============================================================
console.log(`\nResults: ${passed} passed, ${failed} failed`);
if (failed > 0) {
  process.exit(1);
}
