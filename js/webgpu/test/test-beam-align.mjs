/**
 * Tests for wavefront beam-Viterbi alignment.
 *
 * Run: node js/webgpu/test/test-beam-align.mjs
 */

import { readFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

import { prepareMachine, tokenize } from '../internal/machine-prep.mjs';
import { beamAlign2D } from '../cpu/beam-align.mjs';
import { viterbi2D } from '../cpu/viterbi-2d.mjs';

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
// Test: unitindel - beam-align matches exact Viterbi
// ============================================================
console.log('Testing beam-align on unitindel...');

{
  const unitindel = loadJSON('t/machine/unitindel.json');
  const machine = prepareMachine(unitindel, { ins: 0.1, no_ins: 0.9, del: 0.1, no_del: 0.9 });
  const inTok = tokenize('xx', machine.inputAlphabet);
  const outTok = tokenize('xxx', machine.outputAlphabet);

  const beamResult = await beamAlign2D(machine, inTok, outTok, 1000);
  // Reference: boss --viterbi gives -2.82939
  assertClose(beamResult.score, -2.82939, 1e-3, 'beam-align unitindel xx->xxx');

  // Compare with exact Viterbi
  const vitResult = await viterbi2D(machine, inTok, outTok);
  assertClose(beamResult.score, vitResult.score, 1e-6,
    'beam-align matches Viterbi on acyclic machine');
}

// ============================================================
// Test: bitnoise - beam-align matches exact Viterbi
// ============================================================
console.log('Testing beam-align on bitnoise...');

{
  const bitnoise = loadJSON('t/machine/bitnoise.json');
  const machine = prepareMachine(bitnoise, { p: 0.9, q: 0.1 });
  const inTok = tokenize('001', machine.inputAlphabet);
  const outTok = tokenize('101', machine.outputAlphabet);

  const beamResult = await beamAlign2D(machine, inTok, outTok, 1000);
  const vitResult = await viterbi2D(machine, inTok, outTok);
  assertClose(beamResult.score, vitResult.score, 1e-6,
    'beam-align matches Viterbi on bitnoise');
}

// ============================================================
// Test: TKF92
// ============================================================
console.log('Testing beam-align on TKF92...');

{
  const tkf92 = loadJSON('preset/tkf92-branch-prot-f81.json');
  const params = { t: 0.5, insRate: 0.01, delRate: 0.02, r: 0.3 };
  for (const aa of 'ACDEFGHIKLMNPQRSTVWY') params[`pi_${aa}`] = 0.05;

  const machine = prepareMachine(tkf92, params);
  const inTok = tokenize('AC', machine.inputAlphabet);
  const outTok = tokenize('AC', machine.outputAlphabet);

  const result = await beamAlign2D(machine, inTok, outTok, 1000);
  assert(isFinite(result.score), `TKF92 score is finite: ${result.score}`);
  assert(result.score < 0, `TKF92 score is negative: ${result.score}`);
  assert(result.path.length > 0, 'TKF92 path is non-empty');
}

// ============================================================
// Test: TKF92 with different sequences
// ============================================================
{
  const tkf92 = loadJSON('preset/tkf92-branch-prot-f81.json');
  const params = { t: 0.5, insRate: 0.01, delRate: 0.02, r: 0.3 };
  for (const aa of 'ACDEFGHIKLMNPQRSTVWY') params[`pi_${aa}`] = 0.05;

  const machine = prepareMachine(tkf92, params);
  const inTok = tokenize('ACD', machine.inputAlphabet);
  const outTok = tokenize('AD', machine.outputAlphabet);

  const result = await beamAlign2D(machine, inTok, outTok, 1000);
  assert(isFinite(result.score), `TKF92 ACD->AD score is finite: ${result.score}`);
  assert(result.path.length > 0, 'TKF92 ACD->AD path is non-empty');
}

// ============================================================
// Test: beam width monotonicity
// ============================================================
console.log('Testing beam width monotonicity...');

{
  const unitindel = loadJSON('t/machine/unitindel.json');
  const machine = prepareMachine(unitindel, { ins: 0.1, no_ins: 0.9, del: 0.1, no_del: 0.9 });
  const inTok = tokenize('xx', machine.inputAlphabet);
  const outTok = tokenize('xxx', machine.outputAlphabet);

  const small = await beamAlign2D(machine, inTok, outTok, 5);
  const large = await beamAlign2D(machine, inTok, outTok, 1000);
  assert(small.score <= large.score + 1e-10,
    `small beam (${small.score}) <= large beam (${large.score})`);
}

// ============================================================
// Test: empty sequences
// ============================================================
console.log('Testing edge cases...');

{
  const unitindel = loadJSON('t/machine/unitindel.json');
  const machine = prepareMachine(unitindel, { ins: 0.1, no_ins: 0.9, del: 0.1, no_del: 0.9 });

  // Empty input, empty output
  const result = await beamAlign2D(machine, new Uint32Array([]), new Uint32Array([]), 100);
  assert(isFinite(result.score), `empty->empty has finite score: ${result.score}`);
}

// ============================================================
// Summary
// ============================================================
console.log(`\nBeam-align tests: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
