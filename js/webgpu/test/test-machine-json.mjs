/**
 * Roundtrip idempotency tests for Machine and PreparedMachine JSON serialization.
 *
 * Run: node js/webgpu/test/test-machine-json.mjs
 */

import { readFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

import { Machine } from '../internal/machine.mjs';
import { prepareMachine, toMachineJSON } from '../internal/machine-prep.mjs';

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

function assertEqual(actual, expected, msg) {
  const a = JSON.stringify(actual, null, 2);
  const b = JSON.stringify(expected, null, 2);
  if (a !== b) {
    console.error(`  FAIL: ${msg}`);
    console.error(`    expected: ${b.slice(0, 200)}`);
    console.error(`    actual:   ${a.slice(0, 200)}`);
    failed++;
  } else {
    passed++;
  }
}

// ============================================================
// Test: Machine.fromJSON().toJSON() roundtrip (unevaluated)
// ============================================================
console.log('Testing Machine unevaluated roundtrip...');

for (const name of ['bitecho', 'bitstutter']) {
  const raw = loadJSON(`t/machine/${name}.json`);
  const m1 = Machine.fromJSON(raw);
  const j1 = m1.toJSON();
  const m2 = Machine.fromJSON(j1);
  const j2 = m2.toJSON();
  assertEqual(j1, j2, `${name} unevaluated roundtrip`);
}

// ============================================================
// Test: Machine class properties
// ============================================================
console.log('Testing Machine properties...');

{
  const m = Machine.fromJSON(loadJSON('t/machine/bitecho.json'));
  assert(m.nStates === 1, `bitecho nStates === 1, got ${m.nStates}`);
  assertEqual(m.inputAlphabet(), ['0', '1'], 'bitecho inputAlphabet');
  assertEqual(m.outputAlphabet(), ['0', '1'], 'bitecho outputAlphabet');
  assert(m.nTransitions === 2, `bitecho nTransitions === 2, got ${m.nTransitions}`);
}

{
  const m = Machine.fromJSON(loadJSON('t/machine/unitindel.json'));
  assert(m.nStates === 2, `unitindel nStates === 2, got ${m.nStates}`);
  assertEqual(m.inputAlphabet(), ['x'], 'unitindel inputAlphabet');
  assertEqual(m.outputAlphabet(), ['x'], 'unitindel outputAlphabet');
  assert(m.startState === 0, 'unitindel startState === 0');
  assert(m.endState === 1, 'unitindel endState === 1');
}

// ============================================================
// Test: PreparedMachine roundtrip via toMachineJSON
// ============================================================
console.log('Testing PreparedMachine roundtrip...');

const paramMachines = {
  unitindel: { ins: 0.1, no_ins: 0.9, del: 0.1, no_del: 0.9 },
  bitnoise: { p: 0.9, q: 0.1 },
  bsc: { e: 0.1 },
  bitstutter: {},
};

for (const [name, params] of Object.entries(paramMachines)) {
  const raw = loadJSON(`t/machine/${name}.json`);
  const p1 = prepareMachine(raw, params);
  const j1 = toMachineJSON(p1);
  const p2 = prepareMachine(j1);
  const j2 = toMachineJSON(p2);
  assertEqual(j1, j2, `${name} PreparedMachine roundtrip`);
}

// ============================================================
// Test: toMachineJSON preserves structure
// ============================================================
console.log('Testing toMachineJSON structure...');

{
  const raw = loadJSON('t/machine/bitecho.json');
  const p = prepareMachine(raw);
  const j = toMachineJSON(p);
  assert(j.state.length === 1, `bitecho reconstructed has 1 state`);
  assert(j.state[0].trans.length === 2, `bitecho reconstructed has 2 transitions`);
  // Weights should be omitted (=== 1)
  for (const t of j.state[0].trans) {
    assert(!('weight' in t), `bitecho transition should have no weight key`);
  }
}

// ============================================================
// Test: Machine with state name references
// ============================================================
console.log('Testing state name resolution...');

{
  const m = Machine.fromJSON(loadJSON('t/machine/bitstutter.json'));
  // bitstutter has states S, S0, S1, E — all name refs resolved to integers
  assert(m.nStates === 4, `bitstutter nStates === 4, got ${m.nStates}`);
  for (const s of m.state) {
    for (const t of s.trans) {
      assert(typeof t.dest === 'number', `dest should be number, got ${typeof t.dest}`);
    }
  }
}

// ============================================================
// Summary
// ============================================================
console.log(`\nMachine JSON tests: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
