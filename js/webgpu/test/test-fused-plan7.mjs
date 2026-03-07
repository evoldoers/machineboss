/**
 * Tests for fused Plan7+transducer kernel (CPU and GPU).
 *
 * Validates the HMMER parser and fused Forward/Viterbi against
 * reference values from the `boss` CLI. GPU tests compare against
 * CPU reference values with f32 tolerance.
 *
 * Run: node js/webgpu/test/test-fused-plan7.mjs
 * GPU: node --experimental-webgpu js/webgpu/test/test-fused-plan7.mjs
 */

import { readFileSync } from 'fs';
import { execSync } from 'child_process';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

import { MachineBoss } from '../machineboss-gpu.mjs';
import { parseHmmer, calcMatchOccupancy } from '../internal/hmmer-parse.mjs';
import { prepareMachine, tokenize } from '../internal/machine-prep.mjs';
import { buildFusedPlan7, fusedPlan7Forward, fusedPlan7Viterbi } from '../cpu/fused-plan7.mjs';
import { detectBackend } from '../internal/detect-backend.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..', '..', '..');

function loadJSON(path) {
  return JSON.parse(readFileSync(join(ROOT, path), 'utf8'));
}

function loadText(path) {
  return readFileSync(join(ROOT, path), 'utf8');
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
// Build a simple amino acid echo transducer for testing
// ============================================================
const AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'.split('');

function makeAaEchoTransducer() {
  return {
    state: [{
      id: 'S',
      trans: AA_ALPHABET.map(aa => ({ in: aa, out: aa, to: 'S' }))
    }]
  };
}

// ============================================================
// Test: HMMER parser
// ============================================================
console.log('Testing HMMER parser...');

{
  const hmmerText = loadText('t/hmmer/fn3.hmm');
  const model = parseHmmer(hmmerText);

  assert(model.alph.length === 20, `HMMER alphabet size = 20 (got ${model.alph.length})`);
  assert(model.alph[0] === 'A', `First aa = A (got ${model.alph[0]})`);
  assert(model.nodes.length > 0, `Has nodes (got ${model.nodes.length})`);

  // fn3 is Fibronectin type III (PF00041): ~85 nodes
  assert(model.nodes.length >= 80 && model.nodes.length <= 100,
    `fn3 K in range 80-100 (got ${model.nodes.length})`);

  // Check node structure
  const node = model.nodes[0];
  assert(node.match_emit.length === 20, `Node 0 match_emit length = 20`);
  assert(node.ins_emit.length === 20, `Node 0 ins_emit length = 20`);
  assert(node.m_to_m >= 0 && node.m_to_m <= 1, `m_to_m is probability`);

  // Match occupancy
  const occ = calcMatchOccupancy(model);
  assert(occ.length === model.nodes.length, `Occupancy length = K`);
  assert(occ[0] === 0, `occ[0] = 0`);

  // Null model
  assert(model.null_emit.length === 20, `Null model has 20 entries`);
  assertClose(model.null_emit[0], 0.0825, 1e-4, `null_emit[A] = SwissProt bg freq`);
}

// ============================================================
// Test: Build fused data
// ============================================================
console.log('Testing fused Plan7 build...');

{
  const hmmerText = loadText('t/hmmer/fn3.hmm');
  const model = parseHmmer(hmmerText);
  const aaEcho = makeAaEchoTransducer();
  const prepared = prepareMachine(aaEcho, {});

  const fused = buildFusedPlan7(model, prepared);

  assert(fused.K === model.nodes.length, `fused K = ${model.nodes.length}`);
  assert(fused.n_aa === 20, `fused n_aa = 20`);
  assert(fused.S_td === 1, `fused S_td = 1 (echo has 1 state)`);
  assert(fused.log_b_entry.length === fused.K, `log_b_entry length = K`);
  assert(fused.log_match_emit.length === fused.K * fused.n_aa, `log_match_emit flat length`);
}

// ============================================================
// Test: Fused Forward with aa echo transducer
// ============================================================
console.log('Testing fused Plan7 Forward...');

{
  const hmmerText = loadText('t/hmmer/fn3.hmm');
  const model = parseHmmer(hmmerText);
  const aaEcho = makeAaEchoTransducer();
  const prepared = prepareMachine(aaEcho, {});

  const fused = buildFusedPlan7(model, prepared);
  const outSeq = tokenize('ACDE', prepared.outputAlphabet);

  const ll = fusedPlan7Forward(fused, outSeq);
  assert(isFinite(ll), `Forward with aa echo gives finite result: ${ll}`);
  assert(ll < 0, `Forward log-likelihood is negative: ${ll}`);
}

// ============================================================
// Test: Fused Forward vs boss --hmmer-plan7 --compose
// ============================================================
console.log('Testing fused Forward vs boss compose...');

{
  const hmmerText = loadText('t/hmmer/fn3.hmm');
  const model = parseHmmer(hmmerText);
  const bitecho = loadJSON('t/machine/bitecho.json');
  const prepared = prepareMachine(bitecho, {});

  const fused = buildFusedPlan7(model, prepared);
  const outSeq = tokenize('010101', prepared.outputAlphabet);

  const ll = fusedPlan7Forward(fused, outSeq);

  // boss --hmmer-plan7 fn3.hmm --compose bitecho.json --output-chars 010101 -L gives -Infinity
  // (amino acids don't match 0/1 input tokens, so no valid paths)
  assert(ll === -Infinity || ll < -1e30,
    `Forward fn3+bitecho should be -Infinity or very negative: ${ll}`);
}

// ============================================================
// Test: Fused Forward vs reference values (aa echo)
// ============================================================
// Reference values from standard Forward on the Plan7 machine (JS float64).
// The Plan7 machine has cycles so boss CLI can't run Forward directly,
// but the standard JS Forward handles cycles via iterative convergence.
console.log('Testing fused Forward vs reference values...');

{
  const hmmerText = loadText('t/hmmer/fn3.hmm');
  const model = parseHmmer(hmmerText);
  const aaEcho = makeAaEchoTransducer();
  const prepared = prepareMachine(aaEcho, {});
  const fused = buildFusedPlan7(model, prepared);

  // Reference: standard Forward on fn3 Plan7 generator + aa echo transducer
  const cases = [
    { seq: '',     ref: -15.776 },
    { seq: 'A',    ref: -17.234 },
    { seq: 'ACDE', ref: -25.701 },
  ];

  for (const { seq, ref } of cases) {
    const outSeq = seq ? tokenize(seq, prepared.outputAlphabet) : new Uint32Array(0);
    const ll = fusedPlan7Forward(fused, outSeq);
    assertClose(ll, ref, 0.01,
      `Forward fn3+aaEcho '${seq}' matches reference`);
  }
}

// ============================================================
// Test: Viterbi <= Forward
// ============================================================
console.log('Testing Viterbi <= Forward...');

{
  const hmmerText = loadText('t/hmmer/fn3.hmm');
  const model = parseHmmer(hmmerText);
  const aaEcho = makeAaEchoTransducer();
  const prepared = prepareMachine(aaEcho, {});

  const fused = buildFusedPlan7(model, prepared);
  const outSeq = tokenize('ACDE', prepared.outputAlphabet);

  const fwdLl = fusedPlan7Forward(fused, outSeq);
  const vitLl = fusedPlan7Viterbi(fused, outSeq);

  assert(isFinite(fwdLl), `Forward is finite: ${fwdLl}`);
  assert(isFinite(vitLl), `Viterbi is finite: ${vitLl}`);
  assert(vitLl <= fwdLl + 1e-10,
    `Viterbi (${vitLl}) <= Forward (${fwdLl})`);
}

// ============================================================
// Test: Forward != Viterbi (semiring difference)
// ============================================================
console.log('Testing semiring difference...');

{
  const hmmerText = loadText('t/hmmer/fn3.hmm');
  const model = parseHmmer(hmmerText);
  const aaEcho = makeAaEchoTransducer();
  const prepared = prepareMachine(aaEcho, {});

  const fused = buildFusedPlan7(model, prepared);
  const outSeq = tokenize('ACDE', prepared.outputAlphabet);

  const fwdLl = fusedPlan7Forward(fused, outSeq);
  const vitLl = fusedPlan7Viterbi(fused, outSeq);

  // For a multi-state profile HMM, Forward > Viterbi (multiple paths contribute)
  assert(fwdLl > vitLl + 1e-6,
    `Forward (${fwdLl}) > Viterbi (${vitLl}) — different semirings`);
}

// ============================================================
// Test: Empty sequence
// ============================================================
console.log('Testing empty sequence...');

{
  const hmmerText = loadText('t/hmmer/fn3.hmm');
  const model = parseHmmer(hmmerText);
  const aaEcho = makeAaEchoTransducer();
  const prepared = prepareMachine(aaEcho, {});

  const fused = buildFusedPlan7(model, prepared);
  const outSeq = new Uint32Array(0);

  const ll = fusedPlan7Forward(fused, outSeq);
  assert(isFinite(ll), `Empty sequence Forward is finite: ${ll}`);
  assert(ll < 0, `Empty sequence Forward is negative: ${ll}`);
}

// ============================================================
// Test: MachineBoss API
// ============================================================
console.log('Testing MachineBoss fused Plan7 API...');

{
  const hmmerText = loadText('t/hmmer/fn3.hmm');
  const aaEcho = makeAaEchoTransducer();

  const mb = await MachineBoss.createFusedPlan7(hmmerText, aaEcho);
  assert(mb.backend === 'cpu', `backend is cpu`);
  assert(mb._fusedPlan7 !== undefined, `has _fusedPlan7 data`);

  const outTok = mb.tokenize('ACDE', 'output');
  const fwdLl = await mb.fusedForward(outTok);
  assert(isFinite(fwdLl), `API fusedForward is finite: ${fwdLl}`);

  const vitLl = await mb.fusedViterbi(outTok);
  assert(vitLl <= fwdLl + 1e-10, `API fusedViterbi <= fusedForward`);

  mb.destroy();
}

// ============================================================
// Test: MachineBoss batch API (CPU)
// ============================================================
console.log('Testing MachineBoss batch API (CPU)...');

{
  const hmmerText = loadText('t/hmmer/fn3.hmm');
  const aaEcho = makeAaEchoTransducer();

  const mb = await MachineBoss.createFusedPlan7(hmmerText, aaEcho, {}, { backend: 'cpu' });

  const seqs = [
    new Uint32Array(0),                              // empty
    tokenize('A', mb.outputAlphabet),                 // single aa
    tokenize('ACDE', mb.outputAlphabet),              // short
    tokenize('VLIWFYH', mb.outputAlphabet),           // different aas
  ];

  const fwdBatch = await mb.fusedForwardBatch(seqs);
  assert(fwdBatch.length === 4, `batch returns 4 results`);
  for (let i = 0; i < 4; i++) {
    const expected = fusedPlan7Forward(mb._fusedPlan7, seqs[i]);
    assertClose(fwdBatch[i], expected, 1e-10,
      `batch Forward[${i}] matches single Forward`);
  }

  const vitBatch = await mb.fusedViterbiBatch(seqs);
  for (let i = 0; i < 4; i++) {
    const expected = fusedPlan7Viterbi(mb._fusedPlan7, seqs[i]);
    assertClose(vitBatch[i], expected, 1e-10,
      `batch Viterbi[${i}] matches single Viterbi`);
  }

  mb.destroy();
}

// ============================================================
// GPU tests (auto-skip if WebGPU unavailable)
// ============================================================
const GPU_TOL = 0.5; // f32 vs f64 accumulation tolerance

const detected = await detectBackend();
const hasGPU = detected.backend === 'webgpu';

if (!hasGPU) {
  console.log('\nWebGPU not available — skipping GPU tests.');
  console.log('  (Run with: node --experimental-webgpu js/webgpu/test/test-fused-plan7.mjs)');
} else {
  console.log('\nWebGPU available — running GPU tests...');

  const hmmerText = loadText('t/hmmer/fn3.hmm');
  const aaEcho = makeAaEchoTransducer();
  const prepared = prepareMachine(aaEcho, {});
  const fused = buildFusedPlan7(parseHmmer(hmmerText), prepared);

  const testSeqs = [
    { name: 'empty', seq: new Uint32Array(0) },
    { name: 'A', seq: tokenize('A', prepared.outputAlphabet) },
    { name: 'ACDE', seq: tokenize('ACDE', prepared.outputAlphabet) },
    { name: 'VLIWFYH', seq: tokenize('VLIWFYH', prepared.outputAlphabet) },
  ];

  // Compute CPU references
  const cpuFwd = testSeqs.map(t => fusedPlan7Forward(fused, t.seq));
  const cpuVit = testSeqs.map(t => fusedPlan7Viterbi(fused, t.seq));

  // GPU batch Forward
  console.log('Testing GPU batch Forward...');
  {
    const { fusedPlan7BatchGPU } = await import('../gpu/fused-plan7-batch.mjs');
    const gpuFwd = await fusedPlan7BatchGPU(detected.device, fused,
      testSeqs.map(t => t.seq), 'logsumexp');

    for (let i = 0; i < testSeqs.length; i++) {
      if (cpuFwd[i] === -Infinity) {
        assert(gpuFwd[i] === -Infinity || gpuFwd[i] < -1e30,
          `GPU batch Forward '${testSeqs[i].name}' is -Inf`);
      } else {
        assertClose(gpuFwd[i], cpuFwd[i], GPU_TOL,
          `GPU batch Forward '${testSeqs[i].name}'`);
      }
    }
  }

  // GPU batch Viterbi
  console.log('Testing GPU batch Viterbi...');
  {
    const { fusedPlan7BatchGPU } = await import('../gpu/fused-plan7-batch.mjs');
    const gpuVit = await fusedPlan7BatchGPU(detected.device, fused,
      testSeqs.map(t => t.seq), 'maxplus');

    for (let i = 0; i < testSeqs.length; i++) {
      if (cpuVit[i] === -Infinity) {
        assert(gpuVit[i] === -Infinity || gpuVit[i] < -1e30,
          `GPU batch Viterbi '${testSeqs[i].name}' is -Inf`);
      } else {
        assertClose(gpuVit[i], cpuVit[i], GPU_TOL,
          `GPU batch Viterbi '${testSeqs[i].name}'`);
      }
    }
  }

  // GPU single Forward
  console.log('Testing GPU single Forward...');
  {
    const { fusedPlan7SingleGPU } = await import('../gpu/fused-plan7-single.mjs');
    const acdeSeq = tokenize('ACDE', prepared.outputAlphabet);
    const gpuFwd = await fusedPlan7SingleGPU(detected.device, fused, acdeSeq, 'logsumexp');
    assertClose(gpuFwd, cpuFwd[2], GPU_TOL, `GPU single Forward 'ACDE'`);
  }

  // GPU single Viterbi
  console.log('Testing GPU single Viterbi...');
  {
    const { fusedPlan7SingleGPU } = await import('../gpu/fused-plan7-single.mjs');
    const acdeSeq = tokenize('ACDE', prepared.outputAlphabet);
    const gpuVit = await fusedPlan7SingleGPU(detected.device, fused, acdeSeq, 'maxplus');
    assertClose(gpuVit, cpuVit[2], GPU_TOL, `GPU single Viterbi 'ACDE'`);
  }

  // GPU batch-vs-single consistency
  console.log('Testing GPU batch vs single consistency...');
  {
    const { fusedPlan7BatchGPU } = await import('../gpu/fused-plan7-batch.mjs');
    const { fusedPlan7SingleGPU } = await import('../gpu/fused-plan7-single.mjs');
    const acdeSeq = tokenize('ACDE', prepared.outputAlphabet);

    const batchResult = await fusedPlan7BatchGPU(detected.device, fused, [acdeSeq], 'logsumexp');
    const singleResult = await fusedPlan7SingleGPU(detected.device, fused, acdeSeq, 'logsumexp');
    assertClose(batchResult[0], singleResult, 0.01,
      `GPU batch[0] matches single for 'ACDE'`);
  }

  // GPU Viterbi <= Forward
  console.log('Testing GPU Viterbi <= Forward...');
  {
    const { fusedPlan7BatchGPU } = await import('../gpu/fused-plan7-batch.mjs');
    const seqs = testSeqs.filter(t => t.seq.length > 0).map(t => t.seq);
    const gpuFwd = await fusedPlan7BatchGPU(detected.device, fused, seqs, 'logsumexp');
    const gpuVit = await fusedPlan7BatchGPU(detected.device, fused, seqs, 'maxplus');

    for (let i = 0; i < seqs.length; i++) {
      assert(gpuVit[i] <= gpuFwd[i] + 0.01,
        `GPU Viterbi (${gpuVit[i]}) <= Forward (${gpuFwd[i]})`);
    }
  }

  detected.device.destroy();
}

// ============================================================
// Summary
// ============================================================
console.log(`\nResults: ${passed} passed, ${failed} failed`);
if (failed > 0) {
  process.exit(1);
}
