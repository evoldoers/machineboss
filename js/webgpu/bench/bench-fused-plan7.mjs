/**
 * Benchmark: Fused Plan7+transducer Forward/Viterbi (CPU, Float64).
 *
 * Measures fused Plan7 kernel performance across HMMER profile sizes
 * and output sequence lengths. Compares Forward and Viterbi.
 *
 * Usage:
 *   node js/webgpu/bench/bench-fused-plan7.mjs [--reps N] [--json]
 *
 * Output: CSV (default) or JSON timing results.
 */

import { readFileSync, existsSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

import { parseHmmer } from '../internal/hmmer-parse.mjs';
import { prepareMachine, tokenize } from '../internal/machine-prep.mjs';
import { buildFusedPlan7, fusedPlan7Forward, fusedPlan7Viterbi } from '../cpu/fused-plan7.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..', '..', '..');

// Parse CLI args
const args = process.argv.slice(2);
const N_REPS = args.includes('--reps') ? parseInt(args[args.indexOf('--reps') + 1]) : 5;
const JSON_OUT = args.includes('--json');

// Amino acid echo transducer (1 state, identity)
const AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'.split('');
function makeAaEchoTransducer() {
  return {
    state: [{
      id: 'S',
      trans: AA_ALPHABET.map(aa => ({ in: aa, out: aa, to: 'S' })),
    }],
  };
}

// Bitnoise transducer (if available)
function loadBitnoiseTransducer() {
  const path = join(ROOT, 't', 'machine', 'bitnoise.json');
  if (!existsSync(path)) return null;
  return JSON.parse(readFileSync(path, 'utf8'));
}

function medianTime(times) {
  const sorted = [...times].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function bench(fn, n_reps = N_REPS) {
  // Warmup
  const result = fn();

  const times = [];
  for (let i = 0; i < n_reps; i++) {
    const t0 = performance.now();
    fn();
    const elapsed = (performance.now() - t0) / 1000.0;
    times.push(elapsed);
  }

  return { median: medianTime(times), times, result };
}

function generateRandomAA(len) {
  const chars = [];
  for (let i = 0; i < len; i++) {
    chars.push(AA_ALPHABET[Math.floor(Math.random() * AA_ALPHABET.length)]);
  }
  return chars.join('');
}

// Run benchmarks
const results = [];

const hmmFiles = [
  { name: 'fn3', path: join(ROOT, 't', 'hmmer', 'fn3.hmm') },
];
// Try larger profile if available
const pf516 = join(ROOT, 'examples', 'PF00516.hmm');
if (existsSync(pf516)) {
  hmmFiles.push({ name: 'PF00516', path: pf516 });
}

const transducers = [
  { name: 'aa_echo', json: makeAaEchoTransducer(), params: {} },
];
const bitnoise = loadBitnoiseTransducer();
if (bitnoise) {
  const paramsPath = join(ROOT, 't', 'io', 'params.json');
  const params = existsSync(paramsPath) ? JSON.parse(readFileSync(paramsPath, 'utf8')) : {};
  transducers.push({ name: 'bitnoise', json: bitnoise, params });
}

const outputLengths = [5, 10, 20, 50, 100];

if (!JSON_OUT) {
  console.log('hmm,K,transducer,S_td,out_len,algorithm,median_sec,loglike');
}

for (const hmm of hmmFiles) {
  const hmmerText = readFileSync(hmm.path, 'utf8');
  const model = parseHmmer(hmmerText);
  const K = model.nodes.length;

  for (const td of transducers) {
    const prepared = prepareMachine(td.json, td.params);
    const S_td = prepared.nStates;

    let fused;
    try {
      fused = buildFusedPlan7(model, prepared);
    } catch (e) {
      process.stderr.write(`  SKIP ${hmm.name}+${td.name}: ${e.message}\n`);
      continue;
    }

    process.stderr.write(`\n=== ${hmm.name} (K=${K}) + ${td.name} (S_td=${S_td}) ===\n`);

    for (const outLen of outputLengths) {
      // Generate random output sequence appropriate for the transducer
      let outSeq;
      if (td.name === 'aa_echo') {
        const seq = generateRandomAA(outLen);
        outSeq = tokenize(seq, prepared.outputAlphabet);
      } else {
        // Binary output for bitnoise-style transducers
        const seq = Array.from({ length: outLen }, () => Math.random() < 0.5 ? '0' : '1').join('');
        outSeq = tokenize(seq, prepared.outputAlphabet);
      }

      // Forward
      const fwd = bench(() => fusedPlan7Forward(fused, outSeq));
      const row = {
        hmm: hmm.name, K, transducer: td.name, S_td,
        out_len: outLen, algorithm: 'Forward',
        median_sec: fwd.median, loglike: fwd.result,
      };
      results.push(row);
      if (!JSON_OUT) {
        console.log(`${row.hmm},${row.K},${row.transducer},${row.S_td},${row.out_len},${row.algorithm},${row.median_sec.toFixed(6)},${row.loglike.toFixed(4)}`);
      }

      // Viterbi
      const vit = bench(() => fusedPlan7Viterbi(fused, outSeq));
      const vitRow = {
        hmm: hmm.name, K, transducer: td.name, S_td,
        out_len: outLen, algorithm: 'Viterbi',
        median_sec: vit.median, loglike: vit.result,
      };
      results.push(vitRow);
      if (!JSON_OUT) {
        console.log(`${vitRow.hmm},${vitRow.K},${vitRow.transducer},${vitRow.S_td},${vitRow.out_len},${vitRow.algorithm},${vitRow.median_sec.toFixed(6)},${vitRow.loglike.toFixed(4)}`);
      }

      process.stderr.write(`  out=${outLen}: fwd=${fwd.median.toFixed(4)}s vit=${vit.median.toFixed(4)}s\n`);
    }
  }
}

if (JSON_OUT) {
  console.log(JSON.stringify(results, null, 2));
}
