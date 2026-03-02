/**
 * 2D Forward dispatch using anti-diagonal wavefront on GPU.
 *
 * Steps:
 * 1. Precompute transition matrices on CPU (insert, delete, match)
 * 2. Upload DP grid, transition matrices, and silent matrix to GPU
 * 3. Initialize cell (0,0) with silent closure
 * 4. Loop over diagonals d = 1..Li+Lo, dispatching one compute shader per diagonal
 * 5. Read back result from cell (Li, Lo, S-1)
 */

import {
  loadShader, createPipeline, createStorageBuffer,
  createEmptyBuffer, createUniformBuffer, createBindGroup,
  readBuffer,
} from './pipeline.mjs';
import { makeSemiring, NEG_INF } from '../internal/logmath.mjs';
import { propagateSilent } from '../cpu/silent.mjs';

/**
 * Precompute per-position emission-weighted transition matrices on CPU.
 */
function precomputeEmitTrans(seq, logTrans, nIn, nOut, S, isInput, reduce) {
  const nTok = isInput ? nIn : nOut;
  const L = seq.length;
  const result = new Float32Array(L * S * S);
  const tmp = new Float64Array(nTok);

  for (let p = 0; p < L; p++) {
    for (let src = 0; src < S; src++) {
      for (let dst = 0; dst < S; dst++) {
        for (let tok = 0; tok < nTok; tok++) {
          const emission = (tok === seq[p]) ? 0.0 : -Infinity;
          let idx;
          if (isInput) idx = ((tok * nOut + 0) * S + src) * S + dst;
          else idx = ((0 * nOut + tok) * S + src) * S + dst;
          tmp[tok] = emission + logTrans[idx];
        }
        result[(p * S + src) * S + dst] = reduce(tmp);
      }
    }
  }
  return result;
}

function precomputeMatch(inputSeq, outputSeq, logTrans, nIn, nOut, S) {
  const Li = inputSeq.length;
  const Lo = outputSeq.length;
  const result = new Float32Array(Li * Lo * S * S);
  for (let i = 0; i < Li; i++) {
    const inTok = inputSeq[i];
    for (let o = 0; o < Lo; o++) {
      const outTok = outputSeq[o];
      for (let src = 0; src < S; src++) {
        for (let dst = 0; dst < S; dst++) {
          const idx = ((inTok * nOut + outTok) * S + src) * S + dst;
          result[((i * Lo + o) * S + src) * S + dst] = logTrans[idx];
        }
      }
    }
  }
  return result;
}

/**
 * 2D Forward on GPU.
 *
 * @param {GPUDevice} device
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Uint32Array|null} inputSeq
 * @param {Uint32Array|null} outputSeq
 * @param {string} [semiringType='logsumexp']
 * @returns {Promise<number>} log-likelihood
 */
export async function forward2DGPU(device, machine, inputSeq, outputSeq, semiringType = 'logsumexp') {
  const { nStates: S, nInputTokens: nIn, nOutputTokens: nOut, logTrans } = machine;
  const semiring = makeSemiring(semiringType);
  const { plus, reduce } = semiring;

  const Li = inputSeq ? inputSeq.length : 0;
  const Lo = outputSeq ? outputSeq.length : 0;

  // Initialize DP grid on CPU
  const dpSize = (Li + 1) * (Lo + 1) * S;
  const dpInit = new Float32Array(dpSize).fill(-Infinity);

  // Silent transitions
  const silent64 = new Float64Array(S * S);
  for (let i = 0; i < S * S; i++) silent64[i] = logTrans[i];
  const silent32 = new Float32Array(S * S);
  for (let i = 0; i < S * S; i++) silent32[i] = logTrans[i];

  // Initialize (0,0) with silent closure
  dpInit[0] = 0.0;
  const initCell = propagateSilent(
    new Float64Array(dpInit.subarray(0, S)),
    silent64, S, plus, reduce
  );
  for (let s = 0; s < S; s++) dpInit[s] = initCell[s];

  if (Li + Lo === 0) {
    return dpInit[S - 1];
  }

  // Precompute transition matrices
  const allIns = Li > 0 ? precomputeEmitTrans(inputSeq, logTrans, nIn, nOut, S, true, reduce)
    : new Float32Array(1 * S * S).fill(-Infinity);
  const allDel = Lo > 0 ? precomputeEmitTrans(outputSeq, logTrans, nIn, nOut, S, false, reduce)
    : new Float32Array(1 * S * S).fill(-Infinity);
  const allMatch = (Li > 0 && Lo > 0) ? precomputeMatch(inputSeq, outputSeq, logTrans, nIn, nOut, S)
    : new Float32Array(1 * S * S).fill(-Infinity);

  // Upload to GPU
  const dpBuffer = createStorageBuffer(device, dpInit,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
  const insBuffer = createStorageBuffer(device, allIns);
  const delBuffer = createStorageBuffer(device, allDel);
  const matchBuffer = createStorageBuffer(device, allMatch);
  const silentBuffer = createStorageBuffer(device, silent32);

  // Create pipeline
  const shaderSource = loadShader('wavefront-forward.wgsl');
  const pipeline = createPipeline(device, shaderSource, {
    S, LI: Li, LO: Lo,
  });

  // Dispatch per diagonal
  for (let d = 1; d <= Li + Lo; d++) {
    const iMin = Math.max(0, d - Lo);
    const iMax = Math.min(Li, d);
    const nCells = iMax - iMin + 1;

    const diagParamsData = new Uint32Array([d]);
    const diagParamsBuffer = createUniformBuffer(device, diagParamsData);

    const bindGroup = createBindGroup(device, pipeline, 0,
      [dpBuffer, insBuffer, delBuffer, matchBuffer, silentBuffer, diagParamsBuffer]);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(nCells);
    pass.end();
    device.queue.submit([encoder.finish()]);

    diagParamsBuffer.destroy();
  }

  // Read result
  const resultData = await readBuffer(device, dpBuffer, dpSize * 4);
  const ll = resultData[(Li * (Lo + 1) + Lo) * S + S - 1];

  // Cleanup
  dpBuffer.destroy();
  insBuffer.destroy();
  delBuffer.destroy();
  matchBuffer.destroy();
  silentBuffer.destroy();

  return ll;
}
