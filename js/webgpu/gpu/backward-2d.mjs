/**
 * 2D Backward dispatch using anti-diagonal wavefront on GPU.
 *
 * Same as forward but traverses diagonals in reverse order,
 * starting from the terminal cell (Li, Lo).
 */

import {
  loadShader, createPipeline, createStorageBuffer,
  createEmptyBuffer, createUniformBuffer, createBindGroup,
  readBuffer,
} from './pipeline.mjs';
import { makeSemiring, NEG_INF } from '../internal/logmath.mjs';
import { propagateSilentBackward } from '../cpu/silent.mjs';

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
 * 2D Backward on GPU.
 *
 * @param {GPUDevice} device
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Uint32Array|null} inputSeq
 * @param {Uint32Array|null} outputSeq
 * @param {string} [semiringType='logsumexp']
 * @returns {Promise<{logLikelihood: number, bp: Float32Array}>}
 */
export async function backward2DGPU(device, machine, inputSeq, outputSeq, semiringType = 'logsumexp') {
  const { nStates: S, nInputTokens: nIn, nOutputTokens: nOut, logTrans } = machine;
  const semiring = makeSemiring(semiringType);
  const { plus, reduce } = semiring;

  const Li = inputSeq ? inputSeq.length : 0;
  const Lo = outputSeq ? outputSeq.length : 0;

  const bpSize = (Li + 1) * (Lo + 1) * S;
  const bpInit = new Float32Array(bpSize).fill(-Infinity);

  const silent64 = new Float64Array(S * S);
  for (let i = 0; i < S * S; i++) silent64[i] = logTrans[i];
  const silent32 = new Float32Array(S * S);
  for (let i = 0; i < S * S; i++) silent32[i] = logTrans[i];

  // Initialize terminal cell
  const termBase = (Li * (Lo + 1) + Lo) * S;
  bpInit[termBase + S - 1] = 0.0;
  const termCell = propagateSilentBackward(
    new Float64Array(bpInit.subarray(termBase, termBase + S)),
    silent64, S, plus, reduce
  );
  for (let s = 0; s < S; s++) bpInit[termBase + s] = termCell[s];

  if (Li + Lo === 0) {
    return { logLikelihood: bpInit[0], bp: bpInit };
  }

  // Precompute
  const allIns = Li > 0 ? precomputeEmitTrans(inputSeq, logTrans, nIn, nOut, S, true, reduce)
    : new Float32Array(1 * S * S).fill(-Infinity);
  const allDel = Lo > 0 ? precomputeEmitTrans(outputSeq, logTrans, nIn, nOut, S, false, reduce)
    : new Float32Array(1 * S * S).fill(-Infinity);
  const allMatch = (Li > 0 && Lo > 0) ? precomputeMatch(inputSeq, outputSeq, logTrans, nIn, nOut, S)
    : new Float32Array(1 * S * S).fill(-Infinity);

  const bpBuffer = createStorageBuffer(device, bpInit,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
  const insBuffer = createStorageBuffer(device, allIns);
  const delBuffer = createStorageBuffer(device, allDel);
  const matchBuffer = createStorageBuffer(device, allMatch);
  const silentBuffer = createStorageBuffer(device, silent32);

  const shaderSource = loadShader('wavefront-backward.wgsl');
  const pipeline = createPipeline(device, shaderSource, {
    S, LI: Li, LO: Lo,
  });

  // Dispatch per diagonal in reverse
  for (let d = Li + Lo - 1; d >= 0; d--) {
    const iMin = Math.max(0, d - Lo);
    const iMax = Math.min(Li, d);
    const nCells = iMax - iMin + 1;

    const diagParamsData = new Uint32Array([d]);
    const diagParamsBuffer = createUniformBuffer(device, diagParamsData);

    const bindGroup = createBindGroup(device, pipeline, 0,
      [bpBuffer, insBuffer, delBuffer, matchBuffer, silentBuffer, diagParamsBuffer]);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(nCells);
    pass.end();
    device.queue.submit([encoder.finish()]);

    diagParamsBuffer.destroy();
  }

  const resultData = await readBuffer(device, bpBuffer, bpSize * 4);
  const ll = resultData[0];

  bpBuffer.destroy();
  insBuffer.destroy();
  delBuffer.destroy();
  matchBuffer.destroy();
  silentBuffer.destroy();

  return { logLikelihood: ll, bp: resultData };
}
