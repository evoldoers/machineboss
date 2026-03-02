/**
 * 2D Viterbi dispatch using anti-diagonal wavefront on GPU.
 *
 * Uses max semiring with argmax tracking for traceback.
 * Argmax indices stored on GPU, traceback performed on CPU after readback.
 */

import {
  loadShader, createPipeline, createStorageBuffer,
  createEmptyBuffer, createUniformBuffer, createBindGroup,
  readBuffer,
} from './pipeline.mjs';
import { makeSemiring } from '../internal/logmath.mjs';
import { propagateSilent } from '../cpu/silent.mjs';

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

// Traceback type constants (must match WGSL shader)
const TB_NONE = 0;
const TB_MATCH = 1;
const TB_INSERT = 2;
const TB_DELETE = 3;
const TB_SILENT = 4;

/**
 * 2D Viterbi on GPU.
 *
 * @param {GPUDevice} device
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Uint32Array|null} inputSeq
 * @param {Uint32Array|null} outputSeq
 * @returns {Promise<{score: number, path: Array}>}
 */
export async function viterbi2DGPU(device, machine, inputSeq, outputSeq) {
  const { nStates: S, nInputTokens: nIn, nOutputTokens: nOut, logTrans } = machine;
  const semiring = makeSemiring('maxplus');
  const { plus, reduce } = semiring;

  const Li = inputSeq ? inputSeq.length : 0;
  const Lo = outputSeq ? outputSeq.length : 0;

  const dpSize = (Li + 1) * (Lo + 1) * S;
  const dpInit = new Float32Array(dpSize).fill(-Infinity);

  const silent64 = new Float64Array(S * S);
  for (let i = 0; i < S * S; i++) silent64[i] = logTrans[i];
  const silent32 = new Float32Array(S * S);
  for (let i = 0; i < S * S; i++) silent32[i] = logTrans[i];

  dpInit[0] = 0.0;
  const initCell = propagateSilent(
    new Float64Array(dpInit.subarray(0, S)),
    silent64, S, plus, reduce
  );
  for (let s = 0; s < S; s++) dpInit[s] = initCell[s];

  if (Li + Lo === 0) {
    return { score: dpInit[S - 1], path: [] };
  }

  const allIns = Li > 0 ? precomputeEmitTrans(inputSeq, logTrans, nIn, nOut, S, true, reduce)
    : new Float32Array(1 * S * S).fill(-Infinity);
  const allDel = Lo > 0 ? precomputeEmitTrans(outputSeq, logTrans, nIn, nOut, S, false, reduce)
    : new Float32Array(1 * S * S).fill(-Infinity);
  const allMatch = (Li > 0 && Lo > 0) ? precomputeMatch(inputSeq, outputSeq, logTrans, nIn, nOut, S)
    : new Float32Array(1 * S * S).fill(-Infinity);

  // Traceback buffers
  const tbSrcInit = new Int32Array(dpSize).fill(-1);
  const tbTypeInit = new Uint32Array(dpSize).fill(TB_NONE);

  const dpBuffer = createStorageBuffer(device, dpInit,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
  const insBuffer = createStorageBuffer(device, allIns);
  const delBuffer = createStorageBuffer(device, allDel);
  const matchBuffer = createStorageBuffer(device, allMatch);
  const silentBuffer = createStorageBuffer(device, silent32);
  const tbSrcBuffer = createStorageBuffer(device, tbSrcInit,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
  const tbTypeBuffer = createStorageBuffer(device, tbTypeInit,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

  const shaderSource = loadShader('wavefront-viterbi.wgsl');
  const pipeline = createPipeline(device, shaderSource, {
    S, LI: Li, LO: Lo,
  });

  for (let d = 1; d <= Li + Lo; d++) {
    const iMin = Math.max(0, d - Lo);
    const iMax = Math.min(Li, d);
    const nCells = iMax - iMin + 1;

    const diagParamsData = new Uint32Array([d]);
    const diagParamsBuffer = createUniformBuffer(device, diagParamsData);

    const bindGroup = createBindGroup(device, pipeline, 0,
      [dpBuffer, insBuffer, delBuffer, matchBuffer, silentBuffer,
       diagParamsBuffer, tbSrcBuffer, tbTypeBuffer]);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(nCells);
    pass.end();
    device.queue.submit([encoder.finish()]);

    diagParamsBuffer.destroy();
  }

  // Read back
  const dpData = await readBuffer(device, dpBuffer, dpSize * 4);
  const score = dpData[(Li * (Lo + 1) + Lo) * S + S - 1];

  // Read traceback buffers
  const readTbSrc = device.createBuffer({
    size: dpSize * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const readTbType = device.createBuffer({
    size: dpSize * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(tbSrcBuffer, 0, readTbSrc, 0, dpSize * 4);
  enc.copyBufferToBuffer(tbTypeBuffer, 0, readTbType, 0, dpSize * 4);
  device.queue.submit([enc.finish()]);

  await readTbSrc.mapAsync(GPUMapMode.READ);
  await readTbType.mapAsync(GPUMapMode.READ);
  const tbSrcData = new Int32Array(readTbSrc.getMappedRange().slice(0));
  const tbTypeData = new Uint32Array(readTbType.getMappedRange().slice(0));
  readTbSrc.unmap();
  readTbType.unmap();
  readTbSrc.destroy();
  readTbType.destroy();

  // Traceback on CPU
  const path = [];
  let i = Li, o = Lo, s = S - 1;

  while (i > 0 || o > 0 || s !== 0) {
    const idx = (i * (Lo + 1) + o) * S + s;
    const type = tbTypeData[idx];
    const src = tbSrcData[idx];

    if (type === TB_NONE || src < 0) break;

    if (type === TB_SILENT) {
      s = src;
    } else if (type === TB_MATCH) {
      path.push({ state: s, inputToken: i, outputToken: o });
      i--; o--; s = src;
    } else if (type === TB_INSERT) {
      path.push({ state: s, inputToken: i, outputToken: 0 });
      i--; s = src;
    } else if (type === TB_DELETE) {
      path.push({ state: s, inputToken: 0, outputToken: o });
      o--; s = src;
    }
  }
  path.reverse();

  // Cleanup
  dpBuffer.destroy();
  insBuffer.destroy();
  delBuffer.destroy();
  matchBuffer.destroy();
  silentBuffer.destroy();
  tbSrcBuffer.destroy();
  tbTypeBuffer.destroy();

  return { score, path };
}
