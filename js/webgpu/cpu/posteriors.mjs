/**
 * Forward-Backward posterior computation (CPU fallback).
 *
 * posterior[i, o, s] = exp(fwd[i, o, s] + bwd[i, o, s] - logLikelihood)
 *
 * Returns per-position, per-state posterior probabilities.
 */

import { NEG_INF } from '../internal/logmath.mjs';
import { forward1DFull } from './forward-1d.mjs';
import { backward1D } from './backward-1d.mjs';
import { forward2DFull } from './forward-2d.mjs';
import { backward2D } from './backward-2d.mjs';

/**
 * Compute 1D Forward-Backward posteriors.
 *
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Uint32Array|null} inputSeq
 * @param {Uint32Array|null} outputSeq
 * @returns {Promise<{logLikelihood: number, posteriors: Float32Array}>}
 *   posteriors is shape (L+1, S), row-major Float32Array
 */
export async function posteriors1D(machine, inputSeq, outputSeq) {
  const S = machine.nStates;
  const seq = (inputSeq === null || inputSeq === undefined) ? outputSeq : inputSeq;
  const L = seq.length;

  const [fwdResult, bwdResult] = await Promise.all([
    forward1DFull(machine, inputSeq, outputSeq),
    backward1D(machine, inputSeq, outputSeq),
  ]);

  const { logLikelihood, dp: fwd } = fwdResult;
  const { bp: bwd } = bwdResult;

  const size = (L + 1) * S;
  const posteriors = new Float32Array(size);

  for (let p = 0; p <= L; p++) {
    for (let s = 0; s < S; s++) {
      const idx = p * S + s;
      const logPost = fwd[idx] + bwd[idx] - logLikelihood;
      posteriors[idx] = logPost === NEG_INF ? 0 : Math.exp(logPost);
    }
  }

  return { logLikelihood, posteriors };
}

/**
 * Compute 2D Forward-Backward posteriors.
 *
 * @param {import('../internal/machine-prep.mjs').PreparedMachine} machine
 * @param {Uint32Array|null} inputSeq
 * @param {Uint32Array|null} outputSeq
 * @returns {Promise<{logLikelihood: number, posteriors: Float32Array}>}
 *   posteriors is shape ((Li+1)*(Lo+1), S), row-major Float32Array
 */
export async function posteriors2D(machine, inputSeq, outputSeq) {
  const S = machine.nStates;
  const Li = inputSeq ? inputSeq.length : 0;
  const Lo = outputSeq ? outputSeq.length : 0;

  const [fwdResult, bwdResult] = await Promise.all([
    forward2DFull(machine, inputSeq, outputSeq),
    backward2D(machine, inputSeq, outputSeq),
  ]);

  const { logLikelihood, dp: fwd } = fwdResult;
  const { bp: bwd } = bwdResult;

  const size = (Li + 1) * (Lo + 1) * S;
  const posteriors = new Float32Array(size);

  for (let i = 0; i <= Li; i++) {
    for (let o = 0; o <= Lo; o++) {
      for (let s = 0; s < S; s++) {
        const idx = (i * (Lo + 1) + o) * S + s;
        const logPost = fwd[idx] + bwd[idx] - logLikelihood;
        posteriors[idx] = logPost === NEG_INF ? 0 : Math.exp(logPost);
      }
    }
  }

  return { logLikelihood, posteriors };
}
