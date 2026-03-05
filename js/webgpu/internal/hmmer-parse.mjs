/**
 * Pure JavaScript HMMER3 file parser.
 *
 * Port of python/machineboss/hmmer.py.
 * Parses HMMER3 text format into a structured object for use
 * by the fused Plan7 kernel.
 */

// SwissProt background amino acid frequencies
const SWISSPROT_BG = {
  A: 0.0825, C: 0.0138, D: 0.0546, E: 0.0673,
  F: 0.0386, G: 0.0708, H: 0.0227, I: 0.0592,
  K: 0.0581, L: 0.0965, M: 0.0241, N: 0.0405,
  P: 0.0473, Q: 0.0393, R: 0.0553, S: 0.0663,
  T: 0.0535, V: 0.0686, W: 0.0109, Y: 0.0292,
};

/**
 * Convert HMMER log-probability string to probability.
 * @param {string} s
 * @returns {number}
 */
function strToProb(s) {
  return s === '*' ? 0 : Math.exp(-parseFloat(s));
}

/**
 * Parse an HMMER3 format text string.
 *
 * @param {string} text - HMMER3 file contents
 * @returns {{ alph: string[], nodes: Array<{match_emit: number[], ins_emit: number[],
 *   m_to_m: number, m_to_i: number, m_to_d: number,
 *   i_to_m: number, i_to_i: number, d_to_m: number, d_to_d: number}>,
 *   ins0_emit: number[], null_emit: number[],
 *   b_to_m1: number, b_to_i0: number, b_to_d1: number,
 *   i0_to_m1: number, i0_to_i0: number }}
 */
export function parseHmmer(text) {
  const lines = text.split('\n');
  let idx = 0;
  const model = {
    alph: [],
    nodes: [],
    ins0_emit: [],
    null_emit: [],
    b_to_m1: 0, b_to_i0: 0, b_to_d1: 0,
    i0_to_m1: 0, i0_to_i0: 0,
  };

  // Find the HMM line with alphabet
  while (idx < lines.length) {
    const line = lines[idx];
    if (/^HMM\s/.test(line)) {
      const tokens = line.trim().split(/\s+/);
      if (tokens.length <= 1) throw new Error('HMMER parse error: no alphabet found on the HMM line. Is this a valid HMMER3 profile file?');
      model.alph = tokens.slice(1);

      // Skip transition header line, COMPO line, node 0 insert emission line
      idx += 3;

      // Node 0 insert emissions
      const ins0Tokens = lines[idx].trim().split(/\s+/);
      if (ins0Tokens.length !== model.alph.length) {
        throw new Error(`HMMER parse error at node 0 insert emissions: expected ${model.alph.length} values, got ${ins0Tokens.length}. The file may be truncated or corrupted.`);
      }
      model.ins0_emit = ins0Tokens.map(strToProb);
      idx++;

      // Begin transitions
      const bt = lines[idx].trim().split(/\s+/);
      model.b_to_m1 = strToProb(bt[0]);
      model.b_to_i0 = strToProb(bt[1]);
      model.b_to_d1 = strToProb(bt[2]);
      model.i0_to_m1 = strToProb(bt[3]);
      model.i0_to_i0 = strToProb(bt[4]);
      idx++;

      // Parse nodes
      while (idx < lines.length) {
        const line = lines[idx];
        if (/^\/\//.test(line)) break;

        const matchFields = line.trim().split(/\s+/);
        if (matchFields.length !== model.alph.length + 6) {
          throw new Error(`HMMER parse error at match emission line: expected ${model.alph.length + 6} fields, got ${matchFields.length}. The file may be truncated or corrupted.`);
        }

        idx++;
        const insFields = lines[idx].trim().split(/\s+/);
        idx++;
        const transFields = lines[idx].trim().split(/\s+/);
        if (transFields.length !== 7) {
          throw new Error(`HMMER parse error at transition line: expected 7 fields, got ${transFields.length}. The file may be truncated or corrupted.`);
        }
        idx++;

        model.nodes.push({
          match_emit: matchFields.slice(1, model.alph.length + 1).map(strToProb),
          ins_emit: insFields.map(strToProb),
          m_to_m: strToProb(transFields[0]),
          m_to_i: strToProb(transFields[1]),
          m_to_d: strToProb(transFields[2]),
          i_to_m: strToProb(transFields[3]),
          i_to_i: strToProb(transFields[4]),
          d_to_m: strToProb(transFields[5]),
          d_to_d: strToProb(transFields[6]),
        });
      }
      break;
    }
    idx++;
  }

  // Load null model (SwissProt background)
  model.null_emit = model.alph.map(sym =>
    sym in SWISSPROT_BG ? SWISSPROT_BG[sym] : 1.0 / model.alph.length
  );

  return model;
}

/**
 * Calculate match state occupancy for local-mode entry weights.
 * Port of HmmerModel.calc_match_occupancy().
 *
 * @param {{ nodes: Array }} model - parsed HMMER model
 * @returns {number[]} occupancy values, length K
 */
export function calcMatchOccupancy(model) {
  const K = model.nodes.length;
  const mocc = new Array(K).fill(0);
  if (K > 1) {
    mocc[1] = model.nodes[0].m_to_i + model.nodes[0].m_to_m;
  }
  for (let k = 2; k < K; k++) {
    mocc[k] = mocc[k - 1] * (model.nodes[k].m_to_m + model.nodes[k].m_to_i) +
              (1.0 - mocc[k - 1]) * model.nodes[k].d_to_m;
  }
  return mocc;
}
