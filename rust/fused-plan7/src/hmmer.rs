//! HMMER3 profile parser.
//!
//! Faithful port of `js/webgpu/internal/hmmer-parse.mjs` (itself a port of
//! `python/machineboss/hmmer.py`). Parses an HMMER3 text profile into per-node
//! match/insert emissions and transitions, plus the SwissProt background null
//! model. Also computes match-state occupancy for local-mode entry weights.

/// SwissProt background amino acid frequencies (same table as the JS).
const SWISSPROT_BG: &[(char, f64)] = &[
    ('A', 0.0825),
    ('C', 0.0138),
    ('D', 0.0546),
    ('E', 0.0673),
    ('F', 0.0386),
    ('G', 0.0708),
    ('H', 0.0227),
    ('I', 0.0592),
    ('K', 0.0581),
    ('L', 0.0965),
    ('M', 0.0241),
    ('N', 0.0405),
    ('P', 0.0473),
    ('Q', 0.0393),
    ('R', 0.0553),
    ('S', 0.0663),
    ('T', 0.0535),
    ('V', 0.0686),
    ('W', 0.0109),
    ('Y', 0.0292),
];

fn swissprot_bg(sym: &str) -> Option<f64> {
    if sym.len() != 1 {
        return None;
    }
    let c = sym.chars().next().unwrap();
    SWISSPROT_BG.iter().find(|(k, _)| *k == c).map(|(_, v)| *v)
}

/// One Plan7 core node.
#[derive(Clone, Debug)]
pub struct HmmerNode {
    pub match_emit: Vec<f64>,
    pub ins_emit: Vec<f64>,
    pub m_to_m: f64,
    pub m_to_i: f64,
    pub m_to_d: f64,
    pub i_to_m: f64,
    pub i_to_i: f64,
    pub d_to_m: f64,
    pub d_to_d: f64,
}

/// A parsed HMMER3 model.
#[derive(Clone, Debug)]
pub struct HmmerModel {
    /// Emission alphabet (e.g. the 20 amino acids), in file order.
    pub alph: Vec<String>,
    pub nodes: Vec<HmmerNode>,
    pub ins0_emit: Vec<f64>,
    /// Background (null) emission probabilities, parallel to `alph`.
    pub null_emit: Vec<f64>,
    pub b_to_m1: f64,
    pub b_to_i0: f64,
    pub b_to_d1: f64,
    pub i0_to_m1: f64,
    pub i0_to_i0: f64,
}

/// Convert an HMMER log-probability token to a probability.
/// `"*"` → 0; otherwise `exp(-x)`. (Matches `strToProb`.)
#[inline]
fn str_to_prob(s: &str) -> f64 {
    if s == "*" {
        0.0
    } else {
        (-s.parse::<f64>().unwrap_or(f64::INFINITY)).exp()
    }
}

fn ws_split(line: &str) -> Vec<&str> {
    line.split_whitespace().collect()
}

/// Parse an HMMER3-format text string.
///
/// Mirrors `parseHmmer`: locate the `HMM` alphabet line, skip the transition
/// header / COMPO / node-0 insert lines, read node-0 inserts + begin
/// transitions, then read `(match, insert, transition)` triples per node until
/// `//`.
pub fn parse_hmmer(text: &str) -> Result<HmmerModel, String> {
    let lines: Vec<&str> = text.split('\n').collect();
    let mut model = HmmerModel {
        alph: Vec::new(),
        nodes: Vec::new(),
        ins0_emit: Vec::new(),
        null_emit: Vec::new(),
        b_to_m1: 0.0,
        b_to_i0: 0.0,
        b_to_d1: 0.0,
        i0_to_m1: 0.0,
        i0_to_i0: 0.0,
    };

    let mut idx = 0usize;
    while idx < lines.len() {
        let line = lines[idx];
        // Match a line beginning with "HMM" followed by whitespace.
        let is_hmm_line = {
            let t = line;
            (t.starts_with("HMM ") || t.starts_with("HMM\t"))
                && t.len() >= 3
        };
        if is_hmm_line {
            let tokens = ws_split(line);
            if tokens.len() <= 1 {
                return Err("HMMER parse error: no alphabet found on the HMM line. \
                            Is this a valid HMMER3 profile file?"
                    .to_string());
            }
            model.alph = tokens[1..].iter().map(|s| s.to_string()).collect();
            let n_alph = model.alph.len();

            // Skip transition header line, COMPO line, node 0 insert emission line.
            idx += 3;

            // Node 0 insert emissions.
            let ins0 = ws_split(lines[idx]);
            if ins0.len() != n_alph {
                return Err(format!(
                    "HMMER parse error at node 0 insert emissions: expected {} values, got {}. \
                     The file may be truncated or corrupted.",
                    n_alph,
                    ins0.len()
                ));
            }
            model.ins0_emit = ins0.iter().map(|s| str_to_prob(s)).collect();
            idx += 1;

            // Begin transitions.
            let bt = ws_split(lines[idx]);
            model.b_to_m1 = str_to_prob(bt[0]);
            model.b_to_i0 = str_to_prob(bt[1]);
            model.b_to_d1 = str_to_prob(bt[2]);
            model.i0_to_m1 = str_to_prob(bt[3]);
            model.i0_to_i0 = str_to_prob(bt[4]);
            idx += 1;

            // Parse nodes.
            while idx < lines.len() {
                let l = lines[idx];
                if l.starts_with("//") {
                    break;
                }
                let match_fields = ws_split(l);
                if match_fields.len() != n_alph + 6 {
                    return Err(format!(
                        "HMMER parse error at match emission line: expected {} fields, got {}. \
                         The file may be truncated or corrupted.",
                        n_alph + 6,
                        match_fields.len()
                    ));
                }
                idx += 1;
                let ins_fields = ws_split(lines[idx]);
                idx += 1;
                let trans_fields = ws_split(lines[idx]);
                if trans_fields.len() != 7 {
                    return Err(format!(
                        "HMMER parse error at transition line: expected 7 fields, got {}. \
                         The file may be truncated or corrupted.",
                        trans_fields.len()
                    ));
                }
                idx += 1;

                model.nodes.push(HmmerNode {
                    match_emit: match_fields[1..n_alph + 1]
                        .iter()
                        .map(|s| str_to_prob(s))
                        .collect(),
                    ins_emit: ins_fields.iter().map(|s| str_to_prob(s)).collect(),
                    m_to_m: str_to_prob(trans_fields[0]),
                    m_to_i: str_to_prob(trans_fields[1]),
                    m_to_d: str_to_prob(trans_fields[2]),
                    i_to_m: str_to_prob(trans_fields[3]),
                    i_to_i: str_to_prob(trans_fields[4]),
                    d_to_m: str_to_prob(trans_fields[5]),
                    d_to_d: str_to_prob(trans_fields[6]),
                });
            }
            break;
        }
        idx += 1;
    }

    // Null model (SwissProt background; uniform fallback for unknown symbols).
    let n_alph = model.alph.len();
    model.null_emit = model
        .alph
        .iter()
        .map(|sym| swissprot_bg(sym).unwrap_or(1.0 / n_alph as f64))
        .collect();

    Ok(model)
}

/// Match-state occupancy for local-mode entry weights.
/// Port of `calcMatchOccupancy` / `HmmerModel.calc_match_occupancy()`.
pub fn calc_match_occupancy(model: &HmmerModel) -> Vec<f64> {
    let k = model.nodes.len();
    let mut mocc = vec![0.0_f64; k];
    if k > 1 {
        mocc[1] = model.nodes[0].m_to_i + model.nodes[0].m_to_m;
    }
    for kk in 2..k {
        mocc[kk] = mocc[kk - 1] * (model.nodes[kk].m_to_m + model.nodes[kk].m_to_i)
            + (1.0 - mocc[kk - 1]) * model.nodes[kk].d_to_m;
    }
    mocc
}
