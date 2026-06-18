//! Machine preparation: JSON transducer + params -> SPARSE log-transition lists.
//!
//! Faithful port of `js/webgpu/internal/machine-prep.mjs`, EXCEPT the transition
//! tensor is stored sparsely instead of as a dense `[n_in * n_out * S * S]`
//! array. The JS reference iterates the dense `S*S` block for every emission;
//! for a real transducer (e.g. `prot2dna` with ~132 states) that is
//! `O(L * K * S^2)`. We instead store, per `(in_tok, out_tok)`, the list of
//! `(src, dst, log_weight)` actually present, giving `O(L * K * nnz)`.
//!
//! Skipping `NEG_INF` (absent) entries is *exactly* behaviour-preserving: the
//! semiring reduce (logsumexp / max) treats `NEG_INF` as its identity, so a
//! sparse fold over present entries equals the dense reduce over all entries.
//!
//! Token convention (identical to JS): token 0 is the null (epsilon/gap) token;
//! real tokens are 1-based indices into the sorted alphabet.

use serde_json::Value;
use std::collections::BTreeMap;
use std::collections::HashMap;

use crate::logmath::{logaddexp, NEG_INF};

/// One sparse transition: `value[dst] = plus(value[dst], v[src] + log_weight)`.
#[derive(Clone, Copy, Debug)]
pub struct SparseEdge {
    pub src: u32,
    pub dst: u32,
    pub log_weight: f64,
}

/// A prepared machine with sparse log-transition lists.
///
/// Mirrors `PreparedMachine` but stores transitions sparsely. The dense layout
/// it replaces was `logTrans[in*nOut*S*S + out*S*S + src*S + dst]`.
#[derive(Clone, Debug)]
pub struct PreparedMachine {
    pub n_states: usize,
    pub n_input_tokens: usize,
    pub n_output_tokens: usize,
    pub input_alphabet: Vec<String>,
    pub output_alphabet: Vec<String>,
    /// Sparse edges keyed by `(in_tok, out_tok)`. Within each list, edges are
    /// sorted by `dst` ascending then `src` ascending. The DP consumes them
    /// grouped by `dst`: for each `dst` it folds `v[src] + log_weight` over the
    /// contiguous run of edges with that `dst`, in `src`-ascending order. That is
    /// exactly the per-column reduce the dense JS scan performs
    /// (`reduce_src(v[src] + trans[src*S+dst])`), so the logsumexp accumulation is
    /// bit-identical. (Absent `src` entries are NEG_INF = reduce identity, so
    /// skipping them changes nothing.)
    pub edges: HashMap<(usize, usize), Vec<SparseEdge>>,
    /// The silent block `(in=0, out=0)` as a dense `S*S` matrix in row-major
    /// `[src*S + dst]` order (NEG_INF where absent). The JS `td_silent` is used
    /// densely (fixed-point iteration touches every entry), so we keep it dense
    /// too for an exact match.
    pub silent_dense: Vec<f64>,
}

impl PreparedMachine {
    /// Borrow the sparse edge list for `(in_tok, out_tok)`, or an empty slice.
    #[inline]
    pub fn edges_for(&self, in_tok: usize, out_tok: usize) -> &[SparseEdge] {
        match self.edges.get(&(in_tok, out_tok)) {
            Some(v) => v.as_slice(),
            None => &[],
        }
    }
}

/// Evaluate a weight expression to f64 given params + defs.
///
/// Faithful port of `evaluateWeight` (which mirrors `weight.py:evaluate`).
/// Supports number / bool / string (param or def lookup) / `{*,+,-,/,pow,log,exp,not}`.
pub fn evaluate_weight(
    w: &Value,
    params: &HashMap<String, f64>,
    defs: &HashMap<String, Value>,
) -> Result<f64, String> {
    match w {
        Value::Null => Ok(0.0),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| "weight: non-finite number".to_string()),
        Value::String(s) => {
            if let Some(v) = params.get(s) {
                Ok(*v)
            } else if let Some(d) = defs.get(s) {
                evaluate_weight(d, params, defs)
            } else {
                Err(format!(
                    "Unknown parameter \"{}\". Pass it in the params map.",
                    s
                ))
            }
        }
        Value::Object(map) => {
            // Binary operators take a 2-element array.
            let bin = |key: &str| -> Option<(&Value, &Value)> {
                map.get(key).and_then(|v| v.as_array()).and_then(|a| {
                    if a.len() == 2 {
                        Some((&a[0], &a[1]))
                    } else {
                        None
                    }
                })
            };
            if let Some((a, b)) = bin("*") {
                return Ok(evaluate_weight(a, params, defs)? * evaluate_weight(b, params, defs)?);
            }
            if let Some((a, b)) = bin("+") {
                return Ok(evaluate_weight(a, params, defs)? + evaluate_weight(b, params, defs)?);
            }
            if let Some((a, b)) = bin("-") {
                return Ok(evaluate_weight(a, params, defs)? - evaluate_weight(b, params, defs)?);
            }
            if let Some((a, b)) = bin("/") {
                return Ok(evaluate_weight(a, params, defs)? / evaluate_weight(b, params, defs)?);
            }
            if let Some((base, exp)) = bin("pow") {
                return Ok(evaluate_weight(base, params, defs)?
                    .powf(evaluate_weight(exp, params, defs)?));
            }
            if let Some(v) = map.get("log") {
                return Ok(evaluate_weight(v, params, defs)?.ln());
            }
            if let Some(v) = map.get("exp") {
                return Ok(evaluate_weight(v, params, defs)?.exp());
            }
            if let Some(v) = map.get("not") {
                return Ok(1.0 - evaluate_weight(v, params, defs)?);
            }
            Err(format!(
                "Unsupported weight operator \"{}\". Supported: *, +, -, /, pow, log, exp, not",
                map.keys().cloned().collect::<Vec<_>>().join(", ")
            ))
        }
        _ => Err("Unsupported weight expression type".to_string()),
    }
}

/// Canonicalize a state id/`to` reference to a lookup key, matching the JS
/// `Array.isArray(name) ? JSON.stringify(name) : name`.
fn ref_key(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(_) | Value::Object(_) => serde_json::to_string(v).unwrap_or_default(),
        Value::Number(n) => n.to_string(),
        _ => serde_json::to_string(v).unwrap_or_default(),
    }
}

/// Build a sorted token alphabet (index 0 = null/epsilon), mirroring
/// `buildAlphabet`: collect every non-empty `direction` token over all
/// transitions, sort lexicographically, prepend "".
fn build_alphabet(states: &[Value], direction: &str) -> Vec<String> {
    let mut set = std::collections::BTreeSet::new();
    for st in states {
        if let Some(trans) = st.get("trans").and_then(|t| t.as_array()) {
            for t in trans {
                if let Some(tok) = t.get(direction).and_then(|x| x.as_str()) {
                    if !tok.is_empty() {
                        set.insert(tok.to_string());
                    }
                }
            }
        }
    }
    let mut out = vec![String::new()];
    out.extend(set);
    out
}

/// Prepare a machine JSON + params into sparse log-transition lists.
///
/// Faithful port of `prepareMachine` but sparse. Duplicate transitions for the
/// same `(in, out, src, dst)` are combined with `logaddexp`, exactly as the JS.
pub fn prepare_machine(
    machine_json: &Value,
    params: &HashMap<String, f64>,
) -> Result<PreparedMachine, String> {
    let states = machine_json
        .get("state")
        .and_then(|s| s.as_array())
        .ok_or_else(|| "machine JSON has no \"state\" array".to_string())?;

    let defs: HashMap<String, Value> = match machine_json.get("defs").and_then(|d| d.as_object()) {
        Some(obj) => obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        None => HashMap::new(),
    };

    let s = states.len();
    let input_alphabet = build_alphabet(states, "in");
    let output_alphabet = build_alphabet(states, "out");
    let n_in = input_alphabet.len();
    let n_out = output_alphabet.len();

    // token -> index maps
    let in_tok_idx: HashMap<&str, usize> = input_alphabet
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i))
        .collect();
    let out_tok_idx: HashMap<&str, usize> = output_alphabet
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i))
        .collect();

    // Resolve state name references to indices (id can be string/array/number;
    // numeric index always maps to itself).
    let mut name_to_idx: HashMap<String, usize> = HashMap::new();
    for (i, st) in states.iter().enumerate() {
        if let Some(id) = st.get("id") {
            name_to_idx.insert(ref_key(id), i);
        }
        name_to_idx.insert(i.to_string(), i);
    }

    let resolve_dest = |dest: &Value| -> Result<usize, String> {
        if let Some(n) = dest.as_u64() {
            return Ok(n as usize);
        }
        let key = ref_key(dest);
        name_to_idx
            .get(&key)
            .copied()
            .ok_or_else(|| format!("Transition references unknown state \"{}\".", key))
    };

    // Accumulate into a (in,out,src,dst) -> log_weight map so duplicate
    // transitions are logaddexp-combined (matching the JS dense fill).
    // BTreeMap keeps deterministic iteration; we re-sort per (in,out) list below.
    let mut acc: BTreeMap<(usize, usize, u32, u32), f64> = BTreeMap::new();

    for (src, st) in states.iter().enumerate() {
        if let Some(trans) = st.get("trans").and_then(|t| t.as_array()) {
            for t in trans {
                let to = t
                    .get("to")
                    .ok_or_else(|| "transition missing \"to\"".to_string())?;
                let dst = resolve_dest(to)?;
                let in_idx = match t.get("in").and_then(|x| x.as_str()) {
                    Some(tok) if !tok.is_empty() => *in_tok_idx
                        .get(tok)
                        .ok_or_else(|| format!("unknown input token \"{}\"", tok))?,
                    _ => 0,
                };
                let out_idx = match t.get("out").and_then(|x| x.as_str()) {
                    Some(tok) if !tok.is_empty() => *out_tok_idx
                        .get(tok)
                        .ok_or_else(|| format!("unknown output token \"{}\"", tok))?,
                    _ => 0,
                };
                let weight_expr = t.get("weight").cloned().unwrap_or(Value::from(1));
                let log_weight = evaluate_weight(&weight_expr, params, &defs)?.ln();
                let key = (in_idx, out_idx, src as u32, dst as u32);
                acc.entry(key)
                    .and_modify(|w| *w = logaddexp(*w, log_weight))
                    .or_insert(log_weight);
            }
        }
    }

    // Materialize sparse edge lists per (in,out), then sort each list by
    // (dst, src) so the DP can fold per-dst columns in src-ascending order
    // (bit-identical to the dense per-column reduce; see the `edges` doc).
    let mut edges: HashMap<(usize, usize), Vec<SparseEdge>> = HashMap::new();
    let mut silent_dense = vec![NEG_INF; s * s];
    for ((in_idx, out_idx, src, dst), lw) in acc.into_iter() {
        edges
            .entry((in_idx, out_idx))
            .or_default()
            .push(SparseEdge {
                src,
                dst,
                log_weight: lw,
            });
        if in_idx == 0 && out_idx == 0 {
            silent_dense[src as usize * s + dst as usize] = lw;
        }
    }
    for list in edges.values_mut() {
        list.sort_by(|a, b| a.dst.cmp(&b.dst).then(a.src.cmp(&b.src)));
    }

    Ok(PreparedMachine {
        n_states: s,
        n_input_tokens: n_in,
        n_output_tokens: n_out,
        input_alphabet,
        output_alphabet,
        edges,
        silent_dense,
    })
}

/// 1-based token index for `symbol` in `alphabet` (index 0 = null).
/// Mirrors `tokenIndex`.
pub fn token_index(alphabet: &[String], symbol: &str) -> Result<u32, String> {
    alphabet
        .iter()
        .position(|a| a == symbol)
        .map(|p| p as u32)
        .ok_or_else(|| {
            format!(
                "Unknown symbol \"{}\". Valid symbols: {}",
                symbol,
                alphabet[1..].join(", ")
            )
        })
}

/// Convert a string of single-character symbols to 1-based token indices.
/// Mirrors `tokenize` for the string-input case.
pub fn tokenize(seq: &str, alphabet: &[String]) -> Result<Vec<u32>, String> {
    seq.chars()
        .map(|c| token_index(alphabet, &c.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logmath::Semiring;
    use serde_json::json;

    #[test]
    fn evaluate_weight_operators() {
        let p: HashMap<String, f64> = [("x".to_string(), 2.0), ("y".to_string(), 3.0)]
            .into_iter()
            .collect();
        let d: HashMap<String, Value> = HashMap::new();
        assert_eq!(evaluate_weight(&json!({"*": ["x", "y"]}), &p, &d).unwrap(), 6.0);
        assert_eq!(evaluate_weight(&json!({"+": ["x", "y"]}), &p, &d).unwrap(), 5.0);
        assert_eq!(evaluate_weight(&json!({"-": ["y", "x"]}), &p, &d).unwrap(), 1.0);
        assert_eq!(evaluate_weight(&json!({"/": ["y", "x"]}), &p, &d).unwrap(), 1.5);
        assert_eq!(evaluate_weight(&json!({"pow": ["x", "y"]}), &p, &d).unwrap(), 8.0);
        assert_eq!(evaluate_weight(&json!({"not": 0.25}), &p, &d).unwrap(), 0.75);
        assert!((evaluate_weight(&json!({"exp": {"log": "x"}}), &p, &d).unwrap() - 2.0).abs() < 1e-12);
        assert_eq!(evaluate_weight(&json!(1), &p, &d).unwrap(), 1.0);
        assert_eq!(evaluate_weight(&Value::Null, &p, &d).unwrap(), 0.0);
        assert!(evaluate_weight(&json!("missing"), &p, &d).is_err());
    }

    #[test]
    fn defs_lookup_resolves() {
        let machine = json!({
            "defs": { "half": 0.5 },
            "state": [{ "id": "S", "trans": [{ "to": "S", "in": "a", "out": "a", "weight": "half" }] }]
        });
        let prep = prepare_machine(&machine, &HashMap::new()).unwrap();
        // log(0.5) on the single (in=a,out=a) edge.
        let e = prep.edges_for(1, 1);
        assert_eq!(e.len(), 1);
        assert!((e[0].log_weight - 0.5_f64.ln()).abs() < 1e-12);
    }

    /// The core correctness claim of the sparse representation: folding the
    /// sparse per-`dst` edge lists equals reducing the equivalent DENSE `S*S`
    /// matrix (NEG_INF where absent). Verified for both semirings on a small
    /// multi-state transducer with a random-ish source vector.
    #[test]
    fn sparse_fold_equals_dense_reduce() {
        // 3-state machine, a single (in=a,out=b) block plus some silent edges.
        let machine = json!({
            "state": [
                { "id": 0, "trans": [
                    { "to": 1, "in": "a", "out": "b", "weight": 0.5 },
                    { "to": 2, "in": "a", "out": "b", "weight": 0.25 },
                    { "to": 0 } // silent self-loop weight 1
                ]},
                { "id": 1, "trans": [
                    { "to": 2, "in": "a", "out": "b", "weight": 0.1 },
                    { "to": 1, "in": "a", "out": "b", "weight": 0.7 }
                ]},
                { "id": 2, "trans": [
                    { "to": 0, "in": "a", "out": "b", "weight": 0.9 }
                ]}
            ]
        });
        let prep = prepare_machine(&machine, &HashMap::new()).unwrap();
        let s = prep.n_states;
        assert_eq!(s, 3);
        // tokens: in alphabet ["","a"], out ["","b"] -> a=1, b=1.
        let (in_tok, out_tok) = (1usize, 1usize);

        // Build the dense block for (in_tok,out_tok).
        let mut dense = vec![NEG_INF; s * s];
        for e in prep.edges_for(in_tok, out_tok) {
            dense[e.src as usize * s + e.dst as usize] = e.log_weight;
        }

        let v = [-0.3_f64, -1.2, -2.5];
        for sem in [Semiring::LogSumExp, Semiring::MaxPlus] {
            // Dense reduce per column (the JS tdMatvec).
            let dense_out: Vec<f64> = (0..s)
                .map(|dst| sem.reduce((0..s).map(|src| v[src] + dense[src * s + dst])))
                .collect();
            // Sparse fold per dst-group (what td_emit/td_delete do).
            let edges = prep.edges_for(in_tok, out_tok);
            let mut sparse_out = vec![NEG_INF; s];
            let mut i = 0;
            while i < edges.len() {
                let dst = edges[i].dst as usize;
                let mut j = i;
                while j < edges.len() && edges[j].dst as usize == dst {
                    j += 1;
                }
                sparse_out[dst] =
                    sem.reduce(edges[i..j].iter().map(|e| v[e.src as usize] + e.log_weight));
                i = j;
            }
            for st in 0..s {
                let (a, b) = (dense_out[st], sparse_out[st]);
                assert!(
                    (a == NEG_INF && b == NEG_INF) || (a - b).abs() < 1e-15,
                    "{:?} col {}: dense {} vs sparse {}",
                    sem,
                    st,
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn duplicate_transitions_logaddexp_combine() {
        // Two identical (in=a,out=b,src=0,dst=0) edges should combine via logaddexp.
        let machine = json!({
            "state": [{ "id": 0, "trans": [
                { "to": 0, "in": "a", "out": "b", "weight": 0.3 },
                { "to": 0, "in": "a", "out": "b", "weight": 0.4 }
            ]}]
        });
        let prep = prepare_machine(&machine, &HashMap::new()).unwrap();
        let e = prep.edges_for(1, 1);
        assert_eq!(e.len(), 1, "duplicate edges merge into one");
        let expected = crate::logmath::logaddexp(0.3_f64.ln(), 0.4_f64.ln());
        assert!((e[0].log_weight - expected).abs() < 1e-15);
    }
}
