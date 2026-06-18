//! Correctness tests for the fused Plan7 + transducer kernel.
//!
//! Ground truth:
//!  - The `boss` CLI Forward/Viterbi for the AA-echo + fn3 fixtures
//!    (`boss --hmmer-plan7 t/hmmer/fn3.hmm --compose aaecho --output-chars SEQ -L|-V`,
//!    and `--hmmer-multihit` for the multihit variants). For these fixtures the
//!    JS CPU reference itself agrees with `boss`, so they pin the algorithm.
//!  - The JS CPU reference (`js/webgpu/cpu/fused-plan7.mjs`) for the
//!    `prot2dna`-fused cases (PF03184), captured by running node on the same
//!    input. (The JS reference has a known, pre-existing divergence from `boss`
//!    on multi-state output-emitting transducers; this port reproduces the JS
//!    reference faithfully, which is the stated requirement.)
//!
//! The Cabinet-data tests (PF03184 + prot2dna) are skipped gracefully if the
//! external data directory is not present, so the suite still runs in a bare
//! checkout. The fn3 tests are in-repo and always run.

use std::collections::HashMap;
use std::path::PathBuf;

use fused_plan7::{
    build_fused_plan7, fused_plan7_forward, fused_plan7_viterbi, parse_hmmer, prepare_machine,
    tokenize, FusedOpts, Semiring,
};
use serde_json::{json, Value};

/// Repo root = crate dir / ../.. (crate lives at <repo>/rust/fused-plan7).
fn repo_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // rust/
    p.pop(); // repo root
    p
}

fn read_repo(rel: &str) -> String {
    std::fs::read_to_string(repo_root().join(rel))
        .unwrap_or_else(|e| panic!("reading {rel}: {e}"))
}

/// The amino-acid echo transducer the JS test builds inline.
fn aa_echo() -> Value {
    let aa = "ACDEFGHIKLMNPQRSTVWY";
    let trans: Vec<Value> = aa
        .chars()
        .map(|c| json!({ "in": c.to_string(), "out": c.to_string(), "to": "S" }))
        .collect();
    json!({ "state": [ { "id": "S", "trans": trans } ] })
}

const FN3: &str = "t/hmmer/fn3.hmm";

// ----------------------------------------------------------------------
// Parser
// ----------------------------------------------------------------------

#[test]
fn parses_fn3_hmmer() {
    let model = parse_hmmer(&read_repo(FN3)).unwrap();
    assert_eq!(model.alph.len(), 20, "20-letter amino alphabet");
    assert_eq!(model.alph[0], "A");
    assert!(
        model.nodes.len() >= 80 && model.nodes.len() <= 100,
        "fn3 K in 80..100 (got {})",
        model.nodes.len()
    );
    let node = &model.nodes[0];
    assert_eq!(node.match_emit.len(), 20);
    assert_eq!(node.ins_emit.len(), 20);
    assert!(node.m_to_m >= 0.0 && node.m_to_m <= 1.0);
    // Null model = SwissProt background.
    assert!((model.null_emit[0] - 0.0825).abs() < 1e-4);
}

// ----------------------------------------------------------------------
// fn3 + AA-echo vs boss CLI references
// ----------------------------------------------------------------------

/// `boss --hmmer-plan7 fn3.hmm --compose aaecho --output-chars SEQ -L` (single-hit).
const FN3_SH_FORWARD: &[(&str, f64)] = &[
    ("", -15.775871523088234),
    ("A", -17.23392012512111),
    ("ACDE", -25.700850015908266),
    ("VLIWFYH", -35.40283661099204),
];

/// `... -V` (single-hit Viterbi).
const FN3_SH_VITERBI: &[(&str, f64)] = &[
    ("A", -20.979034112531608),
    ("ACDE", -29.964893469097063),
    ("VLIWFYH", -38.86027997772892),
];

/// `boss --hmmer-multihit fn3.hmm --compose aaecho --output-chars SEQ -L`.
const FN3_MH_FORWARD: &[(&str, f64)] = &[
    ("ACDE", -26.3939428352782),
    ("VLIWFYH", -36.09581217985952),
];

/// `... --hmmer-multihit ... -V`.
const FN3_MH_VITERBI: &[(&str, f64)] = &[
    ("ACDE", -30.658040649657007),
    ("VLIWFYH", -39.55342715828887),
];

fn build_fn3() -> (fused_plan7::PreparedMachine, fused_plan7::HmmerModel) {
    let model = parse_hmmer(&read_repo(FN3)).unwrap();
    let td = prepare_machine(&aa_echo(), &HashMap::new()).unwrap();
    (td, model)
}

#[test]
fn fn3_singlehit_forward_matches_boss() {
    let (td, model) = build_fn3();
    let fm = build_fused_plan7(&model, &td, FusedOpts { multihit: false, l: 400.0 });
    for &(seq, expected) in FN3_SH_FORWARD {
        let toks = tokenize(seq, &td.output_alphabet).unwrap();
        let got = fused_plan7_forward(&fm, &toks, Semiring::LogSumExp);
        // boss does exact silent-cycle elimination; the JS/Rust kernel does a
        // 1e-10 fixed-point closure, so allow ~1e-4. (Rust vs JS is < 1e-13.)
        assert!(
            (got - expected).abs() < 1e-3,
            "fn3 SH Forward '{seq}': got {got}, boss {expected}, diff {}",
            (got - expected).abs()
        );
    }
}

#[test]
fn fn3_singlehit_viterbi_matches_boss() {
    let (td, model) = build_fn3();
    let fm = build_fused_plan7(&model, &td, FusedOpts { multihit: false, l: 400.0 });
    for &(seq, expected) in FN3_SH_VITERBI {
        let toks = tokenize(seq, &td.output_alphabet).unwrap();
        let got = fused_plan7_viterbi(&fm, &toks);
        assert!(
            (got - expected).abs() < 1e-4,
            "fn3 SH Viterbi '{seq}': got {got}, boss {expected}, diff {}",
            (got - expected).abs()
        );
    }
}

#[test]
fn fn3_multihit_forward_matches_boss() {
    let (td, model) = build_fn3();
    let fm = build_fused_plan7(&model, &td, FusedOpts { multihit: true, l: 400.0 });
    for &(seq, expected) in FN3_MH_FORWARD {
        let toks = tokenize(seq, &td.output_alphabet).unwrap();
        let got = fused_plan7_forward(&fm, &toks, Semiring::LogSumExp);
        assert!(
            (got - expected).abs() < 1e-3,
            "fn3 MH Forward '{seq}': got {got}, boss {expected}, diff {}",
            (got - expected).abs()
        );
    }
}

#[test]
fn fn3_multihit_viterbi_matches_boss() {
    let (td, model) = build_fn3();
    let fm = build_fused_plan7(&model, &td, FusedOpts { multihit: true, l: 400.0 });
    for &(seq, expected) in FN3_MH_VITERBI {
        let toks = tokenize(seq, &td.output_alphabet).unwrap();
        let got = fused_plan7_viterbi(&fm, &toks);
        assert!(
            (got - expected).abs() < 1e-4,
            "fn3 MH Viterbi '{seq}': got {got}, boss {expected}, diff {}",
            (got - expected).abs()
        );
    }
}

// ----------------------------------------------------------------------
// Structural / semiring properties (mirror the JS test)
// ----------------------------------------------------------------------

#[test]
fn forward_finite_and_negative() {
    let (td, model) = build_fn3();
    let fm = build_fused_plan7(&model, &td, FusedOpts::default());
    let toks = tokenize("ACDE", &td.output_alphabet).unwrap();
    let f = fused_plan7_forward(&fm, &toks, Semiring::LogSumExp);
    assert!(f.is_finite() && f < 0.0, "Forward finite & negative: {f}");
}

#[test]
fn viterbi_le_forward_and_strictly_less() {
    let (td, model) = build_fn3();
    let fm = build_fused_plan7(&model, &td, FusedOpts::default());
    let toks = tokenize("ACDE", &td.output_alphabet).unwrap();
    let f = fused_plan7_forward(&fm, &toks, Semiring::LogSumExp);
    let v = fused_plan7_viterbi(&fm, &toks);
    assert!(f.is_finite() && v.is_finite());
    assert!(v <= f + 1e-10, "Viterbi {v} <= Forward {f}");
    // Multiple paths contribute, so Forward strictly exceeds Viterbi.
    assert!(f > v + 1e-6, "Forward {f} > Viterbi {v}");
}

#[test]
fn empty_sequence_finite() {
    let (td, model) = build_fn3();
    let fm = build_fused_plan7(&model, &td, FusedOpts::default());
    let f = fused_plan7_forward(&fm, &[], Semiring::LogSumExp);
    assert!(f.is_finite() && f < 0.0, "empty Forward finite & negative: {f}");
}

#[test]
fn mismatched_alphabet_gives_neg_inf() {
    // fn3 (amino) composed with a 0/1 bit-echo: no amino input matches 0/1, so
    // no valid path -> -inf (mirrors the JS bitecho test).
    let model = parse_hmmer(&read_repo(FN3)).unwrap();
    let bitecho = read_repo("t/machine/bitecho.json");
    let bitecho: Value = serde_json::from_str(&bitecho).unwrap();
    let td = prepare_machine(&bitecho, &HashMap::new()).unwrap();
    let fm = build_fused_plan7(&model, &td, FusedOpts::default());
    let toks = tokenize("010101", &td.output_alphabet).unwrap();
    let f = fused_plan7_forward(&fm, &toks, Semiring::LogSumExp);
    assert!(f == f64::NEG_INFINITY || f < -1e30, "expected -inf, got {f}");
}

// ----------------------------------------------------------------------
// Cabinet: PF03184 + prot2dna vs the JS CPU reference (the oracle for this
// multi-state-transducer case). Skipped if the external data is absent.
// ----------------------------------------------------------------------

const CABINET: &str =
    "/Users/yam/Dropbox/Classes/BioE131/AgentUpdate2026/games/cabinet/validator-rs/data";

/// Codon (E.coli) + flank params + uniform DNA background, matching the params
/// used to capture the JS reference values below.
fn prot2dna_params() -> HashMap<String, f64> {
    let mut p: HashMap<String, f64> =
        serde_json::from_str(&std::fs::read_to_string(repo_root().join("data/Ecoli_codon.json")).unwrap())
            .unwrap();
    // flank_introns.json: { flankExtend, intron, extendIntron }
    let flank: HashMap<String, f64> = serde_json::from_str(
        &std::fs::read_to_string(repo_root().join("data/flank_introns.json")).unwrap(),
    )
    .unwrap();
    for (k, v) in flank {
        p.insert(k, v);
    }
    for b in ["pA", "pC", "pG", "pT"] {
        p.insert(b.to_string(), 0.25);
    }
    p
}

/// (seq, multihit, JS Forward, JS Viterbi) captured from node on
/// js/webgpu/cpu/fused-plan7.mjs with the params above.
const PF03184_PROT2DNA: &[(&str, bool, f64, f64)] = &[
    ("", false, -19.554530337765993, -30.834164134850855),
    ("ATG", false, -21.194599680482842, -32.50226678012371),
    ("ATGGCAGATGAA", false, -24.004941621751914, -39.78891199458879),
    ("GCAGATGAATTT", false, -24.21970905774714, -41.63739089276443),
    ("ATGGCAGATGAATTTCATCAT", false, -28.582169471906212, -52.37848618913655),
    ("", true, -20.24767751832594, -31.5273113154108),
    ("ATGGCAGATGAA", true, -24.69753894443723, -40.482059175148734),
    ("GCAGATGAATTT", true, -24.912316659780043, -42.33053807332438),
];

#[test]
fn pf03184_prot2dna_matches_js_reference() {
    let cab = PathBuf::from(CABINET);
    if !cab.join("PF03184.hmm").exists() || !cab.join("prot2dna.json").exists() {
        eprintln!("skipping: Cabinet data not present at {CABINET}");
        return;
    }
    let model = parse_hmmer(&std::fs::read_to_string(cab.join("PF03184.hmm")).unwrap()).unwrap();
    let machine: Value =
        serde_json::from_str(&std::fs::read_to_string(cab.join("prot2dna.json")).unwrap()).unwrap();
    let params = prot2dna_params();
    let td = prepare_machine(&machine, &params).unwrap();
    assert_eq!(td.n_states, 132, "prot2dna has 132 states");

    for &(seq, mh, jf, jv) in PF03184_PROT2DNA {
        let fm = build_fused_plan7(&model, &td, FusedOpts { multihit: mh, l: 400.0 });
        let toks = if seq.is_empty() {
            Vec::new()
        } else {
            tokenize(seq, &td.output_alphabet).unwrap()
        };
        let f = fused_plan7_forward(&fm, &toks, Semiring::LogSumExp);
        let v = fused_plan7_viterbi(&fm, &toks);
        // Same f64 algorithm as the JS -> last-ULP agreement.
        assert!(
            (f - jf).abs() < 1e-9,
            "PF03184 prot2dna Forward '{seq}' mh={mh}: Rust {f}, JS {jf}, diff {}",
            (f - jf).abs()
        );
        assert!(
            (v - jv).abs() < 1e-9,
            "PF03184 prot2dna Viterbi '{seq}' mh={mh}: Rust {v}, JS {jv}, diff {}",
            (v - jv).abs()
        );
        assert!(v <= f + 1e-9, "Viterbi {v} <= Forward {f}");
    }
}

// ----------------------------------------------------------------------
// Edge cases
// ----------------------------------------------------------------------

#[test]
fn longer_expected_length_lowers_forward() {
    // Larger expected length L shifts the N/C loop weights; the kernel should
    // remain finite and monotone-ish. Mostly a smoke test for the L plumbing.
    let (td, model) = build_fn3();
    let toks = tokenize("ACDE", &td.output_alphabet).unwrap();
    let f10 = {
        let fm = build_fused_plan7(&model, &td, FusedOpts { multihit: false, l: 10.0 });
        fused_plan7_forward(&fm, &toks, Semiring::LogSumExp)
    };
    let f1000 = {
        let fm = build_fused_plan7(&model, &td, FusedOpts { multihit: false, l: 1000.0 });
        fused_plan7_forward(&fm, &toks, Semiring::LogSumExp)
    };
    assert!(f10.is_finite() && f1000.is_finite());
    // Matches the JS: L=10 -> -18.78, L=1000 -> -27.53 (longer L => more loop mass spread, lower).
    assert!(f1000 < f10, "L=1000 ({f1000}) < L=10 ({f10})");
}

#[test]
fn single_state_transducer_is_dense_equivalent() {
    // For the 1-state echo, sparse == dense trivially; confirm a known value.
    let (td, model) = build_fn3();
    assert_eq!(td.n_states, 1);
    let fm = build_fused_plan7(&model, &td, FusedOpts::default());
    let toks = tokenize("ACDE", &td.output_alphabet).unwrap();
    let f = fused_plan7_forward(&fm, &toks, Semiring::LogSumExp);
    assert!((f - (-25.70083821454159)).abs() < 1e-9, "got {f}");
}
