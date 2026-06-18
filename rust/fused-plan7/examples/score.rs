//! Score one output sequence against a fused Plan7 + transducer model.
//!
//! Usage:
//!   cargo run --release --example score -- \
//!       <profile.hmm> <transducer.json> <params.json|-> <SEQ> [multihit] [L]
//!
//! `params.json` may be `-` for no params (empty map). Prints two lines:
//!   FORWARD <value>
//!   VITERBI <value>
//!
//! Used by the cross-implementation check against the JS CPU reference.

use std::collections::HashMap;
use std::process::exit;

use fused_plan7::{
    build_fused_plan7, fused_plan7_forward, fused_plan7_viterbi, parse_hmmer, prepare_machine,
    tokenize, FusedOpts, Semiring,
};

fn die(msg: String) -> ! {
    eprintln!("error: {msg}");
    exit(1);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        die(format!(
            "usage: {} <profile.hmm> <transducer.json> <params.json|-> <SEQ> [multihit] [L]",
            args[0]
        ));
    }
    let hmm_path = &args[1];
    let td_path = &args[2];
    let params_path = &args[3];
    let seq = &args[4];
    let multihit = args.get(5).map(|s| s == "true" || s == "1").unwrap_or(false);
    let l: f64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(400.0);

    let hmm_text = std::fs::read_to_string(hmm_path).unwrap_or_else(|e| die(e.to_string()));
    let model = parse_hmmer(&hmm_text).unwrap_or_else(|e| die(e));

    let td_text = std::fs::read_to_string(td_path).unwrap_or_else(|e| die(e.to_string()));
    let machine_json: serde_json::Value =
        serde_json::from_str(&td_text).unwrap_or_else(|e| die(e.to_string()));

    let params: HashMap<String, f64> = if params_path == "-" {
        HashMap::new()
    } else {
        let pt = std::fs::read_to_string(params_path).unwrap_or_else(|e| die(e.to_string()));
        serde_json::from_str(&pt).unwrap_or_else(|e| die(e.to_string()))
    };

    let transducer = prepare_machine(&machine_json, &params).unwrap_or_else(|e| die(e));
    let fm = build_fused_plan7(&model, &transducer, FusedOpts { multihit, l });

    let tokens = if seq == "-" || seq.is_empty() {
        Vec::new()
    } else {
        tokenize(seq, &transducer.output_alphabet).unwrap_or_else(|e| die(e))
    };

    let fwd = fused_plan7_forward(&fm, &tokens, Semiring::LogSumExp);
    let vit = fused_plan7_viterbi(&fm, &tokens);
    println!("FORWARD {:.15}", fwd);
    println!("VITERBI {:.15}", vit);
}
