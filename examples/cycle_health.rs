//! CLI binary that reads call graphs and reports cycle metrics.
//!
//! Demonstrates the Cycle Conservation Law: cycle structure is the real
//! structural invariant, not spectral eigenvalues.
//!
//! Run with: cargo run --example cycle_health

use conservation_spectral_topology::cycles::{
    compute_cycle_metrics, compare_repos, print_cycle_report, print_comparison,
};
use std::collections::HashMap;

fn main() {
    // Simulated call graphs from different repos
    // (In production, these would be parsed from actual codebases)

    let mut react_adj: HashMap<String, Vec<String>> = HashMap::new();
    react_adj.insert("useState".into(), vec!["dispatch".into()]);
    react_adj.insert("useEffect".into(), vec!["cleanup".into(), "setState".into()]);
    react_adj.insert("dispatch".into(), vec!["reducer".into()]);
    react_adj.insert("reducer".into(), vec!["dispatch".into()]); // mutual: dispatch↔reducer
    react_adj.insert("cleanup".into(), vec![]);
    react_adj.insert("setState".into(), vec!["useState".into(), "render".into()]); // mutual: setState→useState
    react_adj.insert("render".into(), vec!["useState".into(), "useEffect".into()]);
    react_adj.insert("useCallback".into(), vec!["useCallback".into()]); // self-call (memo pattern)

    let mut express_adj: HashMap<String, Vec<String>> = HashMap::new();
    express_adj.insert("router".into(), vec!["middleware".into(), "handler".into()]);
    express_adj.insert("middleware".into(), vec!["next".into()]);
    express_adj.insert("next".into(), vec!["middleware".into()]); // mutual: middleware↔next
    express_adj.insert("handler".into(), vec!["response".into()]);
    express_adj.insert("response".into(), vec!["handler".into()]); // mutual: handler↔response
    express_adj.insert("errorHandler".into(), vec!["errorHandler".into(), "response".into()]); // self-call
    express_adj.insert("logger".into(), vec!["next".into()]);

    let mut cli_adj: HashMap<String, Vec<String>> = HashMap::new();
    cli_adj.insert("parse".into(), vec!["validate".into()]);
    cli_adj.insert("validate".into(), vec!["execute".into()]);
    cli_adj.insert("execute".into(), vec!["format".into()]);
    cli_adj.insert("format".into(), vec!["output".into()]);
    cli_adj.insert("output".into(), vec![]); // no cycles — pure pipeline

    let repos: Vec<(&str, HashMap<String, Vec<String>>)> = vec![
        ("react-app", react_adj),
        ("express-server", express_adj),
        ("cli-tool", cli_adj),
    ];

    // Compute metrics for each repo
    let metrics: Vec<(&str, _)> = repos
        .iter()
        .map(|(name, adj)| {
            let m = compute_cycle_metrics(adj);
            print_cycle_report(name, &m);
            (*name, m)
        })
        .collect();

    println!();

    // Cross-repo comparison
    let comparisons = compare_repos(&metrics);
    print_comparison(&comparisons);

    println!();
    println!("=== Key Insight ===");
    println!("Metrics with HIGH relative difference are the REAL structural invariants.");
    println!("Metrics with LOW relative difference (like spectral eigenvalues) are trivially");
    println!("conserved — they don't actually distinguish different architectures.");
    println!();
    println!("The Cycle Conservation Law: cycle structure (mutual calls, self-calls,");
    println!("cycle density) captures what actually differs between codebases.");
}
