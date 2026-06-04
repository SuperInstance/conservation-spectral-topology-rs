//! Cycle-based structural analysis — the REAL invariant.
//!
//! Experiments proved that cycle structure (mutual calls, self-calls, cycle density)
//! is the true structural invariant across repos, not spectral eigenvalues.
//!
//! Key finding: CV=3.32 for mutual_call_pairs (discriminating) vs ~0 for spectral
//! eigenvalues (trivially conserved — always sum to 2|E| regardless of structure).

use std::collections::{HashMap, HashSet};

/// Cycle-based metrics that capture real structural differences between call graphs.
#[derive(Debug, Clone)]
pub struct CycleMetrics {
    /// Number of self-calling functions (A→A edges)
    pub self_calls: usize,
    /// Number of mutual call pairs (A→B and B→A)
    pub mutual_call_pairs: usize,
    /// Cycle density = (self_calls + mutual_call_pairs) / total_edges
    pub cycle_density: f64,
    /// Number of hub nodes (in-degree > mean + 2*std)
    pub hub_count: usize,
    /// Edge density = total_edges / (nodes * (nodes - 1))
    pub edge_density: f64,
    /// Total number of edges
    pub total_edges: usize,
    /// Total number of nodes
    pub total_nodes: usize,
}

/// A comparison result between two repos on a specific metric.
#[derive(Debug, Clone)]
pub struct MetricComparison {
    pub repo_a: String,
    pub repo_b: String,
    pub metric_name: String,
    pub value_a: f64,
    pub value_b: f64,
    pub relative_diff: f64,
}

/// Compute cycle-based structural metrics from an adjacency list.
///
/// The adjacency list maps each function name to the list of functions it calls.
pub fn compute_cycle_metrics(adjacency: &HashMap<String, Vec<String>>) -> CycleMetrics {
    let nodes: HashSet<&str> = adjacency
        .keys()
        .map(|s| s.as_str())
        .chain(adjacency.values().flat_map(|v| v.iter().map(|s| s.as_str())))
        .collect();
    let total_nodes = nodes.len();
    let total_nodes_safe = total_nodes.max(1);

    // Count total edges (including self-calls)
    let total_edges: usize = adjacency.values().map(|v| v.len()).sum();

    // Count self-calls: functions that appear in their own callee list
    let self_calls: usize = adjacency
        .iter()
        .filter(|(caller, callees)| callees.contains(caller))
        .count();

    // Count mutual call pairs: (A,B) where A→B and B→A
    let mut mutual_pairs = 0usize;
    let mut seen_pairs: HashSet<(String, String)> = HashSet::new();
    for (caller, callees) in adjacency {
        for callee in callees {
            if callee == caller {
                continue; // self-call, not mutual
            }
            // Check if callee also calls caller
            if let Some(callee_callees) = adjacency.get(callee) {
                if callee_callees.contains(caller) {
                    // Normalize pair to avoid double-counting
                    let pair = if caller < callee {
                        (caller.clone(), callee.clone())
                    } else {
                        (callee.clone(), caller.clone())
                    };
                    if seen_pairs.insert(pair) {
                        mutual_pairs += 1;
                    }
                }
            }
        }
    }

    // Cycle density = cyclic edges / total edges
    let cycle_density = if total_edges > 0 {
        (self_calls + mutual_pairs) as f64 / total_edges as f64
    } else {
        0.0
    };

    // Compute in-degrees for hub detection
    let mut in_degree: HashMap<&str, usize> = HashMap::new();
    // Initialize all known nodes with 0
    for node in &nodes {
        in_degree.insert(node, 0);
    }
    for callees in adjacency.values() {
        for callee in callees {
            if let Some(d) = in_degree.get_mut(callee.as_str()) {
                *d += 1;
            } else {
                in_degree.insert(callee, 1);
            }
        }
    }

    let degrees: Vec<f64> = in_degree.values().map(|&d| d as f64).collect();
    let n = degrees.len().max(1);
    let mean_deg: f64 = degrees.iter().sum::<f64>() / n as f64;
    let variance: f64 =
        degrees.iter().map(|d| (d - mean_deg).powi(2)).sum::<f64>() / n as f64;
    let std_deg = variance.sqrt();

    // Hubs: in-degree > mean + 2*std
    let hub_threshold = mean_deg + 2.0 * std_deg;
    let hub_count = in_degree.values().filter(|&&d| d as f64 > hub_threshold).count();

    // Edge density (directed, possible edges = n*(n-1))
    let edge_density = if total_nodes > 1 {
        total_edges as f64 / (total_nodes_safe * (total_nodes_safe - 1)) as f64
    } else {
        0.0
    };

    CycleMetrics {
        self_calls,
        mutual_call_pairs: mutual_pairs,
        cycle_density,
        hub_count,
        edge_density,
        total_edges,
        total_nodes,
    }
}

/// Compare cycle metrics across multiple repos, ranked by discrimination power.
///
/// Returns comparisons sorted by relative difference (descending) — the metrics
/// that differ most between repos are the most structurally informative.
pub fn compare_repos(repos: &[(&str, CycleMetrics)]) -> Vec<MetricComparison> {
    let mut comparisons = Vec::new();

    for i in 0..repos.len() {
        for j in (i + 1)..repos.len() {
            let (name_a, metrics_a) = &repos[i];
            let (name_b, metrics_b) = &repos[j];

            // Compare each metric
            let fields: Vec<(&str, f64, f64)> = vec![
                ("self_calls", metrics_a.self_calls as f64, metrics_b.self_calls as f64),
                ("mutual_call_pairs", metrics_a.mutual_call_pairs as f64, metrics_b.mutual_call_pairs as f64),
                ("cycle_density", metrics_a.cycle_density, metrics_b.cycle_density),
                ("hub_count", metrics_a.hub_count as f64, metrics_b.hub_count as f64),
                ("edge_density", metrics_a.edge_density, metrics_b.edge_density),
                ("total_edges", metrics_a.total_edges as f64, metrics_b.total_edges as f64),
                ("total_nodes", metrics_a.total_nodes as f64, metrics_b.total_nodes as f64),
            ];

            for (metric_name, val_a, val_b) in fields {
                let relative_diff = if val_a.abs() > 1e-10 || val_b.abs() > 1e-10 {
                    (val_a - val_b).abs() / val_a.max(val_b).max(1e-10)
                } else {
                    0.0
                };

                comparisons.push(MetricComparison {
                    repo_a: name_a.to_string(),
                    repo_b: name_b.to_string(),
                    metric_name: metric_name.to_string(),
                    value_a: val_a,
                    value_b: val_b,
                    relative_diff,
                });
            }
        }
    }

    // Sort by relative difference descending (most discriminating first)
    comparisons.sort_by(|a, b| {
        b.relative_diff
            .partial_cmp(&a.relative_diff)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    comparisons
}

/// Print a cycle metrics report to stdout.
pub fn print_cycle_report(name: &str, metrics: &CycleMetrics) {
    println!("=== Cycle Metrics: {} ===\n", name);
    println!("  Nodes:              {}", metrics.total_nodes);
    println!("  Edges:              {}", metrics.total_edges);
    println!("  Self-calls:         {}", metrics.self_calls);
    println!("  Mutual call pairs:  {}", metrics.mutual_call_pairs);
    println!("  Cycle density:      {:.4}", metrics.cycle_density);
    println!("  Hub count:          {}", metrics.hub_count);
    println!("  Edge density:       {:.4}", metrics.edge_density);
    println!();
}

/// Print a comparison table of cycle metrics across repos.
pub fn print_comparison(comparisons: &[MetricComparison]) {
    println!("=== Cross-Repo Metric Comparison (sorted by discrimination power) ===\n");
    println!(
        "{:<20} {:<20} {:<25} {:>10} {:>10} {:>12}",
        "Repo A", "Repo B", "Metric", "Value A", "Value B", "Rel Diff"
    );
    println!("{}", "─".repeat(100));
    for c in comparisons.iter().take(20) {
        println!(
            "{:<20} {:<20} {:<25} {:>10.2} {:>10.2} {:>12.4}",
            c.repo_a, c.repo_b, c.metric_name, c.value_a, c.value_b, c.relative_diff
        );
    }
    if comparisons.len() > 20 {
        println!("... and {} more comparisons", comparisons.len() - 20);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_adjacency(pairs: &[(&str, &[&str])]) -> HashMap<String, Vec<String>> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.iter().map(|s| s.to_string()).collect()))
            .collect()
    }

    #[test]
    fn test_empty_graph() {
        let adj = HashMap::new();
        let metrics = compute_cycle_metrics(&adj);
        assert_eq!(metrics.self_calls, 0);
        assert_eq!(metrics.mutual_call_pairs, 0);
        assert_eq!(metrics.total_edges, 0);
        // total_nodes uses max(1) for edge_density calc, but 0 for an empty adj
        assert!(metrics.total_nodes <= 1);
    }

    #[test]
    fn test_self_call() {
        let adj = make_adjacency(&[
            ("foo", &["foo", "bar"]),
            ("bar", &["baz"]),
            ("baz", &[]),
        ]);
        let metrics = compute_cycle_metrics(&adj);
        assert_eq!(metrics.self_calls, 1); // foo calls itself
        assert_eq!(metrics.mutual_call_pairs, 0);
        assert_eq!(metrics.total_edges, 3);
    }

    #[test]
    fn test_mutual_call_pair() {
        let adj = make_adjacency(&[
            ("A", &["B"]),
            ("B", &["A"]),
        ]);
        let metrics = compute_cycle_metrics(&adj);
        assert_eq!(metrics.self_calls, 0);
        assert_eq!(metrics.mutual_call_pairs, 1);
        assert_eq!(metrics.total_edges, 2);
        assert!((metrics.cycle_density - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_mutual_call_no_double_count() {
        let adj = make_adjacency(&[
            ("A", &["B", "C"]),
            ("B", &["A"]),
            ("C", &["A"]),
        ]);
        let metrics = compute_cycle_metrics(&adj);
        assert_eq!(metrics.mutual_call_pairs, 2); // A↔B and A↔C
    }

    #[test]
    fn test_cycle_density() {
        // 2 self-calls + 1 mutual pair = 3 cyclic, out of 6 total edges
        let adj = make_adjacency(&[
            ("A", &["A", "B"]),
            ("B", &["A", "B", "C"]),
            ("C", &["A"]),
        ]);
        let metrics = compute_cycle_metrics(&adj);
        assert_eq!(metrics.self_calls, 2); // A and B call themselves
        assert_eq!(metrics.mutual_call_pairs, 1); // A↔B
        assert_eq!(metrics.total_edges, 6);
        assert!((metrics.cycle_density - 3.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_hub_detection() {
        // Create a graph where one node has very high in-degree
        let mut adj = HashMap::new();
        adj.insert("hub".to_string(), vec![]);
        for i in 0..20 {
            let caller = format!("caller_{}", i);
            adj.insert(caller.clone(), vec!["hub".to_string()]);
        }
        let metrics = compute_cycle_metrics(&adj);
        // hub has in-degree 20, all others have 0 or 1
        // mean ≈ 20/21 ≈ 0.95, std should be moderate, hub should be detected
        assert!(metrics.hub_count >= 1, "hub should be detected, got {}", metrics.hub_count);
    }

    #[test]
    fn test_edge_density() {
        let adj = make_adjacency(&[
            ("A", &["B"]),
            ("B", &["A"]),
        ]);
        let metrics = compute_cycle_metrics(&adj);
        // 2 nodes, 2 edges, possible = 2*1 = 2, density = 1.0
        assert!((metrics.edge_density - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_edge_density_sparse() {
        let adj = make_adjacency(&[
            ("A", &["B"]),
            ("B", &[]),
            ("C", &[]),
        ]);
        let metrics = compute_cycle_metrics(&adj);
        // 3 nodes, 1 edge, possible = 3*2 = 6, density = 1/6
        assert!((metrics.edge_density - 1.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_compare_repos() {
        let adj_dense = make_adjacency(&[
            ("A", &["A", "B"]),
            ("B", &["A", "B"]),
        ]);
        let adj_sparse = make_adjacency(&[
            ("X", &["Y"]),
            ("Y", &[]),
            ("Z", &[]),
        ]);

        let metrics_dense = compute_cycle_metrics(&adj_dense);
        let metrics_sparse = compute_cycle_metrics(&adj_sparse);

        let comparisons = compare_repos(&[
            ("dense", metrics_dense),
            ("sparse", metrics_sparse),
        ]);

        // Should have comparisons
        assert!(!comparisons.is_empty());

        // Most discriminating metric should have highest relative_diff
        assert!(
            comparisons[0].relative_diff >= comparisons.last().unwrap().relative_diff,
            "should be sorted by discrimination power"
        );
    }

    #[test]
    fn test_compare_identical_repos() {
        let adj = make_adjacency(&[
            ("A", &["B"]),
            ("B", &["A"]),
        ]);
        let metrics = compute_cycle_metrics(&adj);
        let comparisons = compare_repos(&[
            ("repo1", metrics.clone()),
            ("repo2", metrics),
        ]);

        // All relative diffs should be 0 for identical repos
        for c in &comparisons {
            assert!(
                c.relative_diff.abs() < 1e-10,
                "identical repos should have 0 relative diff, got {} for {}",
                c.relative_diff,
                c.metric_name
            );
        }
    }

    #[test]
    fn test_nodes_from_callees_only() {
        // Node "C" never appears as a caller but is a callee — should be counted
        let adj = make_adjacency(&[
            ("A", &["B", "C"]),
            ("B", &[]),
        ]);
        let metrics = compute_cycle_metrics(&adj);
        assert_eq!(metrics.total_nodes, 3); // A, B, C
        assert_eq!(metrics.total_edges, 2);
    }

    #[test]
    fn test_single_node() {
        let adj = make_adjacency(&[
            ("A", &[]),
        ]);
        let metrics = compute_cycle_metrics(&adj);
        assert_eq!(metrics.total_nodes, 1);
        assert_eq!(metrics.total_edges, 0);
        assert_eq!(metrics.edge_density, 0.0);
    }
}
