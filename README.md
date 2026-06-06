# conservation-spectral-topology-rs

Spectral graph theory meets conservation laws — eigenvalue budgets, Cheeger constants, and the topological invariants that survive when you coarse-grain a graph.

## Why This Exists

Graph Laplacians encode everything about a graph's structure in their eigenvalues. The smallest eigenvalue is always zero (constant vector in the kernel). The second smallest — the **algebraic connectivity** λ₂ — tells you how well-connected the graph is. The **Cheeger constant** tells you the minimum cut ratio. The **spectral gap** tells you how fast random walks mix. These aren't abstract quantities; they're conservation laws. The trace of the Laplacian equals the sum of eigenvalues (always 2|E|), and this invariant constrains what's possible.

This crate provides tools for computing spectral budgets (eigenvalue decompositions), verifying the Cheeger inequality (λ₂ ≥ h²/2Δ), performing Hotelling deflation (removing eigendirections from a matrix), and building standard graph Laplacians (cycle, path, complete).

Beyond the core spectral analysis, the crate includes five domain-specific modules:

- **conservation** — empirically discovered scaling laws from game-theory tile systems: N-player score variance scales as N^(−0.871), negative-space evolves 0.68× slower than positive-space.
- **cycles** — the *real* structural invariant. Spectral eigenvalues are trivially conserved (always sum to 2|E|). Cycle structure — self-calls, mutual call pairs, cycle density — actually discriminates between codebases.
- **ecosystem** — models agent networks as graphs and verifies resource conservation via Laplacian flow. If L·f ≈ 0, the flow is conserved.
- **holographic** — the holographic principle applied to tile systems: 5 tiles capture 99.8% of TTT variance. Within-genre transfer works (+0.6pp), cross-genre is noise.
- **compiler** — applies traditional compiler optimizations (DCE, constant folding, SVD factorization, JIT compilation) to game-theory tile graphs.

## Architecture

```
                    ┌─────────────────────────┐
  Graph Laplacian ──►│  compute_spectral_budget │──► SpectralBudget {eigenvalues}
                    └─────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    cheeger_constant   verify_cheeger    hotelling_deflation
    (sweep cut on      _inequality      (remove eigendirection)
     Fiedler vector)   (λ₂ ≥ h²/2Δ)

    Standard Laplacians: cycle_laplacian(n), path_laplacian(n), complete_laplacian(n)
    Utilities: laplacian_to_adjacency(L)

    ┌─────────────────────────────────────────────────┐
    │              Domain Modules                      │
    │                                                  │
    │  conservation: scaling laws, phase transitions   │
    │  cycles: structural invariants from call graphs  │
    │  ecosystem: agent network health verification    │
    │  holographic: tile compression, transfer learning│
    │  compiler: DCE, constant fold, SVD, JIT         │
    └─────────────────────────────────────────────────┘
```

## Usage

### Core Spectral Analysis

```rust
use conservation_spectral_topology_rs::*;
use nalgebra::DMatrix;

// Build standard graph Laplacians
let cycle = cycle_laplacian(6);     // C₆: 6 vertices in a ring
let path = path_laplacian(5);       // P₅: 5 vertices in a line
let complete = complete_laplacian(4); // K₄: fully connected

// Compute spectral budget (sorted eigenvalues)
let budget = compute_spectral_budget(&cycle);
println!("Eigenvalues: {:?}", budget.eigenvalues);
// C₆: [0, 1, 1, 3, 3, 4]

// Cheeger constant via Fiedler vector sweep cut
let h = cheeger_constant(&cycle);
// C₆: h = 1/3 (minimum cut splits 3v3 with 2 edges)

// Verify Cheeger inequality: λ₂ ≥ h²/(2Δ)
assert!(verify_cheeger_inequality(&cycle));

// Hotelling deflation: remove an eigendirection
let eigen = nalgebra::SymmetricEigen::new(cycle.clone());
let evec = eigen.eigenvectors.column(0).into_owned();
let eval = eigen.eigenvalues[0];
let deflated = hotelling_deflation(&cycle, &evec, eval);

// Convert Laplacian to adjacency
let adj = laplacian_to_adjacency(&cycle);
```

### Conservation Laws

```rust
use conservation_spectral_topology_rs::conservation::*;

// N-player scaling law: score std ~ N^(-0.871)
let std_1 = predicted_score_std(1);  // 1.0
let std_10 = predicted_score_std(10); // ~0.135
let var = predicted_score_variance(7);

// Negative space evolves slower
let neg_rate = negative_space_rate(1.0); // 0.68

// Verify against observed data
let counts = vec![2, 4, 8, 16, 32];
let observed = vec![0.55, 0.31, 0.17, 0.09, 0.05];
let fit = verify_scaling_law(&counts, &observed);
println!("Fitted α = {}, R² = {}", fit.fitted_alpha, fit.r_squared);

// Phase transition detection
let temps = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
let mag = vec![1.0, 0.95, 0.0, 0.05, 0.02, 0.01, 0.01]; // sharp drop
let result = check_phase_transition(&temps, &mag, 0.1);
assert!(result.has_transition);
```

### Cycle-Based Structural Analysis

```rust
use conservation_spectral_topology_rs::cycles::*;
use std::collections::HashMap;

let mut adj = HashMap::new();
adj.insert("main".into(), vec!["init".into(), "run".into()]);
adj.insert("init".into(), vec!["config".into()]);
adj.insert("run".into(), vec!["main".into(), "process".into()]); // mutual: main↔run

let metrics = compute_cycle_metrics(&adj);
println!("Self-calls: {}", metrics.self_calls);
println!("Mutual pairs: {}", metrics.mutual_call_pairs);
println!("Cycle density: {:.4}", metrics.cycle_density);

// Compare repos
let comparisons = compare_repos(&[("repo_a", metrics.clone()), ("repo_b", metrics)]);
// Sorted by discrimination power (highest relative diff first)
```

### Ecosystem Health

```rust
use conservation_spectral_topology_rs::ecosystem::*;
use nalgebra::{DMatrix, DVector};

let agents = vec![
    Agent { name: "scheduler".into(), agent_type: AgentType::Execution, budget: 100.0, current_usage: 72.0 },
    Agent { name: "memory".into(), agent_type: AgentType::Memory, budget: 80.0, current_usage: 65.0 },
];
let adj = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
let flow = DVector::from_vec(vec![1.0, 1.0]);

let report = verify_ecosystem(&agents, &adj, &flow);
println!("Health: {:.2}", report.health_score);
println!("Conservation leakage: {:.2e}", report.conservation_leakage);
println!("Algebraic connectivity: {:.4}", report.algebraic_connectivity);
```

### Holographic Compression

```rust
use conservation_spectral_topology_rs::holographic::*;

// Compress: how many tiles capture 99%+ of variance?
let variances = vec![0.40, 0.25, 0.15, 0.10, 0.098, 0.001, 0.0005];
let result = holographic_compress(&variances, 9, 0.998);
println!("{} tiles capture {:.1}% of variance", result.tile_count, result.coverage * 100.0);

// Transfer learning classification
let transfer = classify_transfer("ttt", "chess", 0.1, 0.3);
assert!(!transfer.significant); // cross-genre = noise
```

### Tile Compiler

```rust
use conservation_spectral_topology_rs::compiler::*;

let tiles = vec![
    Tile { id: 0, score: 1.0, visit_count: 0, score_history: vec![1.0] },    // dead
    Tile { id: 1, score: 2.0, visit_count: 50, score_history: vec![2.0; 50] }, // constant, hot
    Tile { id: 2, score: 3.0, visit_count: 100, score_history: vec![1.0, 5.0, 3.0, 7.0] }, // variable, hot
];

let config = CompilerConfig::default();
let result = compile_pipeline(&tiles, &config);
println!("DCE eliminated: {}", result.dce_eliminated);
println!("Constants folded: {}", result.constants_folded);
println!("JIT compiled: {}", result.compiled_count);
```

## API Reference

### Core (`lib.rs`)

| Function | Description |
|----------|-------------|
| `compute_spectral_budget(laplacian)` | Sorted eigenvalues of a symmetric Laplacian |
| `cheeger_constant(laplacian)` | Isoperimetric number via Fiedler sweep cut |
| `verify_cheeger_inequality(laplacian)` | Check λ₂ ≥ h²/(2Δ) |
| `hotelling_deflation(matrix, eigvec, eigval)` | Remove eigendirection: M − λvvᵀ |
| `laplacian_to_adjacency(laplacian)` | Extract adjacency from L = D − A |
| `cycle_laplacian(n)` | Laplacian of cycle graph Cₙ |
| `path_laplacian(n)` | Laplacian of path graph Pₙ |
| `complete_laplacian(n)` | Laplacian of complete graph Kₙ |

**Types:** `SpectralBudget { eigenvalues: DVector<f64> }`, `BudgetProfile { tolerance, max_iterations }`

### Module: `conservation`

Scaling laws, phase transitions, temporal evolution. Key exports: `predicted_score_std`, `verify_scaling_law`, `check_phase_transition`, `temporal_evolution`.

### Module: `cycles`

Structural analysis of call graphs. Key exports: `compute_cycle_metrics`, `compare_repos`. **Types:** `CycleMetrics`, `MetricComparison`.

### Module: `ecosystem`

Agent network health. Key exports: `verify_ecosystem`. **Types:** `Agent`, `AgentType`, `HealthReport`.

### Module: `holographic`

Holographic compression and transfer learning. Key exports: `holographic_compress`, `verify_ttt_holographic_bound`, `classify_transfer`, `transfer_summary`. **Types:** `HolographicCompression`, `TransferResult`, `TransferSummary`.

### Module: `compiler`

Tile graph optimization. Key exports: `dead_code_elimination`, `constant_folding`, `svd_factorization`, `jit_compile`, `compile_pipeline`. **Types:** `Tile`, `CompilerConfig`, `DeadCodeResult`, `ConstantFoldResult`, `SvdFactorization`, `JitResult`, `CompilePipelineResult`.

## The Deeper Idea

The central finding of this crate is that **spectral eigenvalues are trivially conserved**. The trace of any graph Laplacian is 2|E| (twice the number of edges). By linearity, the sum of eigenvalues is also 2|E|. This means spectral eigenvalues carry exactly one degree of freedom per graph — the edge count. They cannot discriminate between structurally different graphs with the same number of edges.

The real structural invariants are **cycle-based**: self-calls, mutual call pairs, cycle density, and hub structure. These have high coefficient of variation across codebases (CV = 3.32 for mutual_call_pairs) while spectral eigenvalues have CV ≈ 0. The conservation law is there, but it's topological, not spectral.

The holographic principle — 5 tiles capturing 99.8% of TTT variance — suggests that the effective dimensionality of game-theory systems is much smaller than the nominal representation. This is the graph analog of the holographic principle in physics: the "surface area" (few key tiles) encodes the "volume" (full game tree).

The compiler module applies classical optimization techniques to this compressed representation: dead code elimination removes unvisited tiles, constant folding collapses low-variance tiles, and SVD factorization finds the low-rank structure. The result is a compiled representation that preserves the holographic information while discarding the rest.

## Related Crates

- **`ternary-renormalization`** — coarse-graining of ternary fields, the real-space analog of spectral analysis
- **`ternary-percolate`** — percolation on ternary grids, whose connectivity is analyzed by spectral methods
- **`ternary-morphogenesis`** — reaction-diffusion patterns whose structure can be analyzed spectrally
