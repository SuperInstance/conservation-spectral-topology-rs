# conservation-spectral-topology-rs

A Rust library applying **spectral graph theory** and **compiler optimization techniques** to graph-structured tile systems. It computes graph Laplacian spectra, verifies Cheeger inequalities, performs SVD-based low-rank approximation for game-theory score matrices, and provides ecosystem health diagnostics via algebraic connectivity.

## Why It Matters

Spectral graph theory connects the **eigenvalues of a graph's Laplacian** to its structural properties — connectivity bottlenecks, expansion quality, and community structure. This crate operationalizes that connection for:

- **Compiler optimization of tile graphs** — dead code elimination, constant folding, and JIT thresholding applied to game-tree search
- **Ecosystem health monitoring** — spectral gap, Cheeger constant, and conservation leakage as health metrics for multi-agent systems
- **Low-rank approximation** — SVD-based score matrix compression for efficient game-tree evaluation
- **Network science** — algebraic connectivity (Fiedler value λ₂) as a robustness metric

## How It Works

### Graph Laplacian

For a graph $G = (V, E)$ with weighted adjacency matrix $A$ and degree matrix $D$:

$$L = D - A$$

Key properties:
- $L$ is symmetric positive semidefinite (PSD)
- Eigenvalues $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$
- $\lambda_2$ (the *algebraic connectivity* / Fiedler value) measures how well-connected the graph is
- $\text{tr}(L) = \sum \lambda_i = \sum \deg(v_i)$

### Cheeger Constant

The isoperimetric number (Cheeger constant) $h_G$ measures the **bottleneck quality**:

$$h_G = \min_{S \subset V} \frac{|\partial S|}{\min(\text{vol}(S), \text{vol}(\bar{S}))}$$

where $\partial S$ is the edge boundary. The **Cheeger inequality** relates $h_G$ to $\lambda_2$:

$$\frac{h_G^2}{2\Delta} \leq \lambda_2 \leq 2 h_G$$

This crate computes $h_G$ via sweep-cut on the Fiedler vector (the eigenvector of $\lambda_2$).

### SVD Factorization

For a tile score matrix $M \in \mathbb{R}^{m \times n}$, the rank-$r$ truncated SVD is:

$$M_r = U_r \Sigma_r V_r^T$$

The reconstruction error (Frobenius norm) is:

$$\|M - M_r\|_F = \sqrt{\sum_{i=r+1}^{\min(m,n)} \sigma_i^2}$$

And the **variance explained**:

$$R^2 = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{i=1}^{\min(m,n)} \sigma_i^2}$$

### Compiler Passes

| Pass | Analog | Tile Graph Criterion |
|------|--------|---------------------|
| Dead code elimination | Remove unreachable code | `visit_count == 0` |
| Constant folding | Replace invariant computations | Score variance < ε |
| JIT compilation | Compile hot paths only | `visit_count ≥ threshold` |

### Hotelling Deflation

After finding an eigenpair $(\lambda, \mathbf{v})$, remove it from the matrix:

$$L' = L - \lambda \mathbf{v} \mathbf{v}^T$$

This preserves symmetry and is used for iterative eigenvalue computation.

### Big-O Complexity

| Operation | Time | Space |
|-----------|------|-------|
| `compute_spectral_budget(L)` | O(n³) | O(n²) |
| `cheeger_constant(L)` | O(n² log n) | O(n) |
| `svd_factorization(M, r)` | O(mn min(m,n)) | O(mn) |
| `dead_code_elimination(tiles)` | O(t) | O(t) |
| `constant_folding(tiles, cfg)` | O(t × h) | O(t) |
| `verify_cheeger_inequality(L)` | O(n³) | O(n²) |

## Quick Start

```rust
use conservation_spectral_topology::{cycle_laplacian, compute_spectral_budget, cheeger_constant};

let l = cycle_laplacian(6); // C₆ cycle graph
let budget = compute_spectral_budget(&l);

// Eigenvalues of C₆: 0, 1, 1, 3, 3, 4
assert!(budget.eigenvalues[0].abs() < 1e-10);
assert!((budget.eigenvalues[1] - 1.0).abs() < 1e-10);

let h = cheeger_constant(&l);
assert!((h - 1.0/3.0).abs() < 1e-10); // optimal cut: 3v3
```

### Ecosystem health check:

```rust
use conservation_spectral_topology::ecosystem::verify_ecosystem;
// See ecosystem module for full API
```

## API

### Core (lib.rs)

| Function | Description |
|----------|-------------|
| `compute_spectral_budget(&L) → SpectralBudget` | All eigenvalues, sorted ascending |
| `cheeger_constant(&L) → f64` | Sweep-cut isoperimetric number |
| `verify_cheeger_inequality(&L) → bool` | Check λ₂ ≥ h²/(2Δ) |
| `hotelling_deflation(&M, &v, λ) → Matrix` | Remove eigenpair |
| `cycle_laplacian(n)` / `path_laplacian(n)` / `complete_laplacian(n)` | Canonical graph generators |

### Compiler Module

| Function | Description |
|----------|-------------|
| `dead_code_elimination(&[Tile]) → DeadCodeResult` | Remove unvisited tiles |
| `constant_folding(&[Tile], &Config) → ConstantFoldResult` | Fold low-variance tiles |
| `svd_factorization(&Matrix, rank) → SvdFactorization` | Rank-r approximation |

### Ecosystem Module

| Function | Description |
|----------|-------------|
| `verify_ecosystem(&[Agent], &adjacency, &flow) → HealthReport` | Full spectral health check |

## Architecture Notes

The **γ + η = C** link: the algebraic operations (γ — SVD, eigenvalue decomposition, constant folding) transform graph and score data, while the conservation verification (η — Cheeger inequality check, trace conservation, variance explained) validates that these transforms preserve structural invariants. Together they conserve the mathematical invariant C — the trace of the Laplacian equals the sum of eigenvalues (verified in tests), the Cheeger inequality holds (λ₂ ≥ h²/(2Δ)), and SVD reconstruction error equals the sum of discarded singular values squared. The crate's test suite explicitly checks these conservation laws across multiple graph families (cycles, paths, complete graphs) at multiple scales.

## References

- Chung, F. R. K. (1997). *Spectral Graph Theory.* CBMS Regional Conference Series, 92. AMS.
- Cheeger, J. (1970). *A Lower Bound for the Smallest Eigenvalue of the Laplacian.* Problems in Analysis, Princeton.
- Fiedler, M. (1973). *Algebraic Connectivity of Graphs.* Czechoslovak Mathematical Journal, 23(2), 298–305.
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations,* 4th ed. JHU Press. (SVD algorithms.)
- Hoare, C. A. R. (1969). *An Axiomatic Basis for Computer Programming.* CACM. (Invariant-driven verification philosophy.)

## License

MIT
