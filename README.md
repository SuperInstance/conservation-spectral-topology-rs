# conservation-spectral-topology-rs

*Spectral topology conservation. When a graph's structure changes, how much of its spectral fingerprint is preserved? This crate measures the conservation law — the invariant that survives transformation.*

## Why This Exists

Every graph has a spectral fingerprint: the eigenvalues of its adjacency matrix, Laplacian, or normalized Laplacian. When you transform a graph (add/remove edges, contract nodes, apply a filter), the spectrum changes. But some spectral properties are *conserved* — they remain invariant across certain classes of transformations.

This crate implements the measurement of those conservation laws. It's the Rust port of the conservation-matrix crate, extended with spectral analysis tools that quantify how much topological information survives graph transformations.

## Architecture

```
Original Graph G ──→ Spectrum σ(G)
       ↓ transform           ↓ compare
Transformed Graph G' ──→ Spectrum σ(G')
                              ↓
                    Conservation Score: ||σ(G) - σ(G')|| / ||σ(G)||
                    Conserved Eigenvalues: count of λ within ε
                    Spectral Distance: Earth Mover's Distance
```

### Key Types

- **`SpectralProfile`** — Eigenvalues + eigenvectors of a graph's Laplacian.
- **`ConservationScore`** — How much spectral information survives a transformation (0.0 = total loss, 1.0 = perfect conservation).
- **`GraphTransform`** — AddEdge, RemoveEdge, ContractNodes, FilterGraph, etc.
- **`ConservationLaw`** — A named invariant (e.g., "total trace", "spectral radius ratio", "algebraic connectivity").

## Usage

```rust
use conservation_spectral_topology_rs::*;

let graph = Graph::from_edges(&[(0,1), (1,2), (2,3), (3,0)]);
let profile = SpectralProfile::compute(&graph);

// Apply a transformation
let transformed = graph.remove_edge(1, 2);
let profile2 = SpectralProfile::compute(&transformed);

// Measure conservation
let score = profile.conservation_score(&profile2);
println!("Spectral conservation: {:.3}", score);

// Find conserved eigenvalues
let conserved = profile.conserved_eigenvalues(&profile2, 0.01);
println!("{} eigenvalues conserved within ε=0.01", conserved.len());
```

## The Deeper Idea

Conservation laws are the deepest pattern in physics and mathematics. Energy is conserved. Charge is conserved. Information (in quantum mechanics) is conserved. This crate asks: what is conserved when a graph changes?

The answer connects directly to the SuperInstance ecosystem's core thesis: complex systems evolve, but certain structural invariants persist. The spectral fingerprint of a well-designed graph survives noisy transformations. The same principle underpins `ternary-renormalization` (what survives coarse-graining?), `agent-metamorphosis` (what survives developmental phase changes?), and `hodge-belief` (what beliefs survive evidence updates?).

## Related Crates

- `conservation-matrix-rs` — Conservation laws for matrix operations
- `spectral-graph-agent` — Spectral analysis for agent coordination graphs
- `ternary-renormalization` — Renormalization group with ternary states
- `hodge-belief-c` — Hodge decomposition of belief systems
