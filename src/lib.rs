//! conservation-spectral-topology: Spectral graph theory tools for graph Laplacians.
//!
//! Uses nalgebra's built-in `SymmetricEigen` — no custom eigenvalue solvers.

use nalgebra::{DMatrix, DVector, SymmetricEigen};

pub mod ecosystem;

/// Holds the eigenvalues of a graph Laplacian, sorted ascending.
#[derive(Debug, Clone)]
pub struct SpectralBudget {
    pub eigenvalues: DVector<f64>,
}

/// Compute all eigenvalues of a symmetric Laplacian matrix.
///
/// The Laplacian must be symmetric (we sort eigenvalues ascending).
/// Eigenvalues are returned as a `SpectralBudget`.
pub fn compute_spectral_budget(laplacian: &DMatrix<f64>) -> SpectralBudget {
    let n = laplacian.nrows();
    debug_assert!(n == laplacian.ncols(), "Laplacian must be square");

    let eigen = SymmetricEigen::new(laplacian.clone());
    let mut evals = eigen.eigenvalues.clone();
    // Sort ascending — SymmetricEigen may not guarantee order
    evals.as_mut_slice().sort_by(|a, b| a.partial_cmp(b).unwrap());
    SpectralBudget { eigenvalues: evals }
}

/// Compute the Cheeger constant (isoperimetric number) via a sweep cut
/// on the Fiedler vector (eigenvector of the second smallest eigenvalue).
///
/// Returns the minimum edge expansion found.
pub fn cheeger_constant(laplacian: &DMatrix<f64>) -> f64 {
    let n = laplacian.nrows();
    if n < 2 {
        return 1.0; // degenerate
    }

    let eigen = SymmetricEigen::new(laplacian.clone());
    let evals = eigen.eigenvalues.clone();
    let evecs = eigen.eigenvectors.clone();

    // Find the index of the smallest positive eigenvalue (λ₂)
    // Sort eigenvalues and reorder eigenvectors correspondingly
    let mut idx_sorted: Vec<usize> = (0..n).collect();
    idx_sorted.sort_by(|&a, &b| evals[a].partial_cmp(&evals[b]).unwrap());

    // λ₂ is at idx_sorted[1], get its eigenvector index
    let fiedler_idx = idx_sorted[1];
    let fiedler = evecs.column(fiedler_idx);

    // Sweep cut: sort vertices by Fiedler vector component
    let mut verts: Vec<(usize, f64)> = (0..n).map(|i| (i, fiedler[i])).collect();
    verts.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let deg: Vec<f64> = (0..n).map(|i| laplacian[(i, i)]).collect();
    let total_volume: f64 = deg.iter().sum();

    let mut best_h = 1.0;
    for k in 1..n {
        // S = first k vertices in sorted order
        let mut cut = 0.0;
        let mut vol_s = 0.0;
        for i in 0..k {
            let vi = verts[i].0;
            vol_s += deg[vi];
            for j in 0..n {
                if laplacian[(vi, j)] < -1e-12 {
                    // edge vi-j
                    let j_in_s = (0..k).any(|t| verts[t].0 == j);
                    if !j_in_s {
                        cut += 1.0;
                    }
                }
            }
        }
        if vol_s > 1e-12 && vol_s < total_volume - 1e-12 {
            let h = cut / vol_s.min(total_volume - vol_s);
            if h < best_h {
                best_h = h;
            }
        }
    }

    if best_h > 1.0 {
        best_h = 1.0
    };
    best_h
}

/// Hotelling deflation: remove the contribution of an eigendirection from a matrix.
///
/// `matrix` — symmetric matrix (e.g., Laplacian)
/// `eigvec` — eigenvector (unit norm)
/// `eigval` — corresponding eigenvalue
///
/// Returns `matrix - eigval * (eigvec * eigvecᵀ)`
pub fn hotelling_deflation(
    matrix: &DMatrix<f64>,
    eigvec: &DVector<f64>,
    eigval: f64,
) -> DMatrix<f64> {
    let n = matrix.nrows();
    let mut result = matrix.clone();
    for i in 0..n {
        for j in 0..n {
            result[(i, j)] -= eigval * eigvec[i] * eigvec[j];
        }
    }
    result
}

/// Parameters for budget-verification routines.
#[derive(Debug, Clone)]
pub struct BudgetProfile {
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl Default for BudgetProfile {
    fn default() -> Self {
        BudgetProfile {
            tolerance: 1e-10,
            max_iterations: 100,
        }
    }
}

/// Verify the Cheeger inequality: λ₂ ≥ h² / (2·Δ)
/// where λ₂ is the second smallest eigenvalue, h is the Cheeger constant,
/// and Δ is the maximum degree.
///
/// Returns true if the inequality holds within a small numerical tolerance.
pub fn verify_cheeger_inequality(laplacian: &DMatrix<f64>) -> bool {
    let n = laplacian.nrows();
    if n < 2 {
        return true;
    }

    // Compute λ₂
    let eigen = SymmetricEigen::new(laplacian.clone());
    let mut evals: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lambda2 = evals[1];

    // Compute Cheeger constant
    let cheeger = cheeger_constant(laplacian);

    // Compute max degree Δ
    let max_deg: f64 = (0..n).map(|i| laplacian[(i, i)]).fold(0.0, f64::max);

    // Cheeger inequality: λ₂ ≥ h² / (2·Δ)
    // Allow 1e-10 slack for floating-point
    let rhs = (cheeger * cheeger) / (2.0 * max_deg.max(1.0));
    lambda2 >= rhs - 1e-10
}

/// Compute adjacency matrix from Laplacian (assumes standard Laplacian: L = D - A)
pub fn laplacian_to_adjacency(laplacian: &DMatrix<f64>) -> DMatrix<f64> {
    let n = laplacian.nrows();
    let mut adj = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if i != j && laplacian[(i, j)] < -1e-12 {
                adj[(i, j)] = -laplacian[(i, j)];
            }
        }
    }
    adj
}

/// Build the Laplacian of a cycle graph C_n (n vertices in a ring).
pub fn cycle_laplacian(n: usize) -> DMatrix<f64> {
    let mut l = DMatrix::zeros(n, n);
    for i in 0..n {
        l[(i, i)] = 2.0;
        l[(i, (i + 1) % n)] = -1.0;
        l[(i, (i + n - 1) % n)] = -1.0;
    }
    l
}

/// Build the Laplacian of a path graph P_n (n vertices in a line).
pub fn path_laplacian(n: usize) -> DMatrix<f64> {
    let mut l = DMatrix::zeros(n, n);
    for i in 0..n {
        if i > 0 {
            l[(i, i)] += 1.0;
            l[(i, i - 1)] = -1.0;
            l[(i - 1, i)] = -1.0;
        }
        if i + 1 < n {
            l[(i, i)] += 1.0;
        }
    }
    l
}

/// Build the Laplacian of a complete graph K_n.
pub fn complete_laplacian(n: usize) -> DMatrix<f64> {
    let mut l = DMatrix::zeros(n, n);
    for i in 0..n {
        l[(i, i)] = (n - 1) as f64;
        for j in 0..n {
            if i != j {
                l[(i, j)] = -1.0;
            }
        }
    }
    l
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    // ---------------------------------------------------------------
    // 1. Cycle graph C₃
    // ---------------------------------------------------------------
    #[test]
    fn test_c3_spectral_budget() {
        let l = cycle_laplacian(3);
        let budget = compute_spectral_budget(&l);
        // C₃ Laplacian eigenvalues: 0, 3, 3
        assert!(budget.eigenvalues[0].abs() < 1e-10);
        assert!((budget.eigenvalues[1] - 3.0).abs() < 1e-10);
        assert!((budget.eigenvalues[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_c3_cheeger() {
        let l = cycle_laplacian(3);
        let h = cheeger_constant(&l);
        // C₃: best cut separates 1 vertex from 2 → cut=2 edges, vol=min(2,4)=2 → h=1.0
        assert!((h - 1.0).abs() < 1e-10, "C₃ Cheeger should be 1, got {h}");
    }

    #[test]
    fn test_c3_cheeger_inequality() {
        let l = cycle_laplacian(3);
        // λ₂=1.5, h=1.0, Δ=2 → RHS = 1/(4)=0.25, λ₂ >= 0.25 ✓
        assert!(verify_cheeger_inequality(&l));
    }

    #[test]
    fn test_c3_trace_conservation() {
        let l = cycle_laplacian(3);
        let budget = compute_spectral_budget(&l);
        let trace_l: f64 = l.diagonal().sum();
        let trace_evals: f64 = budget.eigenvalues.sum();
        assert!((trace_l - trace_evals).abs() < 1e-10);
    }

    // ---------------------------------------------------------------
    // 2. Cycle graph C₆
    // ---------------------------------------------------------------
    #[test]
    fn test_c6_spectral_budget() {
        let l = cycle_laplacian(6);
        let budget = compute_spectral_budget(&l);
        // C₆ eigenvalues: 0, 1, 1, 3, 3, 4
        assert!(budget.eigenvalues[0].abs() < 1e-10);
        assert!((budget.eigenvalues[1] - 1.0).abs() < 1e-10);
        assert!((budget.eigenvalues[2] - 1.0).abs() < 1e-10);
        assert!((budget.eigenvalues[3] - 3.0).abs() < 1e-10);
        assert!((budget.eigenvalues[4] - 3.0).abs() < 1e-10);
        assert!((budget.eigenvalues[5] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_c6_cheeger() {
        let l = cycle_laplacian(6);
        let h = cheeger_constant(&l);
        // C₆: optimal cut splits 3 vs 3 with cut=2 edges → h = 2/min(6,6) = 1/3
        assert!((h - 1.0 / 3.0).abs() < 1e-10, "C₆ Cheeger should be 1/3, got {h}");
    }

    #[test]
    fn test_c6_cheeger_inequality() {
        let l = cycle_laplacian(6);
        // λ₂≈1.0, h=1/3, Δ=2 → RHS = (1/9)/4 = 1/36 ≈ 0.0278, λ₂ >= 0.0278 ✓
        assert!(verify_cheeger_inequality(&l));
    }

    #[test]
    fn test_c6_trace_conservation() {
        let l = cycle_laplacian(6);
        let budget = compute_spectral_budget(&l);
        let trace_l: f64 = l.diagonal().sum();
        let trace_evals: f64 = budget.eigenvalues.sum();
        assert!((trace_l - trace_evals).abs() < 1e-10);
    }

    // ---------------------------------------------------------------
    // 3. Cycle graph C₁₂
    // ---------------------------------------------------------------
    #[test]
    fn test_c12_spectral_budget() {
        let l = cycle_laplacian(12);
        let budget = compute_spectral_budget(&l);
        let trace_l: f64 = l.diagonal().sum();
        let trace_evals: f64 = budget.eigenvalues.sum();
        assert!((trace_l - trace_evals).abs() < 1e-10);
    }

    #[test]
    fn test_c12_first_eigenvalue_zero() {
        let l = cycle_laplacian(12);
        let budget = compute_spectral_budget(&l);
        assert!(budget.eigenvalues[0].abs() < 1e-10);
    }

    #[test]
    fn test_c12_cheeger_inequality() {
        let l = cycle_laplacian(12);
        assert!(verify_cheeger_inequality(&l));
    }

    // ---------------------------------------------------------------
    // 4. Path graphs
    // ---------------------------------------------------------------
    #[test]
    fn test_path_3_spectral_budget() {
        let l = path_laplacian(3);
        let budget = compute_spectral_budget(&l);
        assert!(budget.eigenvalues[0].abs() < 1e-10);
        // P₃: 0, 1, 3
        assert!((budget.eigenvalues[1] - 1.0).abs() < 1e-10);
        assert!((budget.eigenvalues[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_path_4_trace_conservation() {
        let l = path_laplacian(4);
        let budget = compute_spectral_budget(&l);
        let trace_l: f64 = l.diagonal().sum();
        let trace_evals: f64 = budget.eigenvalues.sum();
        assert!((trace_l - trace_evals).abs() < 1e-10);
    }

    #[test]
    fn test_path_4_cheeger_inequality() {
        let l = path_laplacian(4);
        assert!(verify_cheeger_inequality(&l));
    }

    // ---------------------------------------------------------------
    // 5. Complete graphs
    // ---------------------------------------------------------------
    #[test]
    fn test_k3_spectral_budget() {
        let l = complete_laplacian(3);
        let budget = compute_spectral_budget(&l);
        // K₃: 0, 3, 3
        assert!(budget.eigenvalues[0].abs() < 1e-10);
        assert!((budget.eigenvalues[1] - 3.0).abs() < 1e-10);
        assert!((budget.eigenvalues[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_k4_trace_conservation() {
        let l = complete_laplacian(4);
        let budget = compute_spectral_budget(&l);
        let trace_l: f64 = l.diagonal().sum();
        let trace_evals: f64 = budget.eigenvalues.sum();
        assert!((trace_l - trace_evals).abs() < 1e-10);
    }

    #[test]
    fn test_k4_cheeger() {
        let l = complete_laplacian(4);
        let h = cheeger_constant(&l);
        // K₄: optimal cut splits 2v2 → cut=4 edges, min vol = 6, h = 4/6 = 2/3
        assert!((h - 2.0 / 3.0).abs() < 1e-10, "K₄ Cheeger should be 2/3, got {h}");
    }

    #[test]
    fn test_k4_cheeger_inequality() {
        let l = complete_laplacian(4);
        assert!(verify_cheeger_inequality(&l));
    }

    // ---------------------------------------------------------------
    // 6. Hotelling deflation
    // ---------------------------------------------------------------
    #[test]
    fn test_hotelling_deflation_identity() {
        let n = 4;
        let mat = DMatrix::identity(n, n);
        // eigenvector of identity: any unit vector, eigenvalue = 1
        let e = DVector::from_element(n, 1.0 / (n as f64).sqrt());
        let deflated = hotelling_deflation(&mat, &e, 1.0);
        // mat - 1*e*e^T should have eigenvalue 0 in that direction
        let test_vec = DVector::from_element(n, 1.0);
        let result = &deflated * &test_vec;
        // All rows sum should be 0
        for i in 0..n {
            assert!(result[i].abs() < 1e-10);
        }
    }

    #[test]
    fn test_hotelling_deflation_eigenvector() {
        // Build a 2x2 matrix: [[2, 1], [1, 2]]
        let mat = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        // Eigenvector [1, 1]/√2, eigenvalue 3
        let e = DVector::from_row_slice(&[1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()]);
        let deflated = hotelling_deflation(&mat, &e, 3.0);
        // After deflation, the residual matrix should have eigenvalue 0 in that direction
        let test = &deflated * &e;
        assert!(test[0].abs() < 1e-10);
        assert!(test[1].abs() < 1e-10);
    }

    #[test]
    fn test_hotelling_deflation_zero_eigenvalue() {
        let mat = DMatrix::identity(3, 3);
        let e = DVector::from_row_slice(&[1.0, 0.0, 0.0]);
        let deflated = hotelling_deflation(&mat, &e, 0.0);
        // Should be unchanged
        assert!((deflated[(0, 0)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hotelling_deflation_preserves_symmetry() {
        let n = 5;
        let mat = cycle_laplacian(n);
        let eigen = SymmetricEigen::new(mat.clone());
        let evec = eigen.eigenvectors.column(0).into_owned();
        let ev = eigen.eigenvalues[0];
        let deflated = hotelling_deflation(&mat, &evec, ev);
        for i in 0..n {
            for j in 0..n {
                assert!((deflated[(i, j)] - deflated[(j, i)]).abs() < 1e-10,
                    "Deflated matrix is not symmetric at ({i},{j})");
            }
        }
    }

    // ---------------------------------------------------------------
    // 7. Spectral budget edge cases
    // ---------------------------------------------------------------
    #[test]
    fn test_budget_single_vertex() {
        let l = DMatrix::from_row_slice(1, 1, &[0.0]);
        let budget = compute_spectral_budget(&l);
        assert!(budget.eigenvalues[0].abs() < 1e-10);
    }

    #[test]
    fn test_budget_two_vertices() {
        // Two vertices, one edge
        let mut l = DMatrix::zeros(2, 2);
        l[(0, 0)] = 1.0;
        l[(1, 1)] = 1.0;
        l[(0, 1)] = -1.0;
        l[(1, 0)] = -1.0;
        let budget = compute_spectral_budget(&l);
        assert!(budget.eigenvalues[0].abs() < 1e-10);
        assert!((budget.eigenvalues[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_budget_positive_semidefinite() {
        let l = cycle_laplacian(5);
        let budget = compute_spectral_budget(&l);
        for &v in budget.eigenvalues.iter() {
            assert!(v >= -1e-10, "Eigenvalue {v} is negative (not PSD)");
        }
    }

    #[test]
    fn test_budget_all_nonzero_for_complete() {
        let l = complete_laplacian(6);
        let budget = compute_spectral_budget(&l);
        // K₆: eigenvalues 0 (once), 6 (5 times)
        assert!(budget.eigenvalues[0].abs() < 1e-10);
        for i in 1..6 {
            assert!((budget.eigenvalues[i] - 6.0).abs() < 1e-10,
                "K₆ eigenvalue {i} should be 6, got {}", budget.eigenvalues[i]);
        }
    }

    // ---------------------------------------------------------------
    // 8. Cheeger inequality on various graphs
    // ---------------------------------------------------------------
    #[test]
    fn test_cheeger_inequality_path3() {
        let l = path_laplacian(3);
        assert!(verify_cheeger_inequality(&l));
    }

    #[test]
    fn test_cheeger_inequality_path5() {
        let l = path_laplacian(5);
        assert!(verify_cheeger_inequality(&l));
    }

    #[test]
    fn test_cheeger_inequality_c4() {
        let l = cycle_laplacian(4);
        assert!(verify_cheeger_inequality(&l));
    }

    #[test]
    fn test_cheeger_inequality_c8() {
        let l = cycle_laplacian(8);
        assert!(verify_cheeger_inequality(&l));
    }

    #[test]
    fn test_cheeger_inequality_c10() {
        let l = cycle_laplacian(10);
        assert!(verify_cheeger_inequality(&l));
    }

    #[test]
    fn test_cheeger_inequality_k3() {
        let l = complete_laplacian(3);
        assert!(verify_cheeger_inequality(&l));
    }

    #[test]
    fn test_cheeger_inequality_k5() {
        let l = complete_laplacian(5);
        assert!(verify_cheeger_inequality(&l));
    }

    // ---------------------------------------------------------------
    // 9. Trace conservation (sum of eigenvalues = trace of Laplacian)
    // ---------------------------------------------------------------
    #[test]
    fn test_trace_conservation_path5() {
        let l = path_laplacian(5);
        let budget = compute_spectral_budget(&l);
        let trace_l: f64 = l.diagonal().sum();
        let trace_evals: f64 = budget.eigenvalues.sum();
        assert!((trace_l - trace_evals).abs() < 1e-10);
    }

    #[test]
    fn test_trace_conservation_path7() {
        let l = path_laplacian(7);
        let budget = compute_spectral_budget(&l);
        let trace_l: f64 = l.diagonal().sum();
        let trace_evals: f64 = budget.eigenvalues.sum();
        assert!((trace_l - trace_evals).abs() < 1e-10);
    }

    #[test]
    fn test_trace_conservation_k3() {
        let l = complete_laplacian(3);
        let budget = compute_spectral_budget(&l);
        let trace_l: f64 = l.diagonal().sum();
        let trace_evals: f64 = budget.eigenvalues.sum();
        assert!((trace_l - trace_evals).abs() < 1e-10);
    }

    #[test]
    fn test_trace_conservation_k6() {
        let l = complete_laplacian(6);
        let budget = compute_spectral_budget(&l);
        let trace_l: f64 = l.diagonal().sum();
        let trace_evals: f64 = budget.eigenvalues.sum();
        assert!((trace_l - trace_evals).abs() < 1e-10);
    }

    // ---------------------------------------------------------------
    // 10. BudgetProfile defaults
    // ---------------------------------------------------------------
    #[test]
    fn test_budget_profile_default() {
        let p = BudgetProfile::default();
        assert!((p.tolerance - 1e-10).abs() < 1e-20);
        assert_eq!(p.max_iterations, 100);
    }

    #[test]
    fn test_budget_profile_custom() {
        let p = BudgetProfile {
            tolerance: 1e-6,
            max_iterations: 50,
        };
        assert!((p.tolerance - 1e-6).abs() < 1e-20);
        assert_eq!(p.max_iterations, 50);
    }

    // ---------------------------------------------------------------
    // 11. Laplacian helper functions
    // ---------------------------------------------------------------
    #[test]
    fn test_cycle_laplacian_structure() {
        let l = cycle_laplacian(4);
        for i in 0..4 {
            assert_eq!(l[(i, i)], 2.0);
            assert_eq!(l[(i, (i + 1) % 4)], -1.0);
            assert_eq!(l[(i, (i + 3) % 4)], -1.0);
        }
    }

    #[test]
    fn test_path_laplacian_structure() {
        let l = path_laplacian(4);
        assert_eq!(l[(0, 0)], 1.0);
        assert_eq!(l[(3, 3)], 1.0);
        assert_eq!(l[(1, 1)], 2.0);
        assert_eq!(l[(2, 2)], 2.0);
    }

    #[test]
    fn test_complete_laplacian_structure() {
        let l = complete_laplacian(4);
        for i in 0..4 {
            assert_eq!(l[(i, i)], 3.0);
            for j in 0..4 {
                if i != j {
                    assert_eq!(l[(i, j)], -1.0);
                }
            }
        }
    }

    #[test]
    fn test_laplacian_to_adjacency() {
        let l = cycle_laplacian(4);
        let a = laplacian_to_adjacency(&l);
        for i in 0..4 {
            for j in 0..4 {
                if i != j && (j == (i + 1) % 4 || j == (i + 3) % 4) {
                    assert_eq!(a[(i, j)], 1.0);
                } else {
                    assert_eq!(a[(i, j)], 0.0);
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // 12. Edge cases
    // ---------------------------------------------------------------
    #[test]
    fn test_cheeger_single_vertex() {
        let l = DMatrix::from_row_slice(1, 1, &[0.0]);
        assert!((cheeger_constant(&l) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cheeger_two_vertices() {
        let mut l = DMatrix::zeros(2, 2);
        l[(0, 0)] = 1.0;
        l[(1, 1)] = 1.0;
        l[(0, 1)] = -1.0;
        l[(1, 0)] = -1.0;
        let h = cheeger_constant(&l);
        // Single edge: cut 1 vs 1 = 1/1 = 1
        assert!((h - 1.0).abs() < 1e-10, "2-vertex Cheeger should be 1, got {h}");
    }

    #[test]
    fn test_verify_cheeger_single_vertex() {
        let l = DMatrix::from_row_slice(1, 1, &[0.0]);
        assert!(verify_cheeger_inequality(&l));
    }

    #[test]
    fn test_verify_cheeger_two_vertices() {
        let mut l = DMatrix::zeros(2, 2);
        l[(0, 0)] = 1.0;
        l[(1, 1)] = 1.0;
        l[(0, 1)] = -1.0;
        l[(1, 0)] = -1.0;
        // λ₂=2, h=1, Δ=1 → RHS=1/2=0.5, λ₂=2 >= 0.5 ✓
        assert!(verify_cheeger_inequality(&l));
    }
}
