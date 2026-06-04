//! Tile-graph compiler optimizations inspired by spectral topology analysis.
//!
//! Applies traditional compiler techniques to tile-based game-theory graphs:
//! - Dead code elimination: remove tiles with visit count = 0
//! - Constant folding: tiles where score variance is below threshold
//! - SVD factorization: rank-r approximation of the score matrix
//! - JIT compilation: compile only tiles above a visit threshold

use nalgebra::DMatrix;

/// A tile in the game-theory graph.
#[derive(Debug, Clone)]
pub struct Tile {
    pub id: usize,
    pub score: f64,
    pub visit_count: usize,
    pub score_history: Vec<f64>,
}

/// Configuration for the tile compiler.
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    /// Variance threshold for constant folding — tiles with score variance
    /// below this are treated as constants and folded.
    pub variance_threshold: f64,
    /// Minimum visit count for JIT compilation — tiles below this threshold
    /// are not compiled (lazy evaluation).
    pub jit_visit_threshold: usize,
    /// Target rank for SVD factorization of the score matrix.
    pub svd_rank: usize,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        CompilerConfig {
            variance_threshold: 1e-6,
            jit_visit_threshold: 10,
            svd_rank: 5,
        }
    }
}

/// Result of dead code elimination.
#[derive(Debug, Clone)]
pub struct DeadCodeResult {
    /// Tiles that survived (visit_count > 0)
    pub alive_tiles: Vec<Tile>,
    /// Tiles that were eliminated (visit_count == 0)
    pub eliminated_tiles: Vec<Tile>,
    /// Fraction of tiles eliminated
    pub elimination_ratio: f64,
}

/// Dead code elimination: remove tiles with visit count = 0.
///
/// Returns surviving tiles and eliminated tiles, plus the elimination ratio.
pub fn dead_code_elimination(tiles: &[Tile]) -> DeadCodeResult {
    let mut alive = Vec::new();
    let mut eliminated = Vec::new();

    for tile in tiles {
        if tile.visit_count == 0 {
            eliminated.push(tile.clone());
        } else {
            alive.push(tile.clone());
        }
    }

    let total = tiles.len().max(1);
    let elimination_ratio = eliminated.len() as f64 / total as f64;

    DeadCodeResult {
        alive_tiles: alive,
        eliminated_tiles: eliminated,
        elimination_ratio,
    }
}

/// Result of constant folding.
#[derive(Debug, Clone)]
pub struct ConstantFoldResult {
    /// Tiles identified as constants (score variance < threshold)
    pub constant_tiles: Vec<Tile>,
    /// Tiles still variable
    pub variable_tiles: Vec<Tile>,
    /// The constant value assigned to each constant tile (mean of history)
    pub constant_values: Vec<f64>,
    /// Fraction of tiles folded
    pub fold_ratio: f64,
}

/// Constant folding: identify tiles where score variance is below threshold.
///
/// Such tiles are treated as constants and replaced with their mean score value.
pub fn constant_folding(tiles: &[Tile], config: &CompilerConfig) -> ConstantFoldResult {
    let mut constant_tiles = Vec::new();
    let mut variable_tiles = Vec::new();
    let mut constant_values = Vec::new();

    for tile in tiles {
        let variance = compute_variance(&tile.score_history);
        if variance < config.variance_threshold {
            let mean = if tile.score_history.is_empty() {
                tile.score
            } else {
                tile.score_history.iter().sum::<f64>() / tile.score_history.len() as f64
            };
            constant_tiles.push(tile.clone());
            constant_values.push(mean);
        } else {
            variable_tiles.push(tile.clone());
        }
    }

    let total = tiles.len().max(1);
    let fold_ratio = constant_tiles.len() as f64 / total as f64;

    ConstantFoldResult {
        constant_tiles,
        variable_tiles,
        constant_values,
        fold_ratio,
    }
}

/// Compute variance of a slice of f64 values.
fn compute_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
}

/// Result of SVD factorization.
#[derive(Debug, Clone)]
pub struct SvdFactorization {
    /// The rank-r approximation matrix
    pub approximated: DMatrix<f64>,
    /// The rank used
    pub rank: usize,
    /// Reconstruction error (Frobenius norm of difference)
    pub reconstruction_error: f64,
    /// Fraction of variance explained by the rank-r approximation
    pub variance_explained: f64,
}

/// SVD factorization: compute a rank-r approximation of the tile score matrix.
///
/// The score matrix has tiles as rows and game episodes as columns.
/// Uses nalgebra's SVD to compute the low-rank approximation.
pub fn svd_factorization(
    score_matrix: &DMatrix<f64>,
    rank: usize,
) -> SvdFactorization {
    let (m, n) = score_matrix.shape();
    let effective_rank = rank.min(m).min(n);

    if effective_rank == 0 || m == 0 || n == 0 {
        return SvdFactorization {
            approximated: DMatrix::zeros(m, n),
            rank: 0,
            reconstruction_error: 0.0,
            variance_explained: 0.0,
        };
    }

    let svd = score_matrix.clone().svd(true, true);

    // Reconstruct using only the top-r singular values
    let sigma = svd.singular_values;
    let u = svd.u.unwrap_or_else(|| DMatrix::identity(m, m));
    let vt = svd.v_t.unwrap_or_else(|| DMatrix::identity(n, n));

    // Build rank-r approximation: U[:, :r] * Σ[:r, :r] * V^T[:r, :]
    let u_r = u.columns(0, effective_rank);
    let vt_r = vt.rows(0, effective_rank);
    let mut sigma_r = nalgebra::DMatrix::zeros(effective_rank, effective_rank);
    for i in 0..effective_rank {
        sigma_r[(i, i)] = sigma[i];
    }

    let approx = &u_r * &sigma_r * &vt_r;

    // Reconstruction error
    let diff = score_matrix - &approx;
    let recon_error = diff.norm();

    // Variance explained
    let total_variance: f64 = sigma.iter().map(|s| s * s).sum();
    let explained: f64 = sigma.iter().take(effective_rank).map(|s| s * s).sum();
    let var_explained = if total_variance > 1e-15 {
        explained / total_variance
    } else {
        1.0
    };

    SvdFactorization {
        approximated: approx,
        rank: effective_rank,
        reconstruction_error: recon_error,
        variance_explained: var_explained,
    }
}

/// Result of JIT compilation pass.
#[derive(Debug, Clone)]
pub struct JitResult {
    /// Tiles that passed the visit threshold (compiled)
    pub compiled_tiles: Vec<Tile>,
    /// Tiles below the visit threshold (deferred)
    pub deferred_tiles: Vec<Tile>,
    /// Compilation ratio
    pub compilation_ratio: f64,
}

/// JIT compilation: compile only tiles that exceed the visit threshold.
///
/// Tiles below the threshold are deferred for lazy evaluation.
pub fn jit_compile(tiles: &[Tile], config: &CompilerConfig) -> JitResult {
    let mut compiled = Vec::new();
    let mut deferred = Vec::new();

    for tile in tiles {
        if tile.visit_count >= config.jit_visit_threshold {
            compiled.push(tile.clone());
        } else {
            deferred.push(tile.clone());
        }
    }

    let total = tiles.len().max(1);
    let compilation_ratio = compiled.len() as f64 / total as f64;

    JitResult {
        compiled_tiles: compiled,
        deferred_tiles: deferred,
        compilation_ratio,
    }
}

/// Run the full compiler pipeline: DCE → constant fold → JIT compile.
///
/// Returns the final set of compiled tiles after all optimization passes.
pub fn compile_pipeline(tiles: &[Tile], config: &CompilerConfig) -> CompilePipelineResult {
    // Pass 1: Dead code elimination
    let dce = dead_code_elimination(tiles);

    // Pass 2: Constant folding on alive tiles
    let fold = constant_folding(&dce.alive_tiles, config);

    // Pass 3: JIT compile variable tiles
    let jit = jit_compile(&fold.variable_tiles, config);

    CompilePipelineResult {
        dce_eliminated: dce.eliminated_tiles.len(),
        constants_folded: fold.constant_tiles.len(),
        compiled_count: jit.compiled_tiles.len(),
        deferred_count: jit.deferred_tiles.len(),
        final_tiles: jit.compiled_tiles,
        constant_values: fold.constant_values,
    }
}

/// Result of the full compilation pipeline.
#[derive(Debug, Clone)]
pub struct CompilePipelineResult {
    /// Number of tiles eliminated by DCE
    pub dce_eliminated: usize,
    /// Number of tiles folded as constants
    pub constants_folded: usize,
    /// Number of tiles compiled by JIT
    pub compiled_count: usize,
    /// Number of tiles deferred by JIT
    pub deferred_count: usize,
    /// Final compiled tiles
    pub final_tiles: Vec<Tile>,
    /// Constant values for folded tiles
    pub constant_values: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tile(id: usize, score: f64, visits: usize, history: Vec<f64>) -> Tile {
        Tile {
            id,
            score,
            visit_count: visits,
            score_history: history,
        }
    }

    // --- Dead code elimination tests ---

    #[test]
    fn test_dce_removes_zero_visits() {
        let tiles = vec![
            make_tile(0, 1.0, 5, vec![1.0, 1.0, 1.0]),
            make_tile(1, 2.0, 0, vec![2.0]),
            make_tile(2, 3.0, 3, vec![3.0, 3.0]),
        ];
        let result = dead_code_elimination(&tiles);
        assert_eq!(result.alive_tiles.len(), 2);
        assert_eq!(result.eliminated_tiles.len(), 1);
        assert_eq!(result.eliminated_tiles[0].id, 1);
        assert!((result.elimination_ratio - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dce_no_elimination() {
        let tiles = vec![
            make_tile(0, 1.0, 1, vec![]),
            make_tile(1, 2.0, 2, vec![]),
        ];
        let result = dead_code_elimination(&tiles);
        assert_eq!(result.alive_tiles.len(), 2);
        assert_eq!(result.eliminated_tiles.len(), 0);
        assert!((result.elimination_ratio).abs() < 1e-10);
    }

    #[test]
    fn test_dce_all_dead() {
        let tiles = vec![
            make_tile(0, 1.0, 0, vec![]),
            make_tile(1, 2.0, 0, vec![]),
        ];
        let result = dead_code_elimination(&tiles);
        assert_eq!(result.alive_tiles.len(), 0);
        assert_eq!(result.eliminated_tiles.len(), 2);
        assert!((result.elimination_ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dce_empty() {
        let result = dead_code_elimination(&[]);
        assert_eq!(result.alive_tiles.len(), 0);
        assert!((result.elimination_ratio).abs() < 1e-10);
    }

    // --- Constant folding tests ---

    #[test]
    fn test_constant_fold_constant_tiles() {
        let tiles = vec![
            make_tile(0, 1.0, 5, vec![1.0, 1.0, 1.0, 1.0]),
            make_tile(1, 2.0, 5, vec![2.0, 2.0, 2.0, 2.0]),
        ];
        let config = CompilerConfig::default();
        let result = constant_folding(&tiles, &config);
        assert_eq!(result.constant_tiles.len(), 2);
        assert_eq!(result.variable_tiles.len(), 0);
        assert!((result.constant_values[0] - 1.0).abs() < 1e-10);
        assert!((result.constant_values[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_constant_fold_variable_tiles() {
        let tiles = vec![
            make_tile(0, 1.0, 5, vec![0.0, 5.0, 10.0]),
            make_tile(1, 2.0, 5, vec![-10.0, 10.0]),
        ];
        let config = CompilerConfig::default();
        let result = constant_folding(&tiles, &config);
        assert_eq!(result.constant_tiles.len(), 0);
        assert_eq!(result.variable_tiles.len(), 2);
    }

    #[test]
    fn test_constant_fold_mixed() {
        let tiles = vec![
            make_tile(0, 1.0, 5, vec![1.0, 1.0, 1.0]),
            make_tile(1, 2.0, 5, vec![0.0, 10.0]),
        ];
        let config = CompilerConfig::default();
        let result = constant_folding(&tiles, &config);
        assert_eq!(result.constant_tiles.len(), 1);
        assert_eq!(result.variable_tiles.len(), 1);
        assert!((result.fold_ratio - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_constant_fold_empty_history() {
        let tiles = vec![make_tile(0, 3.0, 5, vec![])];
        let config = CompilerConfig::default();
        let result = constant_folding(&tiles, &config);
        // Empty history → variance = 0 → should be constant with value = score
        assert_eq!(result.constant_tiles.len(), 1);
        assert!((result.constant_values[0] - 3.0).abs() < 1e-10);
    }

    // --- SVD factorization tests ---

    #[test]
    fn test_svd_identity() {
        let eye = DMatrix::identity(4, 4);
        let result = svd_factorization(&eye, 2);
        assert_eq!(result.rank, 2);
        // Rank-2 of 4×4 identity should have reconstruction error
        // The two largest singular values of identity are 1.0
        assert!(result.variance_explained > 0.0);
    }

    #[test]
    fn test_svd_rank_one_matrix() {
        // A rank-1 matrix: outer product of [1,2,3] and [4,5]
        let a = nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = nalgebra::DVector::from_vec(vec![4.0, 5.0]);
        let mat = &a * b.transpose();
        let result = svd_factorization(&mat, 1);
        assert_eq!(result.rank, 1);
        assert!(
            result.reconstruction_error < 1e-8,
            "rank-1 matrix should be perfectly reconstructed at rank 1, error = {}",
            result.reconstruction_error
        );
        assert!(
            (result.variance_explained - 1.0).abs() < 1e-8,
            "should explain 100% variance"
        );
    }

    #[test]
    fn test_svd_low_rank_approximation() {
        // Build a 3×3 matrix that's close to rank 2
        let mat = DMatrix::from_row_slice(
            3,
            3,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let result = svd_factorization(&mat, 2);
        assert_eq!(result.rank, 2);
        // This matrix is rank 2 (third row = sum of first two), so rank-2 should be near-perfect
        assert!(
            result.reconstruction_error < 1e-8,
            "rank-2 approximation of rank-2 matrix should be near-perfect, error = {}",
            result.reconstruction_error
        );
        assert!(
            (result.variance_explained - 1.0).abs() < 1e-6,
            "should explain ~100% variance, got {}",
            result.variance_explained
        );
    }

    #[test]
    fn test_svd_zero_rank() {
        let mat = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let result = svd_factorization(&mat, 0);
        assert_eq!(result.rank, 0);
    }

    #[test]
    fn test_svd_exceeds_dimensions() {
        let mat = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = svd_factorization(&mat, 100);
        assert_eq!(result.rank, 2); // capped at min(m, n)
    }

    // --- JIT compilation tests ---

    #[test]
    fn test_jit_above_threshold() {
        let tiles = vec![
            make_tile(0, 1.0, 20, vec![]),
            make_tile(1, 2.0, 50, vec![]),
        ];
        let config = CompilerConfig::default(); // threshold = 10
        let result = jit_compile(&tiles, &config);
        assert_eq!(result.compiled_tiles.len(), 2);
        assert_eq!(result.deferred_tiles.len(), 0);
    }

    #[test]
    fn test_jit_below_threshold() {
        let tiles = vec![
            make_tile(0, 1.0, 5, vec![]),
            make_tile(1, 2.0, 9, vec![]),
        ];
        let config = CompilerConfig::default();
        let result = jit_compile(&tiles, &config);
        assert_eq!(result.compiled_tiles.len(), 0);
        assert_eq!(result.deferred_tiles.len(), 2);
    }

    #[test]
    fn test_jit_mixed() {
        let tiles = vec![
            make_tile(0, 1.0, 5, vec![]),
            make_tile(1, 2.0, 15, vec![]),
            make_tile(2, 3.0, 10, vec![]),
        ];
        let config = CompilerConfig::default();
        let result = jit_compile(&tiles, &config);
        assert_eq!(result.compiled_tiles.len(), 2);
        assert_eq!(result.deferred_tiles.len(), 1);
    }

    // --- Full pipeline tests ---

    #[test]
    fn test_full_pipeline() {
        let tiles = vec![
            make_tile(0, 1.0, 0, vec![1.0, 1.0]),   // dead
            make_tile(1, 2.0, 5, vec![2.0, 2.0]),    // alive, constant, below JIT threshold
            make_tile(2, 3.0, 20, vec![1.0, 5.0]),   // alive, variable, above JIT threshold
            make_tile(3, 4.0, 30, vec![4.0, 4.0]),   // alive, constant, above JIT threshold
        ];
        let config = CompilerConfig::default();
        let result = compile_pipeline(&tiles, &config);

        assert_eq!(result.dce_eliminated, 1);
        assert_eq!(result.constants_folded, 2); // tiles 1 and 3
        assert_eq!(result.compiled_count, 1); // only tile 2
        assert_eq!(result.deferred_count, 0);
    }

    #[test]
    fn test_pipeline_all_dead() {
        let tiles = vec![
            make_tile(0, 1.0, 0, vec![]),
            make_tile(1, 2.0, 0, vec![]),
        ];
        let config = CompilerConfig::default();
        let result = compile_pipeline(&tiles, &config);
        assert_eq!(result.dce_eliminated, 2);
        assert_eq!(result.compiled_count, 0);
    }

    // --- Config tests ---

    #[test]
    fn test_default_config() {
        let config = CompilerConfig::default();
        assert!((config.variance_threshold - 1e-6).abs() < 1e-15);
        assert_eq!(config.jit_visit_threshold, 10);
        assert_eq!(config.svd_rank, 5);
    }

    #[test]
    fn test_custom_config() {
        let config = CompilerConfig {
            variance_threshold: 0.01,
            jit_visit_threshold: 50,
            svd_rank: 10,
        };
        assert!((config.variance_threshold - 0.01).abs() < 1e-15);
        assert_eq!(config.jit_visit_threshold, 50);
        assert_eq!(config.svd_rank, 10);
    }
}
