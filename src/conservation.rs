//! Conservation laws discovered from empirical analysis of game-theory tile systems.
//!
//! Key findings:
//! - N-player scaling law: score standard deviation scales as N^(-0.871)
//! - Temporal dynamics: negative-space changes evolve 0.68× slower than positive-space
//! - No phase transition: magnetization remains flat across temperature parameter

use nalgebra::DVector;

/// Empirically determined scaling exponent for N-player score variance.
/// score_std ~ N^(-ALPHA)
pub const N_PLAYER_SCALING_ALPHA: f64 = -0.871;

/// Ratio of negative-space change rate to positive-space change rate.
pub const TEMPORAL_SLOWDOWN_RATIO: f64 = 0.68;

/// Predict the score standard deviation for N players.
///
/// Uses the empirically determined power law: σ(N) = N^(-0.871)
pub fn predicted_score_std(n_players: usize) -> f64 {
    if n_players == 0 {
        return 0.0;
    }
    (n_players as f64).powf(N_PLAYER_SCALING_ALPHA)
}

/// Predict the score variance for N players: σ²(N) = N^(2 × -0.871) = N^(-1.742)
pub fn predicted_score_variance(n_players: usize) -> f64 {
    if n_players == 0 {
        return 0.0;
    }
    (n_players as f64).powf(2.0 * N_PLAYER_SCALING_ALPHA)
}

/// Compute the temporal dynamics factor for negative-space evolution.
///
/// Negative-space changes evolve TEMPORAL_SLOWDOWN_RATIO × slower than positive-space.
pub fn negative_space_rate(positive_rate: f64) -> f64 {
    positive_rate * TEMPORAL_SLOWDOWN_RATIO
}

/// Verify the N-player scaling law against observed data.
///
/// Given vectors of player counts and observed score standard deviations,
/// returns the fitted exponent and R² goodness of fit.
pub fn verify_scaling_law(
    player_counts: &[usize],
    observed_stds: &[f64],
) -> ScalingLawFit {
    assert_eq!(
        player_counts.len(),
        observed_stds.len(),
        "player counts and observed stds must have same length"
    );
    let n = player_counts.len();
    if n < 2 {
        return ScalingLawFit {
            fitted_alpha: N_PLAYER_SCALING_ALPHA,
            r_squared: 0.0,
        };
    }

    // Fit log(σ) = α·log(N) + c via least squares
    let log_n: Vec<f64> = player_counts
        .iter()
        .map(|&n| (n as f64).ln())
        .collect();
    let log_sigma: Vec<f64> = observed_stds.iter().map(|s| s.ln()).collect();

    let mean_x: f64 = log_n.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = log_sigma.iter().sum::<f64>() / n as f64;

    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    let mut ss_tot = 0.0;
    for i in 0..n {
        let dx = log_n[i] - mean_x;
        let dy = log_sigma[i] - mean_y;
        ss_xy += dx * dy;
        ss_xx += dx * dx;
    }

    let fitted_alpha = if ss_xx.abs() > 1e-15 {
        ss_xy / ss_xx
    } else {
        N_PLAYER_SCALING_ALPHA
    };

    // R² calculation
    let intercept = mean_y - fitted_alpha * mean_x;
    let mut ss_res = 0.0;
    for i in 0..n {
        let dy = log_sigma[i] - mean_y;
        ss_tot += dy * dy;
        let predicted = fitted_alpha * log_n[i] + intercept;
        let residual = log_sigma[i] - predicted;
        ss_res += residual * residual;
    }

    let r_squared = if ss_tot.abs() > 1e-15 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };

    ScalingLawFit {
        fitted_alpha,
        r_squared,
    }
}

/// Result of fitting the N-player scaling law.
#[derive(Debug, Clone)]
pub struct ScalingLawFit {
    /// The fitted exponent α in σ ~ N^α
    pub fitted_alpha: f64,
    /// R² goodness of fit (1.0 = perfect)
    pub r_squared: f64,
}

/// Check for phase transitions in magnetization data across a temperature parameter.
///
/// Returns true if magnetization is essentially flat (no phase transition detected).
/// A phase transition would show a sharp change in magnetization slope.
pub fn check_phase_transition(
    temperatures: &[f64],
    magnetization: &[f64],
    slope_threshold: f64,
) -> PhaseTransitionResult {
    assert_eq!(
        temperatures.len(),
        magnetization.len(),
        "temperature and magnetization vectors must have same length"
    );
    let n = temperatures.len();
    if n < 3 {
        return PhaseTransitionResult {
            has_transition: false,
            max_slope: 0.0,
            mean_magnetization: magnetization.iter().sum::<f64>() / n.max(1) as f64,
        };
    }

    // Compute slopes: dM/dT
    let mut slopes = Vec::with_capacity(n - 1);
    for i in 1..n {
        let dt = temperatures[i] - temperatures[i - 1];
        let dm = magnetization[i] - magnetization[i - 1];
        if dt.abs() > 1e-15 {
            slopes.push(dm / dt);
        }
    }

    let max_slope = slopes.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
    let mean_mag: f64 = magnetization.iter().sum::<f64>() / n as f64;
    let has_transition = max_slope > slope_threshold;

    PhaseTransitionResult {
        has_transition,
        max_slope,
        mean_magnetization: mean_mag,
    }
}

/// Result of phase transition analysis.
#[derive(Debug, Clone)]
pub struct PhaseTransitionResult {
    /// Whether a phase transition was detected
    pub has_transition: bool,
    /// Maximum absolute slope dM/dT
    pub max_slope: f64,
    /// Mean magnetization across all temperatures
    pub mean_magnetization: f64,
}

/// Simulate temporal evolution of a score vector with asymmetric positive/negative rates.
///
/// Returns the evolved score vector after `steps` iterations, where negative-space
/// entries change at 0.68× the rate of positive-space entries.
pub fn temporal_evolution(
    initial_scores: &DVector<f64>,
    positive_rate: f64,
    steps: usize,
) -> DVector<f64> {
    let mut scores = initial_scores.clone();
    let neg_rate = negative_space_rate(positive_rate);
    for _ in 0..steps {
        for i in 0..scores.nrows() {
            let rate = if scores[i] >= 0.0 {
                positive_rate
            } else {
                neg_rate
            };
            scores[i] += rate * scores[i].signum();
        }
    }
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaling_law_n1() {
        let std = predicted_score_std(1);
        assert!((std - 1.0).abs() < 1e-10, "N=1: σ should be 1, got {std}");
    }

    #[test]
    fn test_scaling_law_n2() {
        let std = predicted_score_std(2);
        let expected = 2.0_f64.powf(-0.871);
        assert!(
            (std - expected).abs() < 1e-10,
            "N=2: σ should be {expected}, got {std}"
        );
    }

    #[test]
    fn test_scaling_law_decreasing() {
        // More players → lower std
        let std_1 = predicted_score_std(1);
        let std_5 = predicted_score_std(5);
        let std_10 = predicted_score_std(10);
        assert!(std_1 > std_5, "σ(1) > σ(5) should hold");
        assert!(std_5 > std_10, "σ(5) > σ(10) should hold");
    }

    #[test]
    fn test_scaling_law_zero() {
        assert_eq!(predicted_score_std(0), 0.0);
    }

    #[test]
    fn test_variance_consistency() {
        let n = 7;
        let std = predicted_score_std(n);
        let var = predicted_score_variance(n);
        assert!(
            (var - std * std).abs() < 1e-10,
            "variance should equal std²"
        );
    }

    #[test]
    fn test_negative_space_rate() {
        let rate = negative_space_rate(1.0);
        assert!((rate - 0.68).abs() < 1e-10, "should be 0.68, got {rate}");
    }

    #[test]
    fn test_negative_space_rate_custom() {
        let rate = negative_space_rate(2.5);
        assert!((rate - 1.7).abs() < 1e-10, "should be 1.7, got {rate}");
    }

    #[test]
    fn test_verify_scaling_law_perfect() {
        // Generate data that perfectly follows the law
        let counts: Vec<usize> = vec![1, 2, 3, 5, 10, 20];
        let stds: Vec<f64> = counts.iter().map(|&n| (n as f64).powf(-0.871)).collect();

        let fit = verify_scaling_law(&counts, &stds);
        assert!(
            (fit.fitted_alpha - (-0.871)).abs() < 1e-8,
            "fitted alpha should be -0.871, got {}",
            fit.fitted_alpha
        );
        assert!(
            (fit.r_squared - 1.0).abs() < 1e-8,
            "R² should be 1.0 for perfect data, got {}",
            fit.r_squared
        );
    }

    #[test]
    fn test_verify_scaling_law_noisy() {
        let counts: Vec<usize> = vec![2, 4, 8, 16, 32];
        let stds: Vec<f64> = counts
            .iter()
            .map(|&n| (n as f64).powf(-0.871) + 0.01 * (n as f64).sin())
            .collect();

        let fit = verify_scaling_law(&counts, &stds);
        // Should still be close to -0.871
        assert!(
            (fit.fitted_alpha - (-0.871)).abs() < 0.1,
            "fitted alpha should be close to -0.871, got {}",
            fit.fitted_alpha
        );
        assert!(
            fit.r_squared > 0.95,
            "R² should be high for slightly noisy data, got {}",
            fit.r_squared
        );
    }

    #[test]
    fn test_phase_transition_flat() {
        let temps: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();
        let mag: Vec<f64> = vec![0.5; 100]; // completely flat

        let result = check_phase_transition(&temps, &mag, 0.1);
        assert!(!result.has_transition, "flat magnetization should have no transition");
        assert!(
            result.max_slope.abs() < 1e-10,
            "max slope should be ~0 for flat data"
        );
        assert!((result.mean_magnetization - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_phase_transition_detected() {
        let temps: Vec<f64> = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let mag: Vec<f64> = vec![1.0, 0.95, 0.0, 0.05, 0.02, 0.01, 0.01]; // sharp drop

        let result = check_phase_transition(&temps, &mag, 0.1);
        assert!(
            result.has_transition,
            "sharp drop should be detected, max_slope={}",
            result.max_slope
        );
    }

    #[test]
    fn test_phase_transition_too_few_points() {
        let temps = vec![1.0];
        let mag = vec![0.5];
        let result = check_phase_transition(&temps, &mag, 0.1);
        assert!(!result.has_transition);
    }

    #[test]
    fn test_temporal_evolution_positive_only() {
        let initial = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let evolved = temporal_evolution(&initial, 0.1, 1);
        assert!(
            (evolved[0] - 1.1).abs() < 1e-10,
            "positive entry should increase by rate"
        );
        assert!((evolved[1] - 2.1).abs() < 1e-10);
        assert!((evolved[2] - 3.1).abs() < 1e-10);
    }

    #[test]
    fn test_temporal_evolution_negative_slower() {
        let initial = DVector::from_vec(vec![-1.0, 1.0]);
        let evolved = temporal_evolution(&initial, 1.0, 1);
        // Negative space changes at 0.68 rate
        assert!(
            (evolved[0] - (-1.68)).abs() < 1e-10,
            "negative entry should change at 0.68× rate, got {}",
            evolved[0]
        );
        // Positive space changes at full rate
        assert!((evolved[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_temporal_evolution_zero_steps() {
        let initial = DVector::from_vec(vec![1.0, -2.0]);
        let evolved = temporal_evolution(&initial, 1.0, 0);
        assert!((evolved[0] - 1.0).abs() < 1e-10);
        assert!((evolved[1] - (-2.0)).abs() < 1e-10);
    }
}
