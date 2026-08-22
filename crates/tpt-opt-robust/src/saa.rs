//! Sample average approximation (SAA) with statistical confidence intervals.
//!
//! SAA replaces the true expectation `E_ξ[Q(x, ξ)]` by the sample average
//! over `N` i.i.d. samples and solves the resulting deterministic problem.
//! Repeating this for `k` independent replications yields:
//!
//! - a statistical **lower bound** on the true optimum from the replication
//!   objective values (`mean(ô_k) ± t·s/√k`),
//! - a statistical **upper bound** from evaluating the best candidate on a
//!   large independent validation sample,
//! - an estimated **optimality gap** with its own confidence interval.
//!
//! The solver is generic: the caller supplies closures that (a) solve one
//! sampled problem returning `(x̂, ô)` and (b) evaluate a candidate on one
//! scenario draw — so any underlying model technology can be plugged in.

use std::vec::Vec;

use tpt_math_prob::Xoshiro256;

/// Configuration for [`SaaSolver`].
#[derive(Debug, Clone)]
pub struct SaaConfig {
    /// Number of samples per replication (the SAA sample size `N`).
    pub samples_per_replication: usize,
    /// Number of independent replications `k`.
    pub replications: usize,
    /// Validation sample size for upper-bound estimation.
    pub validation_samples: usize,
    /// Confidence level in `(0, 1)` (e.g. `0.95`).
    pub confidence: f64,
    /// Deterministic seed driving all sampling.
    pub seed: u64,
}

impl Default for SaaConfig {
    fn default() -> Self {
        Self {
            samples_per_replication: 200,
            replications: 30,
            validation_samples: 10_000,
            confidence: 0.95,
            seed: 0,
        }
    }
}

/// Statistical output of an SAA run.
#[derive(Debug, Clone)]
pub struct SaaResult {
    /// Best candidate solution found across replications.
    pub x_best: Vec<f64>,
    /// Mean replication objective (statistical lower bound estimate).
    pub lower_bound: f64,
    /// Half-width of the lower-bound CI.
    pub lower_bound_half_width: f64,
    /// Validation-based upper bound estimate.
    pub upper_bound: f64,
    /// Half-width of the upper-bound CI.
    pub upper_bound_half_width: f64,
    /// Estimated optimality gap `UB − LB`.
    pub gap: f64,
    /// Half-width of the gap CI.
    pub gap_half_width: f64,
}

/// Generic SAA driver.
///
/// - `solve_sampled`: given `N` scenario draws, solve the SAA problem and
///   return `(x̂, ô)` where `ô = min_x (1/N) Σ q(x, ξ_i)`.
/// - `evaluate`: compute `q(x, ξ)` for one candidate and one scenario draw.
/// - `draw`: produce one scenario sample from the RNG.
pub struct SaaSolver<FS, FE, FD> {
    config: SaaConfig,
    solve_sampled: FS,
    evaluate: FE,
    draw: FD,
}

impl<FS, FE, FD> SaaSolver<FS, FE, FD>
where
    FS: FnMut(&[Vec<f64>]) -> Result<(Vec<f64>, f64), String>,
    FE: Fn(&[f64], &Vec<f64>) -> f64,
    FD: FnMut(&mut Xoshiro256) -> Vec<f64>,
{
    /// Create an SAA driver with the given configuration and callbacks.
    pub fn new(config: SaaConfig, solve_sampled: FS, evaluate: FE, draw: FD) -> Self {
        Self { config, solve_sampled, evaluate, draw }
    }

    /// Run the full SAA procedure and return the statistical estimates.
    pub fn run(mut self) -> Result<SaaResult, String> {
        let mut rng = Xoshiro256::new(self.config.seed);
        let k = self.config.replications.max(2);

        // Replication phase: independent sample sets -> candidates + values.
        let mut obj_values: Vec<f64> = Vec::with_capacity(k);
        let mut candidates: Vec<Vec<f64>> = Vec::with_capacity(k);
        for _ in 0..k {
            let mut sample = Vec::with_capacity(self.config.samples_per_replication);
            for _ in 0..self.config.samples_per_replication {
                sample.push((self.draw)(&mut rng));
            }
            let (x_hat, o_hat) = (self.solve_sampled)(&sample)?;
            obj_values.push(o_hat);
            candidates.push(x_hat);
        }

        // Lower bound: mean ± t_{1-α/2,k-1} · s/√k (t approximated by the
        // normal quantile for k ≥ 30; conservative otherwise via 2.0).
        let mean_o = obj_values.iter().sum::<f64>() / k as f64;
        let var_o = obj_values.iter().map(|&o| (o - mean_o).powi(2)).sum::<f64>() / (k - 1) as f64;
        let z = crate::chance::normal_quantile(0.5 + self.config.confidence / 2.0);
        let lb_hw = z * (var_o / k as f64).sqrt();

        // Pick the best candidate by replication objective (ties -> first).
        let best = obj_values
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let x_best = candidates.swap_remove(best);

        // Upper bound: evaluate the best candidate on a large validation set.
        let mut val_vals = Vec::with_capacity(self.config.validation_samples);
        for _ in 0..self.config.validation_samples {
            let xi = (self.draw)(&mut rng);
            val_vals.push((self.evaluate)(&x_best, &xi));
        }
        let n_v = val_vals.len();
        let mean_v = val_vals.iter().sum::<f64>() / n_v as f64;
        let var_v = val_vals.iter().map(|&v| (v - mean_v).powi(2)).sum::<f64>() / (n_v - 1) as f64;
        let ub_hw = z * (var_v / n_v as f64).sqrt();

        let lb = mean_o;
        let ub = mean_v;
        Ok(SaaResult {
            x_best,
            lower_bound: lb,
            lower_bound_half_width: lb_hw,
            upper_bound: ub,
            upper_bound_half_width: ub_hw,
            gap: ub - lb,
            gap_half_width: (lb_hw.powi(2) + ub_hw.powi(2)).sqrt(),
        })
    }
}
