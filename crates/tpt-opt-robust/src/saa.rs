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

use tpt_math_prob::sampler::SplitMix64;

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
    FD: FnMut(&mut SplitMix64) -> Vec<f64>,
{
    /// Create an SAA driver with the given configuration and callbacks.
    pub fn new(config: SaaConfig, solve_sampled: FS, evaluate: FE, draw: FD) -> Self {
        Self { config, solve_sampled, evaluate, draw }
    }

    /// Run the full SAA procedure and return the statistical estimates.
    pub fn run(mut self) -> Result<SaaResult, String> {
        let mut rng = SplitMix64::new(self.config.seed);
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

        // Lower bound: mean ± t_{1-α/2,k-1} · s/√k. The replication count is
        // small, so the Student-t quantile (not the normal one) is required.
        let p = 0.5 + self.config.confidence / 2.0;
        let mean_o = obj_values.iter().sum::<f64>() / k as f64;
        let var_o = obj_values.iter().map(|&o| (o - mean_o).powi(2)).sum::<f64>() / (k - 1) as f64;
        let t = student_t_quantile(p, (k - 1) as f64);
        let lb_hw = t * (var_o / k as f64).sqrt();

        // Upper bound uses the validation sample mean, which is asymptotically
        // normal, so the normal quantile is appropriate there.
        let z = crate::chance::normal_quantile(p);

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

/// Natural log of the gamma function (Lanczos approximation, g = 7).
fn ln_gamma(x: f64) -> f64 {
    const LANCZOS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let t = x + 7.5;
    let series = LANCZOS
        .iter()
        .enumerate()
        .skip(1)
        .fold(LANCZOS[0], |acc, (i, &c)| acc + c / (x + i as f64));
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + series.ln()
}

/// Student-t quantile `t` with `P(T_dof ≤ t) = p`, for `p ∈ (0.5, 1)`.
///
/// Solved by bisection on the CDF, which is computed by Simpson integration of
/// the density from `0` (where `F = 1/2`). Accurate to well below the
/// precision a confidence interval needs, for any `dof ≥ 1`.
fn student_t_quantile(p: f64, dof: f64) -> f64 {
    const STEPS: usize = 4_000;
    let log_norm =
        ln_gamma((dof + 1.0) / 2.0) - ln_gamma(dof / 2.0) - 0.5 * (dof * std::f64::consts::PI).ln();
    let density = |t: f64| (log_norm - (dof + 1.0) / 2.0 * (1.0 + t * t / dof).ln()).exp();
    // Mass between 0 and t, by composite Simpson's rule.
    let mass = |t: f64| {
        let h = t / STEPS as f64;
        let mut sum = density(0.0) + density(t);
        for i in 1..STEPS {
            let w = if i % 2 == 1 { 4.0 } else { 2.0 };
            sum += w * density(i as f64 * h);
        }
        sum * h / 3.0
    };
    let target = p - 0.5;
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    while mass(hi) < target {
        hi *= 2.0;
    }
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if mass(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::student_t_quantile;

    #[test]
    fn student_t_quantile_matches_tables() {
        // Two-sided 95% critical values (p = 0.975) from standard t-tables.
        let cases = [(1.0, 12.706), (4.0, 2.776), (9.0, 2.262), (29.0, 2.045)];
        for (dof, expected) in cases {
            let t = student_t_quantile(0.975, dof);
            assert!((t - expected).abs() < 1e-3, "dof {dof}: got {t}, want {expected}");
        }
    }

    #[test]
    fn student_t_quantile_approaches_normal_for_large_dof() {
        let t = student_t_quantile(0.975, 1_000.0);
        assert!((t - 1.962).abs() < 2e-3, "got {t}");
    }
}
