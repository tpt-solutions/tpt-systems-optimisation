//! Bayesian optimisation with a Gaussian-process surrogate.
//!
//! The surrogate is a zero-mean Gaussian process with a Matérn 5/2 kernel
//! `k(r) = σ_f² (1 + √5 r + 5r²/3) e^(−√5 r)` on inputs scaled to the unit cube
//! (`r = ‖u − u'‖ / ℓ`). Observations are standardised before fitting. The
//! hyperparameters `(ℓ, σ_f, σ_n)` are chosen by maximising the marginal
//! likelihood with [`tpt_opt_core::unconstrained::lbfgs`] (finite-difference
//! gradients in log-space).
//!
//! The acquisition function is expected improvement (EI) for minimisation.
//! It is maximised over uniform random candidates plus Gaussian perturbations
//! of the incumbent, so no inner gradient method is needed.
//!
//! Every random draw comes from a generator seeded by [`BoConfig::seed`], so the
//! same seed and the same objective give the same sequence of evaluations.
//!
//! # Example
//!
//! ```
//! use tpt_opt_bayes::{minimise, BoConfig};
//!
//! // Minimise a 1-D quadratic with minimiser x = 0.3 on [0, 1].
//! let result = minimise(|x| (x[0] - 0.3).powi(2), &[(0.0, 1.0)], 25, &BoConfig::default())
//!     .expect("valid bounds");
//! assert!((result.x[0] - 0.3).abs() < 0.05);
//! ```

use std::f64::consts::PI;
use std::vec;
use std::vec::Vec;

use tpt_opt_core::unconstrained::{lbfgs, UncParams};

/// Errors reported by the Bayesian optimiser.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoError {
    /// The search box is empty, non-finite, or has `lo ≥ hi` in some dimension.
    InvalidBounds,
    /// A point has the wrong dimension for the search box.
    DimensionMismatch,
    /// An objective value was NaN or infinite.
    NonFiniteValue,
}

/// Configuration for [`BayesOpt`] and [`minimise`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoConfig {
    /// Number of uniform random points evaluated before the surrogate is used.
    /// `0` selects `2·dim + 2`.
    pub n_init: usize,
    /// Uniform random candidates scored by EI at each proposal.
    pub n_candidates: usize,
    /// Seed for the deterministic random generator.
    pub seed: u64,
}

impl Default for BoConfig {
    fn default() -> Self {
        Self { n_init: 0, n_candidates: 2000, seed: 0 }
    }
}

/// Outcome of [`minimise`].
#[derive(Clone, Debug, PartialEq)]
pub struct BoResult {
    /// Best point found, in the original (unscaled) coordinates. Empty if no
    /// evaluation was made.
    pub x: Vec<f64>,
    /// Objective value at `x` (`NaN` if no evaluation was made).
    pub f: f64,
    /// Number of objective evaluations performed.
    pub evaluations: usize,
    /// Best-so-far objective value after each evaluation.
    pub history: Vec<f64>,
}

/// Matérn 5/2 hyperparameters in unit-cube input space.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Hyper {
    lengthscale: f64,
    signal_sd: f64,
    noise_sd: f64,
}

/// Sequential Bayesian optimiser over a box.
///
/// Call [`BayesOpt::ask`] for the next point, evaluate the objective there, and
/// report it with [`BayesOpt::tell`].
#[derive(Clone, Debug)]
pub struct BayesOpt {
    bounds: Vec<(f64, f64)>,
    n_init: usize,
    n_candidates: usize,
    rng: SplitMix64,
    /// Observed inputs in the unit cube.
    xs: Vec<Vec<f64>>,
    /// Observed objective values (original sign).
    ys: Vec<f64>,
    hyper: Hyper,
    /// Number of observations at the last hyperparameter fit.
    fitted_at: usize,
}

impl BayesOpt {
    /// Create an optimiser over the box `bounds` (one `(lo, hi)` per dimension).
    pub fn new(bounds: &[(f64, f64)], cfg: BoConfig) -> Result<Self, BoError> {
        if bounds.is_empty()
            || bounds.iter().any(|&(lo, hi)| !lo.is_finite() || !hi.is_finite() || lo >= hi)
        {
            return Err(BoError::InvalidBounds);
        }
        let dim = bounds.len();
        let n_init = if cfg.n_init == 0 { 2 * dim + 2 } else { cfg.n_init };
        Ok(Self {
            bounds: bounds.to_vec(),
            n_init,
            n_candidates: cfg.n_candidates.max(1),
            rng: SplitMix64::new(cfg.seed),
            xs: Vec::new(),
            ys: Vec::new(),
            hyper: Hyper { lengthscale: 0.3, signal_sd: 1.0, noise_sd: 0.01 },
            fitted_at: 0,
        })
    }

    /// Dimension of the search box.
    pub fn dim(&self) -> usize {
        self.bounds.len()
    }

    /// Number of observations reported so far.
    pub fn len(&self) -> usize {
        self.ys.len()
    }

    /// `true` if no observation has been reported.
    pub fn is_empty(&self) -> bool {
        self.ys.is_empty()
    }

    /// Report the objective value `y` observed at `x` (original coordinates).
    pub fn tell(&mut self, x: &[f64], y: f64) -> Result<(), BoError> {
        if x.len() != self.dim() {
            return Err(BoError::DimensionMismatch);
        }
        if !y.is_finite() {
            return Err(BoError::NonFiniteValue);
        }
        let u: Vec<f64> = x
            .iter()
            .zip(&self.bounds)
            .map(|(xi, &(lo, hi))| ((xi - lo) / (hi - lo)).clamp(0.0, 1.0))
            .collect();
        self.xs.push(u);
        self.ys.push(y);
        Ok(())
    }

    /// Best observation so far, as `(x, f)` in original coordinates.
    pub fn best(&self) -> Option<(Vec<f64>, f64)> {
        let (i, &f) = self.ys.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1))?;
        Some((self.unscale(&self.xs[i]), f))
    }

    /// Current hyperparameters `(lengthscale, signal_sd, noise_sd)`, with
    /// inputs in the unit cube and observations standardised.
    pub fn hyperparameters(&self) -> (f64, f64, f64) {
        let h = self.hyper;
        (h.lengthscale, h.signal_sd, h.noise_sd)
    }

    /// Propose the next point to evaluate, in original coordinates.
    pub fn ask(&mut self) -> Vec<f64> {
        if self.ys.len() < self.n_init {
            let u = self.random_unit();
            return self.unscale(&u);
        }
        // Refit the hyperparameters when the data has grown by about a quarter;
        // refitting after every observation dominates the cost of a run.
        let n = self.ys.len();
        if n >= self.fitted_at + (n / 4).max(1) {
            self.refit();
            self.fitted_at = n;
        }
        let Some(model) = Gp::fit(&self.xs, &self.ys, self.hyper) else {
            let u = self.random_unit();
            return self.unscale(&u);
        };
        let best_std = model.best_standardised();

        let (best_idx, _) =
            self.ys.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).unwrap_or((0, &0.0));
        let centre = self.xs[best_idx].clone();

        let mut best_u = centre.clone();
        let mut best_ei = f64::NEG_INFINITY;
        for k in 0..self.n_candidates {
            let u = if k % 4 == 0 {
                // Local exploitation around the incumbent.
                centre
                    .iter()
                    .map(|c| (c + 0.05 * self.rng.next_gaussian()).clamp(0.0, 1.0))
                    .collect()
            } else {
                self.random_unit()
            };
            let (mu, sd) = model.predict(&u);
            let ei = expected_improvement(best_std, mu, sd);
            if ei > best_ei {
                best_ei = ei;
                best_u = u;
            }
        }
        self.unscale(&best_u)
    }

    /// Re-estimate the hyperparameters by maximising the marginal likelihood.
    fn refit(&mut self) {
        let (ys_std, _, _) = standardise(&self.ys);
        let nlml = |theta: &[f64]| -> f64 {
            let h = hyper_from_log(theta);
            match Gp::fit_standardised(&self.xs, &ys_std, h) {
                Some(gp) => gp.neg_log_marginal(),
                None => 1e10,
            }
        };
        let start = [(0.3_f64).ln(), 0.0, (0.01_f64).ln()];
        let params = UncParams { max_iters: 100, f_tol: 1e-10, x_tol: 1e-10, g_tol: 1e-6 };
        let fg = |theta: &[f64], g: &mut [f64]| {
            let f0 = nlml(theta);
            let h = 1e-5;
            let mut t = theta.to_vec();
            for i in 0..theta.len() {
                t[i] = theta[i] + h;
                let fp = nlml(&t);
                t[i] = theta[i] - h;
                let fm = nlml(&t);
                t[i] = theta[i];
                g[i] = (fp - fm) / (2.0 * h);
            }
            f0
        };
        let res = lbfgs(fg, &start, 5, &params);
        if res.f.is_finite() && res.f < 1e9 {
            self.hyper = hyper_from_log(&res.x);
        }
    }

    fn random_unit(&mut self) -> Vec<f64> {
        (0..self.dim()).map(|_| self.rng.next_uniform()).collect()
    }

    fn unscale(&self, u: &[f64]) -> Vec<f64> {
        u.iter().zip(&self.bounds).map(|(ui, &(lo, hi))| lo + ui * (hi - lo)).collect()
    }
}

/// Minimise `f` over the box `bounds` with a total budget of `budget`
/// evaluations.
pub fn minimise<F>(
    mut f: F,
    bounds: &[(f64, f64)],
    budget: usize,
    cfg: &BoConfig,
) -> Result<BoResult, BoError>
where
    F: FnMut(&[f64]) -> f64,
{
    let mut opt = BayesOpt::new(bounds, *cfg)?;
    let mut history = Vec::with_capacity(budget);
    let mut best_f = f64::INFINITY;
    for _ in 0..budget {
        let x = opt.ask();
        let y = f(&x);
        opt.tell(&x, y)?;
        best_f = best_f.min(y);
        history.push(best_f);
    }
    match opt.best() {
        Some((x, f)) => Ok(BoResult { x, f, evaluations: budget, history }),
        None => Ok(BoResult { x: Vec::new(), f: f64::NAN, evaluations: 0, history }),
    }
}

/// Fitted Gaussian process on standardised observations.
struct Gp {
    xs: Vec<Vec<f64>>,
    hyper: Hyper,
    /// Lower Cholesky factor of `K + σ_n² I`.
    chol: Vec<Vec<f64>>,
    /// `α = (K + σ_n² I)⁻¹ y`.
    alpha: Vec<f64>,
    ys_std: Vec<f64>,
    /// Log marginal likelihood terms: `yᵀα/2 + Σ log Lᵢᵢ`.
    fit_term: f64,
}

impl Gp {
    /// Fit to raw observations, standardising them first. Predictions are
    /// then in standardised units, which is all EI needs.
    fn fit(xs: &[Vec<f64>], ys: &[f64], hyper: Hyper) -> Option<Self> {
        let (ys_std, _, _) = standardise(ys);
        Self::fit_standardised(xs, &ys_std, hyper)
    }

    fn fit_standardised(xs: &[Vec<f64>], ys_std: &[f64], hyper: Hyper) -> Option<Self> {
        let n = xs.len();
        let noise_var = hyper.noise_sd * hyper.noise_sd + 1e-10;
        let mut k = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let v = matern52(&xs[i], &xs[j], hyper);
                k[i][j] = v;
                k[j][i] = v;
            }
            k[i][i] += noise_var;
        }
        let chol = cholesky(&k)?;
        let alpha = solve_chol(&chol, ys_std);
        let quad = 0.5 * dot(ys_std, &alpha);
        let logdet: f64 = (0..n).map(|i| chol[i][i].ln()).sum();
        Some(Self {
            xs: xs.to_vec(),
            hyper,
            chol,
            alpha,
            ys_std: ys_std.to_vec(),
            fit_term: quad + logdet,
        })
    }

    /// Negative log marginal likelihood (up to the constant `n/2 · log 2π`).
    fn neg_log_marginal(&self) -> f64 {
        self.fit_term + 0.5 * self.ys_std.len() as f64 * (2.0 * PI).ln()
    }

    /// Posterior mean and standard deviation at the unit-cube point `u`, in
    /// standardised units.
    fn predict(&self, u: &[f64]) -> (f64, f64) {
        let kstar: Vec<f64> = self.xs.iter().map(|xi| matern52(xi, u, self.hyper)).collect();
        let mu_std = dot(&kstar, &self.alpha);
        let v = forward_sub(&self.chol, &kstar);
        let prior = self.hyper.signal_sd * self.hyper.signal_sd;
        let var_std = (prior - dot(&v, &v)).max(1e-12);
        (mu_std, var_std.sqrt())
    }

    /// Best standardised observation (the EI incumbent).
    fn best_standardised(&self) -> f64 {
        self.ys_std.iter().fold(f64::INFINITY, |m, v| m.min(*v))
    }
}

fn hyper_from_log(theta: &[f64]) -> Hyper {
    // Parameters are optimised in log-space; map back and keep them in range.
    Hyper {
        lengthscale: theta[0].exp().clamp(1e-3, 1e2),
        signal_sd: theta[1].exp().clamp(1e-3, 1e3),
        noise_sd: theta[2].exp().clamp(1e-4, 1e2),
    }
}

fn matern52(a: &[f64], b: &[f64], h: Hyper) -> f64 {
    let r2: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let r = r2.sqrt() / h.lengthscale;
    let s5 = 5.0_f64.sqrt();
    h.signal_sd * h.signal_sd * (1.0 + s5 * r + 5.0 * r * r / 3.0) * (-s5 * r).exp()
}

/// Expected improvement for minimisation at posterior `(mu, sd)` with
/// incumbent `best`.
fn expected_improvement(best: f64, mu: f64, sd: f64) -> f64 {
    if sd <= 0.0 {
        return (best - mu).max(0.0);
    }
    let z = (best - mu) / sd;
    sd * (z * normal_cdf(z) + normal_pdf(z))
}

fn normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * PI).sqrt()
}

/// Standard normal CDF via the Abramowitz-Stegun 7.1.26 erf approximation
/// (absolute error below 1.5·10⁻⁷).
fn normal_cdf(z: f64) -> f64 {
    let x = z / 2.0_f64.sqrt();
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf_abs = 1.0 - poly * (-x * x).exp();
    let erf = if x >= 0.0 { erf_abs } else { -erf_abs };
    0.5 * (1.0 + erf)
}

/// Standardise `ys` to zero mean and unit variance; returns `(z, mean, sd)`.
fn standardise(ys: &[f64]) -> (Vec<f64>, f64, f64) {
    let n = ys.len().max(1) as f64;
    let mean = ys.iter().sum::<f64>() / n;
    let var = ys.iter().map(|y| (y - mean) * (y - mean)).sum::<f64>() / n;
    let sd = if var > 0.0 { var.sqrt() } else { 1.0 };
    (ys.iter().map(|y| (y - mean) / sd).collect(), mean, sd)
}

/// Lower Cholesky factor of a symmetric positive-definite matrix.
fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                let d = a[i][i] - s;
                if d.is_nan() || d <= 0.0 || !d.is_finite() {
                    return None;
                }
                l[i][i] = d.sqrt();
            } else {
                l[i][j] = (a[i][j] - s) / l[j][j];
            }
        }
    }
    Some(l)
}

/// Solve `L Lᵀ x = b`.
fn solve_chol(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let y = forward_sub(l, b);
    let n = l.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| l[k][i] * x[k]).sum();
        x[i] = (y[i] - s) / l[i][i];
    }
    x
}

/// Solve `L y = b` for lower-triangular `L`.
fn forward_sub(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let s: f64 = (0..i).map(|k| l[i][k] * y[k]).sum();
        y[i] = (b[i] - s) / l[i][i];
    }
    y
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Deterministic SplitMix64 generator with Box-Muller Gaussian draws.
#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
    spare: Option<f64>,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed, spare: None }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in the open interval (0, 1).
    fn next_uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }

    fn next_gaussian(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn branin(x: &[f64]) -> f64 {
        let (x1, x2) = (x[0], x[1]);
        let a = 1.0;
        let b = 5.1 / (4.0 * PI * PI);
        let c = 5.0 / PI;
        let r = 6.0;
        let s = 10.0;
        let t = 1.0 / (8.0 * PI);
        a * (x2 - b * x1 * x1 + c * x1 - r).powi(2) + s * (1.0 - t) * x1.cos() + s
    }

    #[test]
    fn normal_cdf_reference_values() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!((normal_cdf(1.959_964) - 0.975).abs() < 1e-6);
        assert!((normal_cdf(-1.0) - 0.158_655).abs() < 1e-6);
    }

    #[test]
    fn matern_at_zero_distance_is_signal_variance() {
        let h = Hyper { lengthscale: 0.5, signal_sd: 2.0, noise_sd: 0.1 };
        let v = matern52(&[0.2, 0.7], &[0.2, 0.7], h);
        assert!((v - 4.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_bad_bounds_and_mismatched_points() {
        assert_eq!(BayesOpt::new(&[], BoConfig::default()).err(), Some(BoError::InvalidBounds));
        assert_eq!(
            BayesOpt::new(&[(1.0, 1.0)], BoConfig::default()).err(),
            Some(BoError::InvalidBounds)
        );
        let mut opt = BayesOpt::new(&[(0.0, 1.0)], BoConfig::default()).expect("valid");
        assert_eq!(opt.tell(&[0.5, 0.5], 1.0), Err(BoError::DimensionMismatch));
        assert_eq!(opt.tell(&[0.5], f64::NAN), Err(BoError::NonFiniteValue));
    }

    #[test]
    fn quadratic_minimum_is_located() {
        let r = minimise(|x| (x[0] - 0.3).powi(2), &[(0.0, 1.0)], 25, &BoConfig::default())
            .expect("valid");
        assert!((r.x[0] - 0.3).abs() < 0.02, "{r:?}");
        assert_eq!(r.evaluations, 25);
    }

    #[test]
    fn branin_reaches_near_global_minimum() {
        // Global minimum 0.397887 on [-5, 10] × [0, 15].
        let r = minimise(
            branin,
            &[(-5.0, 10.0), (0.0, 15.0)],
            60,
            &BoConfig { seed: 7, ..BoConfig::default() },
        )
        .expect("valid");
        assert!(r.f < 0.5, "best {} at {:?}", r.f, r.x);
        assert_eq!(r.history.len(), 60);
        assert!(r.history.windows(2).all(|w| w[1] <= w[0]));
    }

    #[test]
    fn same_seed_reproduces_history() {
        let cfg = BoConfig { seed: 3, ..BoConfig::default() };
        let a = minimise(branin, &[(-5.0, 10.0), (0.0, 15.0)], 20, &cfg).expect("valid");
        let b = minimise(branin, &[(-5.0, 10.0), (0.0, 15.0)], 20, &cfg).expect("valid");
        assert_eq!(a, b);
    }
}
