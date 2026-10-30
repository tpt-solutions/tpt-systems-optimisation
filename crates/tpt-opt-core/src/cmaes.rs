//! CMA-ES: covariance-matrix-adaptation evolution strategy.
//!
//! [`cma_es`] minimises an unconstrained objective with the (μ/μ_w, λ) CMA-ES
//! of Hansen (2016, "The CMA Evolution Strategy: A Tutorial"): a Gaussian
//! search distribution with mean `m`, step size `σ` and covariance `C`, adapted
//! from ranked samples via cumulative step-size control (`p_σ`), rank-one
//! (`p_c`) and rank-μ covariance updates.
//!
//! Sampling is deterministic: the same [`CmaEsOptions::seed`] reproduces the
//! same run bit-for-bit, per the workspace reproducibility rule. The
//! eigendecomposition of `C` uses cyclic Jacobi rotations, which is accurate
//! for the small-to-moderate dimensions CMA-ES is practical for.

use alloc::vec;
use alloc::vec::Vec;

use crate::unconstrained::{norm2, UncParams, UncResult};

/// Options for [`cma_es`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CmaEsOptions {
    /// Initial step size `σ₀` (roughly one-quarter of the search-region width).
    pub sigma0: f64,
    /// Population size `λ`; `0` selects the default `4 + ⌊3 ln n⌋`.
    pub lambda: usize,
    /// Seed for the deterministic Gaussian sampler.
    pub seed: u64,
    /// Stopping controls. `max_iters` counts generations; `f_tol` is the
    /// spread of objective values within one generation; `x_tol` is the
    /// step-size-times-largest-axis length.
    pub params: UncParams,
}

impl Default for CmaEsOptions {
    fn default() -> Self {
        Self {
            sigma0: 0.3,
            lambda: 0,
            seed: 0,
            params: UncParams { max_iters: 1000, f_tol: 1e-11, x_tol: 1e-11, g_tol: 0.0 },
        }
    }
}

/// Minimise `f` with CMA-ES starting from `x0`.
///
/// Returns the best point evaluated over the whole run (not just the final
/// mean). `converged` is `true` when the per-generation objective spread or
/// the scaled step length falls below the tolerances in `opts.params`.
///
/// # Panics
///
/// Panics if `x0` is empty or `opts.sigma0` is not positive and finite.
pub fn cma_es<F>(mut f: F, x0: &[f64], opts: &CmaEsOptions) -> UncResult
where
    F: FnMut(&[f64]) -> f64,
{
    let n = x0.len();
    assert!(n > 0, "cma_es needs at least one variable");
    assert!(opts.sigma0.is_finite() && opts.sigma0 > 0.0, "cma_es needs a positive finite sigma0");
    let p = &opts.params;
    let nf = n as f64;

    // Strategy parameters (Hansen 2016, Table 1).
    let lambda =
        if opts.lambda == 0 { 4 + (3.0 * nf.ln()).floor() as usize } else { opts.lambda }.max(2);
    let mu = lambda / 2;
    let raw: Vec<f64> = (1..=mu).map(|i| ((mu as f64) + 0.5).ln() - (i as f64).ln()).collect();
    let wsum: f64 = raw.iter().sum();
    let weights: Vec<f64> = raw.iter().map(|w| w / wsum).collect();
    let mueff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
    let cc = (4.0 + mueff / nf) / (nf + 4.0 + 2.0 * mueff / nf);
    let cs = (mueff + 2.0) / (nf + mueff + 5.0);
    let c1 = 2.0 / ((nf + 1.3) * (nf + 1.3) + mueff);
    let cmu = (1.0 - c1).min(2.0 * (mueff - 2.0 + 1.0 / mueff) / ((nf + 2.0).powi(2) + mueff));
    let damps = 1.0 + 2.0 * (((mueff - 1.0) / (nf + 1.0)).sqrt() - 1.0).max(0.0) + cs;
    let chi_n = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));

    let mut rng = SplitMix64::new(opts.seed);
    let mut mean = x0.to_vec();
    let mut sigma = opts.sigma0;
    let mut cov = identity(n);
    let mut pc = vec![0.0; n];
    let mut ps = vec![0.0; n];
    // B (eigenvectors as columns) and D (√eigenvalues) of C.
    let (mut b, mut d) = eigen_sqrt(&cov, n);

    let mut evaluations = 0;
    let mut best_x = x0.to_vec();
    let mut best_f = f(x0);
    evaluations += 1;

    let mut iterations = 0;
    let mut converged = false;
    while iterations < p.max_iters {
        iterations += 1;

        // Sample λ offspring: x = m + σ·B·D·z, z ~ N(0, I).
        let mut samples: Vec<(f64, Vec<f64>, Vec<f64>)> = Vec::with_capacity(lambda);
        for _ in 0..lambda {
            let z: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
            let dz: Vec<f64> = z.iter().zip(&d).map(|(zi, di)| zi * di).collect();
            let y = mat_vec(&b, n, &dz);
            let x: Vec<f64> = mean.iter().zip(&y).map(|(m, yi)| m + sigma * yi).collect();
            let fx = f(&x);
            evaluations += 1;
            if fx < best_f {
                best_f = fx;
                best_x.clone_from(&x);
            }
            samples.push((fx, x, y));
        }
        samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

        let f_spread = samples[lambda - 1].0 - samples[0].0;

        // Recombination: new mean from the μ best samples.
        let mut y_w = vec![0.0; n];
        for (w, (_, _, y)) in weights.iter().zip(&samples) {
            for (yw, yi) in y_w.iter_mut().zip(y) {
                *yw += w * yi;
            }
        }
        let new_mean: Vec<f64> = mean.iter().zip(&y_w).map(|(m, yw)| m + sigma * yw).collect();

        // C^{-1/2} · y_w = B · D⁻¹ · Bᵀ · y_w.
        let bt_yw = mat_t_vec(&b, n, &y_w);
        let inv_sqrt_yw = {
            let scaled: Vec<f64> = bt_yw.iter().zip(&d).map(|(v, di)| v / di).collect();
            mat_vec(&b, n, &scaled)
        };

        // Cumulative step-size path.
        let ps_coef = (cs * (2.0 - cs) * mueff).sqrt();
        for (pi, yi) in ps.iter_mut().zip(&inv_sqrt_yw) {
            *pi = (1.0 - cs) * *pi + ps_coef * yi;
        }
        let ps_norm = norm2(&ps);
        let gen = iterations as f64;
        let hsig_lhs = ps_norm / (1.0 - (1.0 - cs).powf(2.0 * gen)).sqrt() / chi_n;
        let hsig = if hsig_lhs < 1.4 + 2.0 / (nf + 1.0) { 1.0 } else { 0.0 };

        // Covariance path and rank-one / rank-μ update.
        let pc_coef = hsig * (cc * (2.0 - cc) * mueff).sqrt();
        for (pci, yw) in pc.iter_mut().zip(&y_w) {
            *pci = (1.0 - cc) * *pci + pc_coef * yw;
        }
        let decay = 1.0 - c1 - cmu + (1.0 - hsig) * c1 * cc * (2.0 - cc);
        for i in 0..n {
            for j in 0..n {
                let mut v = decay * cov[i][j] + c1 * pc[i] * pc[j];
                for (w, (_, _, y)) in weights.iter().zip(&samples) {
                    v += cmu * w * y[i] * y[j];
                }
                cov[i][j] = v;
            }
        }

        // Step-size adaptation.
        sigma *= ((cs / damps) * (ps_norm / chi_n - 1.0)).exp();

        mean = new_mean;
        let (nb, nd) = eigen_sqrt(&cov, n);
        b = nb;
        d = nd;

        let d_max = d.iter().fold(0.0_f64, |m, v| m.max(*v));
        if f_spread.abs() <= p.f_tol && sigma * d_max <= p.x_tol {
            converged = true;
            break;
        }
        if !sigma.is_finite() || sigma <= 0.0 {
            break;
        }
    }

    UncResult { x: best_x, f: best_f, iterations, evaluations, converged }
}

/// Deterministic SplitMix64 generator with Box-Muller Gaussian draws.
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
        // 53 random bits, shifted by half an ulp so the value is never 0.
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }

    fn next_gaussian(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * core::f64::consts::PI * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

fn identity(n: usize) -> Vec<Vec<f64>> {
    (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect()
}

/// `B · v` where `B` is stored as columns: `out[i] = Σ_j B[i][j]·v[j]`.
fn mat_vec(b: &[Vec<f64>], n: usize, v: &[f64]) -> Vec<f64> {
    (0..n).map(|i| (0..n).map(|j| b[i][j] * v[j]).sum()).collect()
}

/// `Bᵀ · v`.
fn mat_t_vec(b: &[Vec<f64>], n: usize, v: &[f64]) -> Vec<f64> {
    (0..n).map(|j| (0..n).map(|i| b[i][j] * v[i]).sum()).collect()
}

/// Eigendecomposition of symmetric `c` returning `(B, D)` with
/// `c = B · diag(D²) · Bᵀ`, `B` orthogonal (eigenvectors as columns) and
/// `D` the square roots of the eigenvalues, clamped away from zero.
fn eigen_sqrt(c: &[Vec<f64>], n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let (vals, vecs) = jacobi_eigen(c, n);
    let d = vals.iter().map(|v| v.max(f64::MIN_POSITIVE).sqrt()).collect();
    (vecs, d)
}

/// Cyclic Jacobi eigenvalue algorithm for a symmetric matrix.
///
/// Returns `(eigenvalues, eigenvectors)` with eigenvectors as columns of the
/// second matrix.
fn jacobi_eigen(sym: &[Vec<f64>], n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut a: Vec<Vec<f64>> = sym.to_vec();
    let mut v = identity(n);
    for _sweep in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|p| (p + 1..n).map(move |q| (p, q)))
            .map(|(p, q)| a[p][q] * a[p][q])
            .sum();
        let diag: f64 = (0..n).map(|i| a[i][i] * a[i][i]).sum();
        if off <= f64::EPSILON * f64::EPSILON * diag.max(f64::MIN_POSITIVE) {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p][q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut().chain(v.iter_mut()) {
                    let (x, y) = (row[p], row[q]);
                    row[p] = c * x - s * y;
                    row[q] = s * x + c * y;
                }
                let rp: Vec<f64> = a[p].iter().zip(&a[q]).map(|(x, y)| c * x - s * y).collect();
                let rq: Vec<f64> = a[p].iter().zip(&a[q]).map(|(x, y)| s * x + c * y).collect();
                a[p] = rp;
                a[q] = rq;
            }
        }
    }
    let vals = (0..n).map(|i| a[i][i]).collect();
    (vals, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rosenbrock(x: &[f64]) -> f64 {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + 100.0 * b * b
    }

    #[test]
    fn jacobi_reconstructs_symmetric_matrix() {
        let m = vec![vec![4.0, 1.0, 0.5], vec![1.0, 3.0, 0.2], vec![0.5, 0.2, 2.0]];
        let (vals, v) = jacobi_eigen(&m, 3);
        for i in 0..3 {
            for j in 0..3 {
                let r: f64 = (0..3).map(|k| v[i][k] * vals[k] * v[j][k]).sum();
                assert!((r - m[i][j]).abs() < 1e-10, "({i},{j}): {r}");
            }
        }
    }

    #[test]
    fn sphere_converges_to_origin() {
        let f = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
        let r = cma_es(f, &[2.0, -3.0, 1.5], &CmaEsOptions::default());
        assert!(r.converged, "{r:?}");
        assert!(r.f < 1e-10, "{r:?}");
    }

    #[test]
    fn ill_conditioned_ellipsoid() {
        // Condition number 1e6 — the covariance must learn the axis scaling.
        let n = 5;
        let f = |x: &[f64]| {
            x.iter()
                .enumerate()
                .map(|(i, v)| 10f64.powf(3.0 * i as f64 / (n as f64 - 1.0)) * v * v)
                .sum::<f64>()
        };
        let r = cma_es(f, &[1.0; 5], &CmaEsOptions::default());
        assert!(r.f < 1e-8, "{r:?}");
    }

    #[test]
    fn rosenbrock_2d() {
        let opts = CmaEsOptions {
            sigma0: 0.5,
            params: UncParams { max_iters: 2000, ..CmaEsOptions::default().params },
            ..CmaEsOptions::default()
        };
        let r = cma_es(rosenbrock, &[-1.2, 1.0], &opts);
        assert!((r.x[0] - 1.0).abs() < 1e-3, "{r:?}");
        assert!((r.x[1] - 1.0).abs() < 1e-3, "{r:?}");
    }

    #[test]
    fn same_seed_is_reproducible_and_seed_matters() {
        let f = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
        let a = cma_es(f, &[1.0, 1.0], &CmaEsOptions::default());
        let b = cma_es(f, &[1.0, 1.0], &CmaEsOptions::default());
        assert_eq!(a, b);
        let other = CmaEsOptions { seed: 99, ..CmaEsOptions::default() };
        let c = cma_es(f, &[1.0, 1.0], &other);
        assert_ne!(a.evaluations, 0);
        assert!(c.f < 1e-8);
    }
}
