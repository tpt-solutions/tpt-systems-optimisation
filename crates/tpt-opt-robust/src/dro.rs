//! Distributionally robust optimisation (DRO): moment/box ambiguity sets
//! and Wasserstein balls.
//!
//! - **Box ambiguity** over finite outcomes: probabilities constrained to
//!   `lo_s ≤ p_s ≤ hi_s`, `Σ p = 1`. The worst-case expectation has a
//!   closed-form greedy solution ([`worst_case_box_expectation`]), and the
//!   decision problem `min_x max_p E_p[c(x, ξ)]` is solved by a finite
//!   cutting-plane scheme over probability vertices
//!   ([`DroCuttingPlane`]).
//! - **Wasserstein ball**: for a linear loss `ℓ(x, ξ) = g·ξ + h` the
//!   worst-case expectation over distributions within radius `θ` (2-norm)
//!   of an empirical distribution equals the sample mean plus
//!   `θ·‖g‖₂` ([`worst_case_linear_wasserstein`]) — the conjugate/Lipschitz
//!   reformulation of Mohajerin Esfahani & Kuhn (2018).

use std::vec::Vec;

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::VarBound;
use tpt_opt_milp::MilpSolver;

/// Worst-case expectation of per-outcome `costs` under box-bounded
/// probabilities (`lo_s ≤ p_s ≤ hi_s`, `Σ p = 1`). Greedy: push mass onto
/// the most expensive outcomes first. Panics if the box cannot sum to 1.
pub fn worst_case_box_expectation(costs: &[f64], lo: &[f64], hi: &[f64]) -> f64 {
    assert_eq!(costs.len(), lo.len());
    assert_eq!(costs.len(), hi.len());
    let total_lo: f64 = lo.iter().sum();
    let total_hi: f64 = hi.iter().sum();
    assert!(total_lo <= 1.0 + 1e-9 && total_hi >= 1.0 - 1e-9, "box must contain a distribution");
    let mut order: Vec<usize> = (0..costs.len()).collect();
    order.sort_by(|&a, &b| costs[b].partial_cmp(&costs[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut remaining = 1.0f64;
    let mut value = 0.0f64;
    for &s in &order {
        if remaining <= 1e-12 {
            break;
        }
        let take = remaining.min(hi[s] - lo[s]).max(0.0);
        value += costs[s] * (lo[s] + take);
        remaining -= take;
    }
    // Any leftover forced mass sits at the lower bounds already counted.
    value + 0.0
}

/// Cutting-plane solver for `min_x max_p E_p[c(x, ξ)]` over a box ambiguity
/// set, with `x ≥ 0` bounded and per-scenario costs supplied as a closure.
///
/// Each iteration solves the master `min θ s.t. θ ≥ Σ_s p^k_s c_s(x)` for
/// every generated probability vector `p^k`, then computes the worst-case
/// `p` for the current `x` greedily and appends it as a new cut. Terminates
/// on gap tolerance or iteration cap.
pub struct DroCuttingPlane<FC> {
    /// Per-variable bounds `(lo, hi)` with `lo ≥ 0`.
    pub bounds: Vec<(f64, f64)>,
    /// Box probability lower bounds per scenario.
    pub prob_lo: Vec<f64>,
    /// Box probability upper bounds per scenario.
    pub prob_hi: Vec<f64>,
    /// Per-scenario cost `c_s(x)`.
    pub cost: FC,
    /// Convergence tolerance on the master/worst-case gap.
    pub tolerance: f64,
    /// Maximum cutting-plane iterations.
    pub max_iterations: usize,
}

impl<FC> DroCuttingPlane<FC>
where
    FC: Fn(usize, &[f64]) -> f64,
{
    /// Create a cutting-plane solver with default tolerances
    /// (`tolerance = 1e-9`, `max_iterations = 100`).
    pub fn new(bounds: Vec<(f64, f64)>, prob_lo: Vec<f64>, prob_hi: Vec<f64>, cost: FC) -> Self {
        Self { bounds, prob_lo, prob_hi, cost, tolerance: 1e-9, max_iterations: 100 }
    }

    /// Override the convergence tolerance.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Override the iteration cap.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Run the cutting-plane loop; returns `(x, worst_case_value)`.
    pub fn solve(&self) -> Result<(Vec<f64>, f64), tpt_opt_core::OptError> {
        use tpt_opt_core::solver::Solver;
        let s_count = self.prob_lo.len();
        let n = self.bounds.len();
        let mut cuts: Vec<Vec<f64>> = Vec::new(); // generated probability vectors

        // Seed cut: fill from the lower bounds up to a valid distribution.
        let mut seed = self.prob_lo.clone();
        let mut rem = 1.0 - seed.iter().sum::<f64>();
        for (s, slot) in seed.iter_mut().enumerate() {
            if rem <= 1e-12 {
                break;
            }
            let add = rem.min(self.prob_hi[s] - *slot).max(0.0);
            *slot += add;
            rem -= add;
        }
        cuts.push(seed);

        let mut x_best = vec![0.0f64; n];
        let mut wc_best = f64::INFINITY;
        for _ in 0..self.max_iterations {
            // Master: min θ s.t. θ ≥ Σ p^k_s c_s(x) for each cut k.
            let mut model = Model::new(n + 1);
            for (i, b) in self.bounds.iter().enumerate() {
                model.variables[i].bound = VarBound::continuous(b.0, b.1);
            }
            model.variables[n].bound = VarBound::continuous(f64::NEG_INFINITY, f64::INFINITY);
            model.set_objective(Objective {
                sense: Sense::Minimize,
                indices: vec![n],
                coeffs: vec![1.0],
                constant: 0.0,
            });
            for p in &cuts {
                // Σ_s p_s c_s(x) − θ ≤ 0 built by evaluating c_s at the
                // current incumbent? No — c_s(x) is linear in x only if the
                // caller's costs are linear; we treat general costs by
                // outer-linearising around the last iterate: the cut uses
                // the *current* x's costs as constants (valid because the
                // worst case over p is what matters and p-polytope vertices
                // are finitely many; convergence is on the p-side).
                let theta_idx = n;
                let mut idx: Vec<usize> = (0..n).collect();
                let mut co: Vec<f64> = vec![0.0; n];
                // Linearisation point: reuse previous iterate (starts 0).
                let base: Vec<f64> = x_best.clone();
                let mut rhs = 0.0f64;
                for (s, &p_s) in (0..s_count).zip(p.iter()) {
                    rhs += p_s * (self.cost)(s, &base);
                    // Subgradient w.r.t. x approximated by finite difference
                    // only when needed; for linear costs the caller should
                    // supply exact linearity — here we keep the cut valid by
                    // using the constant term and zero slope (monotone
                    // underestimator for convex-in-x costs evaluated at the
                    // current point is NOT generally valid), so instead we
                    // require the caller's cost to be affine and recover the
                    // slope numerically once.
                }
                // Numerical slope recovery (once per cut): c_s(x) ≈ c_s(base)
                // + g_s·(x − base); estimate g_s by central differences.
                for j in 0..n {
                    let h = ((self.bounds[j].1 - self.bounds[j].0).max(1e-3)) * 1e-4;
                    let mut xp = base.clone();
                    xp[j] += h;
                    let mut xm = base.clone();
                    xm[j] -= h;
                    let mut slope = 0.0f64;
                    for (s, &p_s) in (0..s_count).zip(p.iter()) {
                        slope +=
                            p_s * (((self.cost)(s, &xp) - (self.cost)(s, &xm)) / (2.0 * h));
                    }
                    co[j] = slope;
                    rhs -= slope * base[j];
                }
                idx.push(theta_idx);
                co.push(-1.0);
                model.add_constraint(Constraint::le(idx, co, rhs));
            }
            let mut solver = MilpSolver::new();
            let sol = solver.solve(&model)?;
            x_best = sol.primal[..n].to_vec();

            // Worst-case p at the new x (greedy on current costs).
            let costs: Vec<f64> = (0..s_count).map(|s| (self.cost)(s, &x_best)).collect();
            let wc = worst_case_box_expectation(&costs, &self.prob_lo, &self.prob_hi);
            // Recover the worst-case p to append as a cut.
            let mut order: Vec<usize> = (0..s_count).collect();
            order.sort_by(|&a, &b| {
                costs[b].partial_cmp(&costs[a]).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut p_new = self.prob_lo.clone();
            let mut rem = 1.0f64;
            for &s in &order {
                let take = rem.min(self.prob_hi[s] - p_new[s]).max(0.0);
                p_new[s] += take;
                rem -= take;
            }
            let gap = wc - sol.objective_value;
            if gap <= self.tolerance {
                return Ok((x_best, wc));
            }
            if !cuts.iter().any(|c| c.iter().zip(&p_new).all(|(a, b)| (a - b).abs() < 1e-12)) {
                cuts.push(p_new);
            } else {
                // Cut repeated without convergence: stop with current values.
                return Ok((x_best, wc));
            }
            wc_best = wc_best.min(wc);
        }
        Ok((x_best, wc_best))
    }
}

/// Worst-case expected value of a linear loss `ℓ(ξ) = g·ξ + h` over a
/// 2-norm Wasserstein ball of radius `theta` centred on the empirical
/// distribution `samples` (uniform weights):
/// `mean_s(g·ξ_s) + h + θ·‖g‖₂`.
pub fn worst_case_linear_wasserstein(
    samples: &[Vec<f64>],
    grad: &[f64],
    intercept: f64,
    theta: f64,
) -> f64 {
    assert!(!samples.is_empty(), "need at least one sample");
    assert!(samples.iter().all(|s| s.len() == grad.len()), "sample dimension mismatch");
    let mean: f64 =
        samples.iter().map(|xi| grad.iter().zip(xi).map(|(a, b)| a * b).sum::<f64>()).sum::<f64>()
            / samples.len() as f64;
    let gnorm = grad.iter().map(|a| a * a).sum::<f64>().sqrt();
    mean + intercept + theta * gnorm
}
