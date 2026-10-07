//! Problem representation: residual blocks, builder, and evaluation.
//!
//! A least-squares problem is a set of residual blocks `r_k: Rⁿ → R^{m_k}`;
//! the total residual is their concatenation and the cost is
//! `½ Σ ρ(rᵢ(x)²)`. Blocks are written once as closures over
//! [`dual::Dual`](crate::dual) and evaluated in plain mode (values) or
//! seeded mode (values + exact Jacobian rows) — see [`JacobianMode`].
//!
//! # Example
//!
//! ```rust
//! use tpt_opt_nls::problem::NlsProblem;
//!
//! // Rosenbrock residuals r = (1 − x, 10(y − x²)).
//! let problem = NlsProblem::builder(2)
//!     .residual_block(2, |x, out| {
//!         out[0] = 1.0 - x[0].clone();
//!         out[1] = 10.0 * (x[1].clone() - x[0].clone() * x[0].clone());
//!     })
//!     .build();
//! assert_eq!(problem.n_residuals(), 2);
//! ```

use crate::dual::{self, Dual};
use crate::types::NlsError;

/// Boxed residual block: writes `m` outputs from `n` inputs.
pub type ResidualFn = Box<dyn Fn(&[Dual], &mut [Dual])>;

/// How the Jacobian is obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JacobianMode {
    /// Forward-mode autodiff (exact, one seeded evaluation per block).
    /// The default.
    Autodiff,
    /// Central finite differences (fallback for closures that cannot be
    /// written generically).
    FiniteDifference,
}

/// How the normal-equation system is solved each iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolver {
    /// Dense Cholesky on `JᵀJ` (default).
    Dense,
    /// Sparse up-looking LDLᵀ with a fill-reducing minimum-degree ordering
    /// (worthwhile when `JᵀJ` is large and structurally sparse).
    Sparse,
}

/// A nonlinear least-squares problem.
pub struct NlsProblem {
    /// Number of parameters.
    pub n: usize,
    blocks: Vec<(usize, ResidualFn)>,
    bounds: Vec<(f64, f64)>,
    m: usize,
}

impl NlsProblem {
    /// Start building a problem over `n` parameters.
    pub fn builder(n: usize) -> NlsProblemBuilder {
        NlsProblemBuilder {
            n,
            blocks: Vec::new(),
            bounds: vec![(f64::NEG_INFINITY, f64::INFINITY); n],
        }
    }

    /// Number of parameters.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Total number of scalar residuals.
    pub fn n_residuals(&self) -> usize {
        self.m
    }

    /// Parameter bounds as `(lower, upper)` pairs.
    pub fn bounds(&self) -> &[(f64, f64)] {
        &self.bounds
    }

    /// Clamp a parameter vector into the declared bounds.
    pub fn project(&self, x: &mut [f64]) {
        for (i, v) in x.iter_mut().enumerate() {
            let (lo, hi) = self.bounds[i];
            if *v < lo {
                *v = lo;
            }
            if *v > hi {
                *v = hi;
            }
        }
    }

    /// `true` if any parameter has a finite bound.
    pub fn is_bounded(&self) -> bool {
        self.bounds.iter().any(|&(lo, hi)| lo.is_finite() || hi.is_finite())
    }

    /// Evaluate the full residual vector at `x` (plain mode; values only).
    pub fn residual(&self, x: &[f64], out: &mut [f64]) {
        let mut off = 0;
        for (m_blk, f) in &self.blocks {
            dual::eval_plain(f.as_ref(), x, &mut out[off..off + m_blk]);
            off += m_blk;
        }
    }

    /// Evaluate the residual vector and the row-major Jacobian (`m × n`) at
    /// `x`, per [`JacobianMode`].
    pub fn residual_and_jacobian(
        &self,
        x: &[f64],
        mode: JacobianMode,
        out: &mut [f64],
        jac: &mut [f64],
    ) {
        match mode {
            JacobianMode::Autodiff => {
                let mut off = 0;
                for (m_blk, f) in &self.blocks {
                    dual::eval_jacobian(
                        f.as_ref(),
                        x,
                        &mut out[off..off + m_blk],
                        &mut jac[off * self.n..(off + m_blk) * self.n],
                    );
                    off += m_blk;
                }
            }
            JacobianMode::FiniteDifference => {
                self.residual(x, out);
                let n = self.n;
                let mut rp = vec![0.0f64; self.m];
                let mut rm = vec![0.0f64; self.m];
                let mut xs = x.to_vec();
                for j in 0..n {
                    let h = 1e-7 * (1.0 + x[j].abs());
                    let xj = xs[j];
                    xs[j] = xj + h;
                    self.residual(&xs, &mut rp);
                    xs[j] = xj - h;
                    self.residual(&xs, &mut rm);
                    xs[j] = xj;
                    for i in 0..self.m {
                        jac[i * n + j] = (rp[i] - rm[i]) / (2.0 * h);
                    }
                }
            }
        }
    }

    /// Raw cost `½‖r‖²` (no loss reweighting) at `x`.
    pub fn cost(&self, x: &[f64]) -> f64 {
        let mut r = vec![0.0f64; self.m];
        self.residual(x, &mut r);
        0.5 * r.iter().map(|&v| v * v).sum::<f64>()
    }

    /// Robust cost `½ Σ ρ(rᵢ(x)²)` under `loss` at `x`.
    pub fn loss_cost(&self, x: &[f64], loss: crate::loss::Loss) -> f64 {
        let mut r = vec![0.0f64; self.m];
        self.residual(x, &mut r);
        0.5 * r.iter().map(|&ri| loss.rho(ri * ri)).sum::<f64>()
    }

    /// Validate internal consistency (block dims, bounds) — called by the
    /// builder, exposed for problems assembled manually.
    pub fn validate(&self) -> Result<(), NlsError> {
        if self.n == 0 {
            return Err(NlsError::InvalidProblem("problem has zero parameters".into()));
        }
        for &(lo, hi) in &self.bounds {
            if lo > hi {
                return Err(NlsError::InvalidProblem(
                    "parameter lower bound exceeds upper bound".into(),
                ));
            }
        }
        Ok(())
    }
}

impl std::fmt::Debug for NlsProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NlsProblem")
            .field("n", &self.n)
            .field("m", &self.m)
            .field("blocks", &self.blocks.len())
            .finish()
    }
}

/// Builder for [`NlsProblem`].
pub struct NlsProblemBuilder {
    n: usize,
    blocks: Vec<(usize, ResidualFn)>,
    bounds: Vec<(f64, f64)>,
}

impl NlsProblemBuilder {
    /// Append a residual block with `m` outputs. The closure receives the
    /// parameter vector as dual numbers and must write all `m` outputs.
    pub fn residual_block(mut self, m: usize, f: impl Fn(&[Dual], &mut [Dual]) + 'static) -> Self {
        self.blocks.push((m, Box::new(f)));
        self
    }

    /// Set per-parameter bounds (lower, upper). Unset parameters stay free.
    pub fn with_bounds(mut self, bounds: Vec<(f64, f64)>) -> Self {
        assert_eq!(bounds.len(), self.n, "bounds length must equal parameter count");
        self.bounds = bounds;
        self
    }

    /// Set one parameter's bounds.
    pub fn with_bound(mut self, i: usize, lo: f64, hi: f64) -> Self {
        assert!(i < self.n, "parameter index out of range");
        self.bounds[i] = (lo, hi);
        self
    }

    /// Finish the problem.
    pub fn build(self) -> NlsProblem {
        let m = self.blocks.iter().map(|(m_blk, _)| *m_blk).sum();
        NlsProblem { n: self.n, blocks: self.blocks, bounds: self.bounds, m }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_accumulates_blocks() {
        let p = NlsProblem::builder(3)
            .residual_block(2, |_x, _out| {})
            .residual_block(1, |_x, _out| {})
            .build();
        assert_eq!(p.n(), 3);
        assert_eq!(p.n_residuals(), 3);
    }

    #[test]
    fn projection_clamps() {
        let p = NlsProblem::builder(2)
            .with_bound(0, 0.0, 1.0)
            .with_bounds(vec![(0.0, 1.0), (-5.0, 5.0)])
            .build();
        assert!(p.is_bounded());
        let mut x = vec![-1.0, 7.0];
        p.project(&mut x);
        assert_eq!(x, vec![0.0, 5.0]);
    }

    #[test]
    fn fd_and_autodiff_jacobians_agree() {
        let p = NlsProblem::builder(2)
            .residual_block(2, |x, out| {
                out[0] = x[0].exp() * x[1].clone();
                out[1] = x[0].clone() * x[1].clone() + x[0].sin();
            })
            .build();
        let x = [0.4, -1.2];
        let mut r1 = vec![0.0; 2];
        let mut j1 = vec![0.0; 4];
        p.residual_and_jacobian(&x, JacobianMode::Autodiff, &mut r1, &mut j1);
        let mut r2 = vec![0.0; 2];
        let mut j2 = vec![0.0; 4];
        p.residual_and_jacobian(&x, JacobianMode::FiniteDifference, &mut r2, &mut j2);
        for k in 0..2 {
            assert!((r1[k] - r2[k]).abs() < 1e-12);
        }
        for k in 0..4 {
            assert!((j1[k] - j2[k]).abs() < 1e-5, "jac[{k}] {} vs {}", j1[k], j2[k]);
        }
    }

    #[test]
    fn validate_rejects_crossed_bounds() {
        let p = NlsProblem::builder(1).with_bound(0, 1.0, 0.0).build();
        assert!(p.validate().is_err());
    }
}
