//! Piecewise-linear objectives via SOS2 reformulation.
//!
//! A [`PiecewiseObjective`] attaches a piecewise-linear cost curve
//! `f(x_var)` (defined by ordered breakpoints) to one model variable. The
//! solver reformulates it into an exact mixed-integer extension:
//!
//! ```text
//! λ_i >= 0,  Σ λ_i = 1,
//! x_var      = Σ λ_i · bx_i          (breakpoint abscissae)
//! f(x_var)   ≈ Σ λ_i · by_i          (breakpoint ordinates)
//! λ          ∈ SOS2                  (at most two adjacent λ non-zero)
//! ```
//!
//! With the SOS2 restriction the interpolation reproduces `f` exactly at every
//! feasible point of the branch-and-bound tree, so the MILP optimum equals the
//! true piecewise optimum. The relaxation (without SOS2) is the convex
//! envelope of `f` — tight automatically when `f` is convex and the sense is
//! minimisation.
//!
//! The variable carrying the curve is clamped to `[bx_first, bx_last]`: the
//! piecewise function is only defined on that interval.

use std::vec::Vec;

use tpt_opt_core::bounds::VarBound;
use tpt_opt_core::model::{Constraint, Model, Objective};
use tpt_opt_core::OptError;

use crate::sos::{SosSet, SosType};

/// A piecewise-linear objective term on model variable `var`.
#[derive(Debug, Clone, PartialEq)]
pub struct PiecewiseObjective {
    /// Index of the variable the curve applies to.
    pub var: usize,
    /// Breakpoints `(x, f(x))`, strictly increasing in `x`.
    pub breakpoints: Vec<(f64, f64)>,
}

impl PiecewiseObjective {
    /// Build from `(x, f(x))` pairs; they are sorted by `x`.
    ///
    /// # Errors
    /// Returns [`OptError::InvalidModel`] with fewer than two breakpoints or
    /// duplicate abscissae.
    pub fn new(var: usize, mut breakpoints: Vec<(f64, f64)>) -> Result<Self, OptError> {
        if breakpoints.len() < 2 {
            return Err(OptError::invalid_model(
                "piecewise objective needs at least two breakpoints",
            ));
        }
        breakpoints.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("NaN breakpoint"));
        for w in breakpoints.windows(2) {
            if w[0].0 >= w[1].0 {
                return Err(OptError::invalid_model("piecewise breakpoints must be distinct"));
            }
        }
        Ok(Self { var, breakpoints })
    }

    /// Evaluate the interpolant at `x` (clamped to the breakpoint range).
    pub fn eval(&self, x: f64) -> f64 {
        let bps = &self.breakpoints;
        let (x0, y0) = bps[0];
        let (xn, yn) = bps[bps.len() - 1];
        if x <= x0 {
            return y0;
        }
        if x >= xn {
            return yn;
        }
        for w in bps.windows(2) {
            let (xa, ya) = w[0];
            let (xb, yb) = w[1];
            if x <= xb {
                let t = (x - xa) / (xb - xa);
                return ya + t * (yb - ya);
            }
        }
        yn
    }

    /// Build the augmented model implementing this curve.
    ///
    /// Returns the new model (original variables keep their indices; `k+1`
    /// λ-variables are appended), the SOS2 set over the λ-variables, and the
    /// index of the first appended λ-variable.
    ///
    /// # Errors
    /// Propagates [`OptError`] if `var` is out of range.
    pub fn augment(&self, model: &Model) -> Result<(Model, SosSet, usize), OptError> {
        if self.var >= model.num_vars {
            return Err(OptError::invalid_model("piecewise variable index out of range"));
        }
        let mut aug = model.clone();
        let k = self.breakpoints.len();
        let first_lambda = aug.num_vars;
        let xs: Vec<f64> = self.breakpoints.iter().map(|&(x, _)| x).collect();
        let ys: Vec<f64> = self.breakpoints.iter().map(|&(_, y)| y).collect();

        // Clamp the carrier variable to the breakpoint domain.
        let lo = xs[0];
        let hi = xs[k - 1];
        let old = &aug.variables[self.var];
        let new_lo = old.bound.bound.lower.max(lo);
        let new_hi = old.bound.bound.upper.min(hi);
        if new_lo > new_hi + 1e-12 {
            return Err(OptError::infeasible(
                tpt_opt_core::error::InfeasibilityReport::new(
                    "variable bounds do not intersect the piecewise breakpoint range",
                )
                .with_conflict(self.var),
            ));
        }
        let kind = old.bound.kind;
        aug.variables[self.var].bound =
            VarBound { kind, bound: tpt_opt_core::bounds::Bound::boxed(new_lo, new_hi) };

        // Append λ variables in [0, 1].
        for _ in 0..k {
            aug.add_variable(VarBound::continuous(0.0, 1.0));
        }

        // Σ λ = 1.
        let all_l: Vec<usize> = (first_lambda..first_lambda + k).collect();
        let ones: Vec<f64> = vec![1.0; k];
        aug.add_constraint(Constraint::equality(all_l.clone(), ones.clone(), 1.0));

        // x_var = Σ λ_i · bx_i.
        let mut idx = vec![self.var];
        let mut coefs = vec![1.0];
        for (i, &l) in all_l.iter().enumerate() {
            idx.push(l);
            coefs.push(-xs[i]);
        }
        aug.add_constraint(Constraint::equality(idx, coefs, 0.0));

        // Extend the objective with Σ λ_i · by_i (both senses).
        let mut obj_indices = aug.objective.indices.clone();
        let mut obj_coeffs = aug.objective.coeffs.clone();
        for (i, &l) in all_l.iter().enumerate() {
            obj_indices.push(l);
            obj_coeffs.push(ys[i]);
        }
        aug.set_objective(Objective {
            sense: aug.objective.sense,
            indices: obj_indices,
            coeffs: obj_coeffs,
            constant: aug.objective.constant,
        });

        let sos = SosSet::new(
            SosType::Sos2,
            all_l.iter().enumerate().map(|(i, &v)| (v, i as f64)).collect(),
        );
        Ok((aug, sos, first_lambda))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_opt_core::Solver;

    fn base_model() -> Model {
        let mut m = Model::new(1);
        m.variables[0].bound = VarBound::continuous(0.0, 10.0);
        m
    }

    #[test]
    fn eval_interpolates() {
        let pw = PiecewiseObjective::new(0, vec![(0.0, 0.0), (2.0, 4.0), (5.0, 6.0)]).unwrap();
        assert!((pw.eval(1.0) - 2.0).abs() < 1e-12);
        assert!((pw.eval(3.5) - 5.0).abs() < 1e-12);
        assert!((pw.eval(-1.0) - 0.0).abs() < 1e-12); // clamped left
        assert!((pw.eval(7.0) - 6.0).abs() < 1e-12); // clamped right
    }

    #[test]
    fn augment_adds_lambdas_and_rows() {
        let pw = PiecewiseObjective::new(0, vec![(0.0, 0.0), (2.0, 4.0), (5.0, 6.0)]).unwrap();
        let m = base_model();
        let (aug, sos, first) = pw.augment(&m).unwrap();
        assert_eq!(aug.num_vars, 4); // 1 original + 3 lambdas
        assert_eq!(first, 1);
        assert_eq!(sos.vars.len(), 3);
        assert_eq!(aug.constraints.len(), 2); // Σλ=1 and linking row
                                              // Carrier clamped to [0, 5].
        assert_eq!(aug.variables[0].bound.bound.upper, 5.0);
    }

    #[test]
    fn rejects_bad_breakpoints() {
        assert!(PiecewiseObjective::new(0, vec![(0.0, 0.0)]).is_err());
        assert!(PiecewiseObjective::new(0, vec![(1.0, 0.0), (1.0, 2.0)]).is_err());
    }

    #[test]
    fn end_to_end_convex_curve() {
        // minimise f(x) = |x - 2| over x ∈ [0, 4]; optimum x=2, f=0.
        let m = base_model();
        let pw = PiecewiseObjective::new(0, vec![(0.0, 2.0), (2.0, 0.0), (4.0, 2.0)]).unwrap();
        let (aug, sos, _) = pw.augment(&m).unwrap();

        let mut solver = crate::MilpSolver::new();
        solver.add_sos(sos);
        let sol = solver.solve(&aug).unwrap();
        assert_eq!(sol.status, tpt_opt_core::solver::SolverStatus::Optimal);
        assert!(sol.objective_value.abs() < 1e-6, "obj={}", sol.objective_value);
        assert!((sol.primal_value(0).unwrap() - 2.0).abs() < 1e-4);
    }
}
