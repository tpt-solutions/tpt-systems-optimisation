//! Mixed-integer nonlinear program representation.
//!
//! A [`MinlpModel`] stores per-variable domains ([`VarKind`]), a
//! minimisation objective and a list of nonlinear constraints expressed as
//! `g(x) <= 0` ([`ConstraintKind::Le`]) or `h(x) = 0`
//! ([`ConstraintKind::Eq`]). Objective and constraint functions are boxed
//! closures; gradients are optional and default to central finite
//! differences ([`finite_diff_grad`]).
//!
//! Constraints may carry an **indicator** (`active_if`): the consequent is
//! enforced only while the named binary variable equals the given value, the
//! nonlinear analogue of MILP indicator constraints.

use std::vec::Vec;

/// Domain of one variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarKind {
    /// Continuous on `[lb, ub]`.
    Continuous,
    /// Integer on `[lb, ub]`.
    Integer,
    /// Binary (`{0, 1}`).
    Binary,
}

/// Sense of a nonlinear constraint row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    /// `g(x) <= 0`.
    Le,
    /// `h(x) = 0`.
    Eq,
}

/// Boxed scalar function of the decision vector.
pub type ScalarFn = Box<dyn Fn(&[f64]) -> f64>;
/// Boxed gradient routine: writes ∇f into the caller's buffer.
pub type GradFn = Box<dyn Fn(&[f64], &mut [f64])>;

/// A nonlinear constraint `g(x) rel 0`, optionally gated behind a binary
/// indicator variable.
pub struct NlConstraint {
    /// Row sense.
    pub kind: ConstraintKind,
    /// Constraint body: `Le` means `f(x) <= 0`, `Eq` means `f(x) = 0`.
    pub f: ScalarFn,
    /// Optional gradient of `f`. Defaults to finite differences.
    pub grad: Option<GradFn>,
    /// When set, the constraint is enforced only if variable `.0` (a binary)
    /// equals `.1`.
    pub active_if: Option<(usize, bool)>,
}

impl std::fmt::Debug for NlConstraint {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("NlConstraint")
            .field("kind", &self.kind)
            .field("active_if", &self.active_if)
            .finish_non_exhaustive()
    }
}

/// Mixed-integer nonlinear program (minimisation).
///
/// Maximisation problems are expressed by negating the objective.
pub struct MinlpModel {
    /// Variable domains, indexed by variable handle.
    pub vars: Vec<VarKind>,
    /// Variable lower bounds.
    pub lbs: Vec<f64>,
    /// Variable upper bounds.
    pub ubs: Vec<f64>,
    /// Objective `f(x)` (minimised).
    pub objective: ScalarFn,
    /// Optional objective gradient. Defaults to finite differences.
    pub objective_grad: Option<GradFn>,
    /// Nonlinear constraints.
    pub constraints: Vec<NlConstraint>,
}

impl std::fmt::Debug for MinlpModel {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("MinlpModel").field("num_vars", &self.vars.len()).finish_non_exhaustive()
    }
}

impl MinlpModel {
    /// Create a model with `n` continuous variables on `[0, ub]` and the
    /// given (minimised) objective.
    pub fn new(n: usize, objective: impl Fn(&[f64]) -> f64 + 'static) -> Self {
        Self {
            vars: vec![VarKind::Continuous; n],
            lbs: vec![0.0; n],
            ubs: vec![if n > 0 { 1.0 } else { 0.0 }; n],
            objective: Box::new(objective),
            objective_grad: None,
            constraints: Vec::new(),
        }
    }

    /// Number of variables.
    pub fn num_vars(&self) -> usize {
        self.vars.len()
    }

    /// Set the domain of variable `i`.
    pub fn set_var(&mut self, i: usize, kind: VarKind, lb: f64, ub: f64) {
        self.vars[i] = kind;
        self.lbs[i] = lb;
        self.ubs[i] = ub;
    }

    /// Add a `g(x) <= 0` constraint; returns its index.
    pub fn add_le(
        &mut self,
        f: impl Fn(&[f64]) -> f64 + 'static,
        grad: impl Fn(&[f64], &mut [f64]) + 'static,
    ) -> usize {
        self.constraints.push(NlConstraint {
            kind: ConstraintKind::Le,
            f: Box::new(f),
            grad: Some(Box::new(grad)),
            active_if: None,
        });
        self.constraints.len() - 1
    }

    /// Add an `h(x) = 0` constraint; returns its index.
    pub fn add_eq(
        &mut self,
        f: impl Fn(&[f64]) -> f64 + 'static,
        grad: impl Fn(&[f64], &mut [f64]) + 'static,
    ) -> usize {
        self.constraints.push(NlConstraint {
            kind: ConstraintKind::Eq,
            f: Box::new(f),
            grad: Some(Box::new(grad)),
            active_if: None,
        });
        self.constraints.len() - 1
    }

    /// Attach an indicator gate to constraint `ci`: enforced iff binary
    /// variable `bin` equals `value`.
    pub fn set_indicator(&mut self, ci: usize, bin: usize, value: bool) {
        self.constraints[ci].active_if = Some((bin, value));
    }

    /// Evaluate the objective.
    pub fn eval_objective(&self, x: &[f64]) -> f64 {
        (self.objective)(x)
    }

    /// Gradient of the objective (analytic if provided, else finite diff).
    pub fn eval_objective_grad(&self, x: &[f64]) -> Vec<f64> {
        match &self.objective_grad {
            Some(g) => {
                let mut out = vec![0.0; x.len()];
                g(x, &mut out);
                out
            }
            None => finite_diff_grad(|y| (self.objective)(y), x),
        }
    }

    /// Whether constraint `ci` is active at the (integer-fixed) point `x`.
    pub fn is_active(&self, ci: usize, x: &[f64]) -> bool {
        match self.constraints[ci].active_if {
            Some((b, val)) => x[b].round() == if val { 1.0 } else { 0.0 },
            None => true,
        }
    }

    /// Evaluate constraint `ci`'s body.
    pub fn eval_constraint(&self, ci: usize, x: &[f64]) -> f64 {
        (self.constraints[ci].f)(x)
    }

    /// Gradient of constraint `ci`'s body.
    pub fn eval_constraint_grad(&self, ci: usize, x: &[f64]) -> Vec<f64> {
        let c = &self.constraints[ci];
        match &c.grad {
            Some(g) => {
                let mut out = vec![0.0; x.len()];
                g(x, &mut out);
                out
            }
            None => finite_diff_grad(|y| (c.f)(y), x),
        }
    }

    /// Violation of constraint `ci` at `x` (> 0 means violated).
    pub fn violation(&self, ci: usize, x: &[f64], tol: f64) -> f64 {
        let v = self.eval_constraint(ci, x);
        match self.constraints[ci].kind {
            ConstraintKind::Le => (v - tol).max(0.0),
            ConstraintKind::Eq => (v.abs() - tol).max(0.0),
        }
    }
}

/// Central finite-difference gradient of `f` at `x`.
pub fn finite_diff_grad(f: impl Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    let h = 1e-6;
    let mut g = vec![0.0f64; x.len()];
    for i in 0..x.len() {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[i] += h;
        xm[i] -= h;
        g[i] = (f(&xp) - f(&xm)) / (2.0 * h);
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_diff_matches_analytic() {
        // f = x0^2 + 3*x1 → grad = (2 x0, 3).
        let f = |x: &[f64]| x[0] * x[0] + 3.0 * x[1];
        let g = finite_diff_grad(f, &[1.5, -2.0]);
        assert!((g[0] - 3.0).abs() < 1e-4);
        assert!((g[1] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn indicator_activity_follows_binary() {
        let mut m = MinlpModel::new(2, |x| x[0] + x[1]);
        m.set_var(1, VarKind::Binary, 0.0, 1.0);
        let ci = m.add_le(|x| x[0] - 2.0, |_x, g| g[0] = 1.0);
        m.set_indicator(ci, 1, true);
        assert!(m.is_active(ci, &[0.0, 1.0]));
        assert!(!m.is_active(ci, &[0.0, 0.0]));
    }
}
