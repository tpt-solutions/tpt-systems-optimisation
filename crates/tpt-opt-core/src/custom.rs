//! Extensibility hook for user-defined constraints (spec §4 `CustomConstraint`).
//!
//! A [`CustomConstraint`] describes a (possibly non-linear) relationship over a
//! fixed subset of variables. Solvers evaluate `evaluate`/`gradient` to drive
//! their custom-constraint path and use `is_violated` for feasibility checks.
//! The canonical [`crate::model::Constraint`] carries an `is_custom` flag so a
//! solver can route custom rows through this trait instead of the sparse matrix.

/// A user-supplied constraint over a fixed subset of variables.
///
/// Implementors define `arity` (number of variables referenced), `evaluate`
/// (scalar violation, `0` = satisfied), and `gradient` (partial derivatives of
/// the violation with respect to the referenced variables). `is_violated` has a
/// default implementation based on `evaluate` and a tolerance.
pub trait CustomConstraint {
    /// Number of variables this constraint references.
    fn arity(&self) -> usize;

    /// Evaluate the violation at `x`, where `x` has length `arity()` (the
    /// subset of the full variable vector this constraint depends on).
    /// A return value of `0.0` means the constraint is satisfied.
    fn evaluate(&self, x: &[f64]) -> f64;

    /// Write the gradient of `evaluate` at `x` into `grad` (length `arity()`).
    fn gradient(&self, x: &[f64], grad: &mut [f64]);

    /// `true` if the constraint is violated at `x` within `tol`.
    fn is_violated(&self, x: &[f64], tol: f64) -> bool {
        self.evaluate(x).abs() > tol
    }
}

/// Helper: evaluate a box of custom constraints over the same `x` subset.
pub fn max_violation<C: CustomConstraint>(constraints: &[C], x: &[f64], tol: f64) -> f64 {
    let mut max = 0.0;
    for c in constraints {
        let v = c.evaluate(x).abs();
        if v > max {
            max = v;
        }
        let _ = tol;
    }
    max
}
