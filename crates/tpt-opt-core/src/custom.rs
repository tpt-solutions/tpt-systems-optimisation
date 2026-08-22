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
/// (scalar *violation* in `g(x) <= 0` form: any non-positive value is
/// satisfied, positive values measure by how much the constraint is broken),
/// and `gradient` (partial derivatives of the violation with respect to the
/// referenced variables). `is_violated` has a default implementation based on
/// `evaluate` and a tolerance.
pub trait CustomConstraint {
    /// Number of variables this constraint references.
    fn arity(&self) -> usize;

    /// Evaluate the violation at `x`, where `x` has length `arity()` (the
    /// subset of the full variable vector this constraint depends on).
    /// Any value `<= 0.0` means the constraint is satisfied; a positive value
    /// measures the violation magnitude.
    fn evaluate(&self, x: &[f64]) -> f64;

    /// Write the gradient of `evaluate` at `x` into `grad` (length `arity()`).
    fn gradient(&self, x: &[f64], grad: &mut [f64]);

    /// `true` if the constraint is violated at `x` by more than `tol`.
    ///
    /// The default uses one-sided violation semantics (`evaluate(x) > tol`),
    /// matching the `g(x) <= 0`-means-satisfied convention of [`Self::evaluate`]:
    /// strictly-satisfied inequality constraints are *not* violations.
    /// Equality-style constraints (`g(x) == 0`) should override this with an
    /// absolute-value check.
    fn is_violated(&self, x: &[f64], tol: f64) -> bool {
        self.evaluate(x) > tol
    }
}

/// Helper: largest violation across a box of custom constraints over the same
/// `x` subset (zero when every constraint is satisfied).
pub fn max_violation<C: CustomConstraint>(constraints: &[C], x: &[f64], _tol: f64) -> f64 {
    let mut max = 0.0;
    for c in constraints {
        let v = c.evaluate(x);
        if v > max {
            max = v;
        }
    }
    max
}
