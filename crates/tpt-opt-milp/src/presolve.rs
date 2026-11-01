//! Light presolve that keeps variable indices unchanged.
//!
//! Three reductions run in one pass over the linear constraints:
//!
//! - **Empty rows** (no nonzero coefficients) are dropped, or flag the model as
//!   infeasible when `0` lies outside their bounds.
//! - **Singleton rows** `a·x_j ∈ [lo, hi]` are turned into bounds on `x_j`,
//!   rounded inward for integer variables, and then dropped.
//! - **Duplicate terms** on the same variable are merged and zero coefficients
//!   removed.
//!
//! Variables are never removed, so a solution of the presolved model is a
//! solution of the original with the same indexing. The report says what
//! changed; when `infeasible` is set the returned model is not valid to solve.

use tpt_opt_core::bounds::{Bound, VarType};
use tpt_opt_core::model::{Constraint, Model};

const TOL: f64 = 1e-9;

/// What [`presolve_model`] changed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PresolveReport {
    /// Constraints removed (empty or singleton rows).
    pub removed_rows: usize,
    /// Variable bounds tightened by singleton rows.
    pub tightened_bounds: usize,
    /// Variables whose presolved bounds pin them to a single value.
    pub fixed_variables: usize,
    /// A row or bound proved the model infeasible.
    pub infeasible: bool,
}

/// Presolve `model`, returning the reduced model and what was done.
///
/// ```
/// use tpt_opt_core::expr::ModelBuilder;
/// use tpt_opt_milp::presolve_model;
///
/// let mut b = ModelBuilder::new();
/// let x = b.integer("x", 0.0, 10.0);
/// b.constrain((2.0 * x).le(5.0)); // singleton: x <= 2.5, so x <= 2
/// b.minimize(-1.0 * x);
/// let (reduced, report) = presolve_model(&b.build().unwrap());
///
/// assert_eq!(report.removed_rows, 1);
/// assert_eq!(reduced.variables[0].bound.bound.upper, 2.0);
/// ```
pub fn presolve_model(model: &Model) -> (Model, PresolveReport) {
    let mut report = PresolveReport::default();
    let mut out = model.clone();
    let mut kept = Vec::with_capacity(model.constraints.len());

    for c in &model.constraints {
        if c.is_custom {
            kept.push(c.clone());
            continue;
        }
        let terms = merged_terms(c);
        match terms.as_slice() {
            [] => {
                if c.lower > TOL || c.upper < -TOL {
                    report.infeasible = true;
                }
                report.removed_rows += 1;
            }
            [(j, a)] => {
                report.removed_rows += 1;
                let (lo, hi) =
                    if *a > 0.0 { (c.lower / a, c.upper / a) } else { (c.upper / a, c.lower / a) };
                tighten(&mut out, *j, lo, hi, &mut report);
            }
            _ => kept.push(Constraint {
                indices: terms.iter().map(|&(j, _)| j).collect(),
                coeffs: terms.iter().map(|&(_, a)| a).collect(),
                lower: c.lower,
                upper: c.upper,
                is_custom: false,
            }),
        }
    }

    out.constraints = kept;
    report.fixed_variables = out
        .variables
        .iter()
        .filter(|v| {
            v.bound.bound.upper - v.bound.bound.lower <= TOL && v.bound.bound.lower.is_finite()
        })
        .count();
    (out, report)
}

/// Sum duplicate indices and drop zero coefficients, preserving first-seen order.
fn merged_terms(c: &Constraint) -> Vec<(usize, f64)> {
    let mut merged: Vec<(usize, f64)> = Vec::with_capacity(c.indices.len());
    for (&j, &a) in c.indices.iter().zip(&c.coeffs) {
        match merged.iter_mut().find(|(k, _)| *k == j) {
            Some(entry) => entry.1 += a,
            None => merged.push((j, a)),
        }
    }
    merged.retain(|&(_, a)| a.abs() > TOL);
    merged
}

/// Intersect variable `j`'s bounds with `[lo, hi]`, rounding inward for
/// integer variables, and record the change.
fn tighten(model: &mut Model, j: usize, lo: f64, hi: f64, report: &mut PresolveReport) {
    let var = &mut model.variables[j];
    let mut new_lo = var.bound.bound.lower.max(lo);
    let mut new_hi = var.bound.bound.upper.min(hi);
    if matches!(var.kind, VarType::Integer | VarType::Binary) {
        new_lo = (new_lo - TOL).ceil();
        new_hi = (new_hi + TOL).floor();
    }
    if new_lo > new_hi + TOL {
        report.infeasible = true;
        return;
    }
    if new_lo > var.bound.bound.lower + TOL || new_hi < var.bound.bound.upper - TOL {
        report.tightened_bounds += 1;
    }
    // Keep the bounds ordered when rounding has collapsed the interval.
    let new_hi = new_hi.max(new_lo);
    var.bound.bound = Bound { lower: new_lo, upper: new_hi };
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_opt_core::expr::ModelBuilder;

    #[test]
    fn singleton_row_tightens_and_is_removed() {
        let mut b = ModelBuilder::new();
        let x = b.continuous("x", 0.0, 10.0);
        b.constrain((-2.0 * x).ge(-6.0)); // -2x >= -6  ⇔  x <= 3
        b.minimize(x);
        let (m, r) = presolve_model(&b.build().unwrap());
        assert_eq!(m.num_constraints(), 0);
        assert_eq!(m.variables[0].bound.bound.upper, 3.0);
        assert_eq!(r.removed_rows, 1);
        assert_eq!(r.tightened_bounds, 1);
        assert!(!r.infeasible);
    }

    #[test]
    fn integer_bounds_round_inward() {
        let mut b = ModelBuilder::new();
        let n = b.integer("n", 0.0, 10.0);
        b.constrain(n.ge(2.2));
        b.minimize(n);
        let (m, _) = presolve_model(&b.build().unwrap());
        assert_eq!(m.variables[0].bound.bound.lower, 3.0);
    }

    #[test]
    fn empty_row_with_violated_bounds_is_infeasible() {
        let mut m = Model::new(1);
        m.add_constraint(Constraint::new(vec![], vec![], 1.0, 2.0).unwrap());
        let (_, r) = presolve_model(&m);
        assert!(r.infeasible);
        assert_eq!(r.removed_rows, 1);
    }

    #[test]
    fn conflicting_singletons_are_flagged() {
        let mut b = ModelBuilder::new();
        let x = b.continuous("x", 0.0, 10.0);
        b.constrain(x.ge(4.0));
        b.constrain(x.le(1.0));
        let (_, r) = presolve_model(&b.build().unwrap());
        assert!(r.infeasible);
    }

    #[test]
    fn duplicate_terms_merge_before_classifying_the_row() {
        let mut b = ModelBuilder::new();
        let x = b.continuous("x", 0.0, 10.0);
        let y = b.continuous("y", 0.0, 10.0);
        b.constrain((x + x - 2.0 * x + y).le(4.0)); // reduces to y <= 4
        b.minimize(y);
        let (m, r) = presolve_model(&b.build().unwrap());
        assert_eq!(r.removed_rows, 1);
        assert_eq!(m.variables[1].bound.bound.upper, 4.0);
    }

    #[test]
    fn multi_term_rows_survive_untouched() {
        let mut b = ModelBuilder::new();
        let x = b.continuous("x", 0.0, 10.0);
        let y = b.continuous("y", 0.0, 10.0);
        b.constrain((x + y).le(4.0));
        b.minimize(x);
        let (m, r) = presolve_model(&b.build().unwrap());
        assert_eq!(m.num_constraints(), 1);
        assert_eq!(r.removed_rows, 0);
    }
}
