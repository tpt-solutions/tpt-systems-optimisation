//! Extensibility end-to-end test (spec §4 / todo.md cross-cutting checklist):
//! a user-defined global constraint implemented against the public
//! [`tpt_opt_cp::constraints::Constraint`] trait, plugged into a [`CpModel`]
//! and solved through the standard search.

use std::vec::Vec;

use tpt_opt_cp::constraints::{Constraint, Inconsistency};
use tpt_opt_cp::model::CpModel;
use tpt_opt_cp::solver::{solutions, solve};

/// Custom constraint: `x^2 + y^2 <= limit` over two integer variables.
///
/// Propagation is bounds-consistent: value `v` is removed from `x`'s domain
/// when even the smallest possible `y^2` cannot keep the sum within `limit`
/// (and symmetrically for `y`). `check` is the exact arithmetic test used by
/// the solver on complete assignments.
struct SumOfSquaresLe {
    x: usize,
    y: usize,
    limit: i64,
    /// Owned mirror of `[x, y]` so `vars()` can return a slice.
    scope: Vec<usize>,
}

impl SumOfSquaresLe {
    fn new(x: usize, y: usize, limit: i64) -> Self {
        Self { x, y, limit, scope: vec![x, y] }
    }
}

impl Constraint for SumOfSquaresLe {
    fn vars(&self) -> &[usize] {
        &self.scope
    }

    fn propagate(&self, doms: &mut [tpt_opt_cp::domain::Domain]) -> Result<(), Inconsistency> {
        let sq = |&v: &usize| (v as i64) * (v as i64);
        let min_y_sq = doms[self.y].values().iter().map(sq).min().unwrap_or(0);
        let min_x_sq = doms[self.x].values().iter().map(sq).min().unwrap_or(0);
        for &v in doms[self.x].values().to_vec().as_slice() {
            if sq(&v) + min_y_sq > self.limit {
                doms[self.x].remove(v);
            }
        }
        for &v in doms[self.y].values().to_vec().as_slice() {
            if sq(&v) + min_x_sq > self.limit {
                doms[self.y].remove(v);
            }
        }
        if doms[self.x].is_empty() || doms[self.y].is_empty() {
            return Err(Inconsistency);
        }
        Ok(())
    }

    fn check(&self, assign: &[usize]) -> bool {
        let a = assign[self.x] as i64;
        let b = assign[self.y] as i64;
        a * a + b * b <= self.limit
    }
}

fn main_test_model(limit: i64) -> (CpModel, usize, usize) {
    let mut m = CpModel::new();
    let x = m.add_var(0, 10);
    let y = m.add_var(0, 10);
    m.add_constraint(Box::new(SumOfSquaresLe::new(x, y, limit)));
    (m, x, y)
}

#[test]
fn custom_constraint_finds_feasible_point() {
    // 3^2 + 4^2 = 25 <= 25 exactly on the boundary; plenty of feasible pairs.
    let (m, x, y) = main_test_model(25);
    let sol = solve(&m).expect("feasible model");
    let (a, b) = (sol.assignment[x] as i64, sol.assignment[y] as i64);
    assert!(a * a + b * b <= 25, "solution ({a}, {b}) violates the sum-of-squares bound");
}

#[test]
fn custom_constraint_proves_infeasibility() {
    // Minimum possible sum of squares on [0,10]^2 is 0; force both variables
    // away from small values so that no pair fits under the limit.
    let mut m = CpModel::new();
    let x = m.add_var_values(vec![5, 6, 7, 8, 9, 10]);
    let y = m.add_var_values(vec![5, 6, 7, 8, 9, 10]);
    m.add_constraint(Box::new(SumOfSquaresLe::new(x, y, 30)));
    assert!(solve(&m).is_none(), "no pair of values >= 5 squares to <= 30");
}

#[test]
fn custom_constraint_propagation_prunes_domains() {
    // With limit 1 and y >= 1, x can only be 0 or 1 (x^2 <= 1 - y^2 <= 0).
    let mut m = CpModel::new();
    let x = m.add_var(0, 10);
    let y = m.add_var_values(vec![1]);
    m.add_constraint(Box::new(SumOfSquaresLe::new(x, y, 1)));
    let sol = solve(&m).expect("x=0,y=1 is feasible");
    assert_eq!(sol.assignment[x], 0, "propagation must force x to 0");
    assert_eq!(sol.assignment[y], 1);
}

#[test]
fn custom_constraint_enumeration_all_satisfy() {
    // Enumerate every solution of a loose instance and check each one against
    // the exact predicate.
    let (m, x, y) = main_test_model(40);
    let sols = solutions(&m, 500);
    assert!(!sols.is_empty());
    for s in &sols {
        let (a, b) = (s.assignment[x] as i64, s.assignment[y] as i64);
        assert!(a * a + b * b <= 40, "enumerated solution ({a}, {b}) violates the bound");
    }
}
