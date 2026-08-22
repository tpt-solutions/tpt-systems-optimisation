//! Linearised complementarity constraints.
//!
//! The pair `0 <= x ⊥ y >= 0` (`x·y = 0`) is linearised with one switching
//! binary `u` and big-M rows:
//!
//! ```text
//! x <= M·u          (x > 0 forces u = 1)
//! y <= M·(1 − u)    (y > 0 forces u = 0)
//! ```
//!
//! together with `x, y ∈ [0, M]` bounds supplied by the caller. This is the
//! standard big-M reformulation; it is exact when `M` dominates both
//! variables' ranges.

use std::vec::Vec;

use tpt_opt_core::{
    bounds::VarBound,
    model::{Constraint, Model},
};

/// The two big-M rows for `x·y = 0` with switching binary `u`:
/// `x − M·u <= 0` and `y + M·u <= M`. Caller sets `x, y ∈ [0, M]` bounds
/// and marks `u` binary.
pub fn complementarity_rows(x: usize, y: usize, u: usize, big_m: f64) -> Vec<Constraint> {
    vec![
        Constraint::le(vec![x, u], vec![1.0, -big_m], 0.0),
        Constraint::le(vec![y, u], vec![1.0, big_m], big_m),
    ]
}

/// Helper that builds a fresh model of size `n + 1` whose extra trailing
/// variable is the complementarity switch, returning `(model, u_index)`.
pub fn with_complementarity_var<F>(n: usize, build: F) -> (Model, usize)
where
    F: FnOnce(&mut Model, usize),
{
    let mut m = Model::new(n + 1);
    m.variables[n].bound = VarBound::binary();
    build(&mut m, n);
    (m, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_opt_core::{bounds::VarBound, solver::SolverStatus, Solver};
    use tpt_opt_milp::MilpSolver;

    const M: f64 = 10.0;

    fn solve_min(sum_coef: f64) -> Vec<f64> {
        // min sum_coef*(x+y) s.t. x ⊥ y, x,y ∈ [0,M].
        let (m, u) = with_complementarity_var(2, |m, u| {
            m.set_objective(tpt_opt_core::model::Objective::minimize(
                vec![0, 1],
                vec![sum_coef, sum_coef],
            ));
            m.variables[0].bound = VarBound::continuous(0.0, M);
            m.variables[1].bound = VarBound::continuous(0.0, M);
            for r in complementarity_rows(0, 1, u, M) {
                m.add_constraint(r);
            }
        });
        let _ = u;
        let sol = MilpSolver::new().solve(&m).unwrap();
        assert_eq!(sol.status, SolverStatus::Optimal);
        sol.primal
    }

    #[test]
    fn minimisation_drives_both_to_zero() {
        let x = solve_min(1.0);
        assert!(x[0].abs() < 1e-6 && x[1].abs() < 1e-6, "{x:?}");
    }

    #[test]
    fn product_is_zero_at_optimum() {
        // Maximise x+y → one of them hits M, the other 0; product still 0.
        let (m, _u) = with_complementarity_var(2, |m, u| {
            m.set_objective(tpt_opt_core::model::Objective::maximize(vec![0, 1], vec![1.0, 1.0]));
            m.variables[0].bound = VarBound::continuous(0.0, M);
            m.variables[1].bound = VarBound::continuous(0.0, M);
            for r in complementarity_rows(0, 1, u, M) {
                m.add_constraint(r);
            }
        });
        let sol = MilpSolver::new().solve(&m).unwrap();
        assert_eq!(sol.status, SolverStatus::Optimal);
        assert!((sol.objective_value - M).abs() < 1e-6, "obj {}", sol.objective_value);
        assert!(sol.primal[0] * sol.primal[1] < 1e-6, "product nonzero: {:?}", sol.primal);
    }
}
