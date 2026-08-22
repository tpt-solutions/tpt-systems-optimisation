//! Convex relaxations for factorable nonlinear terms.
//!
//! Currently provides **McCormick envelopes** for bilinear products
//! `w = x · y` over box-bounded factors: the four classic tangent planes
//! that form the tightest convex relaxation of a bilinear term.

use std::vec::Vec;

use tpt_opt_core::model::Constraint;

/// Box bounds on the two factors of a bilinear product.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FactorBounds {
    /// Lower bound on `x`.
    pub lx: f64,
    /// Upper bound on `x`.
    pub ux: f64,
    /// Lower bound on `y`.
    pub ly: f64,
    /// Upper bound on `y`.
    pub uy: f64,
}

/// The four McCormick envelope rows for `w = x·y` with `x`, `y`, `w` at the
/// given variable indices. The rows are necessary and sufficient for `w` to
/// lie in the convex hull of `{(x, y, x·y) : x ∈ [lx,ux], y ∈ [ly,uy]}`
/// projected onto `w`.
///
/// Returns rows in order:
/// 1. `w >= lx·y + ly·x − lx·ly`
/// 2. `w >= ux·y + uy·x − ux·uy`
/// 3. `w <= ux·y + ly·x − ux·ly`
/// 4. `w <= lx·y + uy·x − lx·uy`
pub fn mccormick_envelope(x: usize, y: usize, w: usize, b: FactorBounds) -> Vec<Constraint> {
    vec![
        // w − lx·y − ly·x >= −lx·ly
        Constraint::ge(vec![w, y, x], vec![1.0, -b.lx, -b.ly], -b.lx * b.ly),
        // w − ux·y − uy·x >= −ux·uy
        Constraint::ge(vec![w, y, x], vec![1.0, -b.ux, -b.uy], -b.ux * b.uy),
        // w − ux·y − ly·x <= −ux·ly
        Constraint::le(vec![w, y, x], vec![1.0, -b.ux, -b.ly], -b.ux * b.ly),
        // w − lx·y − uy·x <= −lx·uy
        Constraint::le(vec![w, y, x], vec![1.0, -b.lx, -b.uy], -b.lx * b.uy),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_opt_core::model::Model;

    #[test]
    fn envelope_contains_true_product() {
        // For sampled factor values, w = x·y must satisfy all four rows.
        let b = FactorBounds { lx: -1.0, ux: 2.0, ly: 0.5, uy: 3.0 };
        let rows = mccormick_envelope(0, 1, 2, b);
        let samples = [
            [-1.0, 0.5],
            [-1.0, 3.0],
            [2.0, 0.5],
            [2.0, 3.0],
            [0.37, 1.42],
            [-0.62, 2.71],
            [1.0, 1.0],
        ];
        for [x, y] in samples {
            let pt = vec![x, y, x * y];
            for r in &rows {
                assert!(r.is_satisfied(&pt, 1e-9), "row violated at ({x},{y})");
            }
        }
    }

    #[test]
    fn envelope_is_tight_at_corners() {
        // At the four corners the envelope forces w == x·y exactly: minimise
        // and maximise w subject to the rows with x, y fixed at a corner.
        let b = FactorBounds { lx: 0.0, ux: 2.0, ly: 1.0, uy: 4.0 };
        let corners = [[0.0, 1.0], [0.0, 4.0], [2.0, 1.0], [2.0, 4.0]];
        for [cx, cy] in corners {
            for sense_min in [true, false] {
                use tpt_opt_core::Solver;
                let mut m = Model::new(3);
                m.set_objective(if sense_min {
                    tpt_opt_core::model::Objective::minimize(vec![2], vec![1.0])
                } else {
                    tpt_opt_core::model::Objective::maximize(vec![2], vec![1.0])
                });
                m.variables[0].bound = tpt_opt_core::bounds::VarBound::continuous(cx, cx);
                m.variables[1].bound = tpt_opt_core::bounds::VarBound::continuous(cy, cy);
                m.variables[2].bound = tpt_opt_core::bounds::VarBound::continuous(-100.0, 100.0);
                for r in mccormick_envelope(0, 1, 2, b) {
                    m.add_constraint(r);
                }
                let sol = tpt_opt_milp::MilpSolver::new().solve(&m).unwrap();
                assert_eq!(sol.status, tpt_opt_core::solver::SolverStatus::Optimal);
                assert!(
                    (sol.primal[2] - cx * cy).abs() < 1e-6,
                    "corner ({cx},{cy}): w={} != {}",
                    sol.primal[2],
                    cx * cy
                );
            }
        }
    }
}
