//! Logical constraints over binary variables, compiled to linear rows.

use std::vec::Vec;

use tpt_opt_core::model::{Constraint, Model};

/// A propositional constraint over binary variables (by index).
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalConstraint {
    /// All listed variables must be 1.
    And(Vec<usize>),
    /// At least one listed variable must be 1.
    Or(Vec<usize>),
    /// Exactly one listed variable must be 1.
    Xor(Vec<usize>),
    /// At least `k` of the listed variables are 1.
    AtLeast(Vec<usize>, usize),
    /// At most `k` of the listed variables are 1.
    AtMost(Vec<usize>, usize),
    /// Exactly `k` of the listed variables are 1.
    Exactly(Vec<usize>, usize),
    /// `a → b`: if `a` is 1 then `b` must be 1.
    Implies(usize, usize),
}

/// Compile `lc` into linear rows added to `model`. The referenced variables
/// must already be binary in `model`.
pub fn add_logical(model: &mut Model, lc: &LogicalConstraint) {
    match lc {
        LogicalConstraint::And(vars) => {
            for &v in vars {
                model.add_constraint(Constraint::ge(vec![v], vec![1.0], 1.0));
            }
        }
        LogicalConstraint::Or(vars) => {
            let (idx, coefs) = ones(vars);
            model.add_constraint(Constraint::ge(idx, coefs, 1.0));
        }
        LogicalConstraint::Xor(vars) => {
            let (idx, coefs) = ones(vars);
            model.add_constraint(Constraint::equality(idx, coefs, 1.0));
        }
        LogicalConstraint::AtLeast(vars, k) => {
            let (idx, coefs) = ones(vars);
            model.add_constraint(Constraint::ge(idx, coefs, *k as f64));
        }
        LogicalConstraint::AtMost(vars, k) => {
            let (idx, coefs) = ones(vars);
            model.add_constraint(Constraint::le(idx, coefs, *k as f64));
        }
        LogicalConstraint::Exactly(vars, k) => {
            let (idx, coefs) = ones(vars);
            model.add_constraint(Constraint::equality(idx, coefs, *k as f64));
        }
        LogicalConstraint::Implies(a, b) => {
            model.add_constraint(Constraint::ge(vec![*b, *a], vec![1.0, -1.0], 0.0));
        }
    }
}

fn ones(vars: &[usize]) -> (Vec<usize>, Vec<f64>) {
    (vars.to_vec(), vec![1.0; vars.len()])
}

/// Enumerate all assignments of `k` binaries as booleans (for tests).
#[cfg(test)]
fn enumerate_binaries(k: usize) -> Vec<Vec<bool>> {
    let total = 1usize << k;
    (0..total).map(|mask| (0..k).map(|b| mask >> b & 1 == 1).collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_opt_core::Solver;

    fn binary_model(n: usize) -> Model {
        use tpt_opt_core::bounds::VarBound;
        let mut m = Model::new(n);
        for v in m.variables.iter_mut() {
            v.bound = VarBound::binary();
        }
        m
    }

    fn satisfies(lc: &LogicalConstraint, bits: &[bool]) -> bool {
        match lc {
            LogicalConstraint::And(vs) => vs.iter().all(|&v| bits[v]),
            LogicalConstraint::Or(vs) => vs.iter().any(|&v| bits[v]),
            LogicalConstraint::Xor(vs) => vs.iter().filter(|&&v| bits[v]).count() == 1,
            LogicalConstraint::AtLeast(vs, k) => vs.iter().filter(|&&v| bits[v]).count() >= *k,
            LogicalConstraint::AtMost(vs, k) => vs.iter().filter(|&&v| bits[v]).count() <= *k,
            LogicalConstraint::Exactly(vs, k) => vs.iter().filter(|&&v| bits[v]).count() == *k,
            LogicalConstraint::Implies(a, b) => !bits[*a] || bits[*b],
        }
    }

    #[test]
    fn compiled_rows_match_semantics() {
        let cases: Vec<LogicalConstraint> = vec![
            LogicalConstraint::And(vec![0, 1]),
            LogicalConstraint::Or(vec![0, 1, 2]),
            LogicalConstraint::Xor(vec![0, 1, 2]),
            LogicalConstraint::AtLeast(vec![0, 1, 2], 2),
            LogicalConstraint::AtMost(vec![0, 1, 2], 1),
            LogicalConstraint::Exactly(vec![0, 1, 2], 2),
            LogicalConstraint::Implies(0, 2),
        ];
        for lc in &cases {
            let mut m = binary_model(3);
            m.set_objective(tpt_opt_core::model::Objective::minimize(vec![0], vec![1.0]));
            add_logical(&mut m, lc);
            for bits in enumerate_binaries(3) {
                let x: Vec<f64> = bits.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
                let rows_ok = m.constraints.iter().all(|c| c.is_satisfied(&x, 1e-9));
                assert_eq!(rows_ok, satisfies(lc, &bits), "{lc:?} mismatch at {bits:?}");
            }
        }
    }

    #[test]
    fn feasible_assignment_survives_compilation() {
        // Or([0,1,2]) with objective maximising b0 keeps a satisfying point.
        let mut m = binary_model(3);
        m.set_objective(tpt_opt_core::model::Objective::maximize(vec![0], vec![1.0]));
        add_logical(&mut m, &LogicalConstraint::Or(vec![0, 1, 2]));
        let sol = tpt_opt_milp::MilpSolver::new().solve(&m).unwrap();
        assert_eq!(sol.status, tpt_opt_core::solver::SolverStatus::Optimal);
        assert_eq!(sol.primal[0], 1.0);
    }
}
