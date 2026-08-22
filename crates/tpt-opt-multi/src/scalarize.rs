//! Linear scalarisation helpers for multi-objective problems.
//!
//! A [`LinearMultiObjective`] is a set of linear objectives sharing one decision
//! vector `x`. It can be collapsed to a single-objective [`tpt_opt_core::Model`]
//! via the weighted-sum or ε-constraint method and solved with
//! [`tpt_opt_milp::MilpSolver`].

use std::vec::Vec;

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::solver::Solver;
use tpt_opt_core::{OptError, SolverStatus, VarBound};
use tpt_opt_milp::MilpSolver;

/// A single linear objective `c·x + constant`.
#[derive(Debug, Clone)]
pub struct LinearTerm {
    /// Per-variable coefficients.
    pub coeffs: Vec<f64>,
    /// Constant offset.
    pub constant: f64,
}

/// A multi-objective problem with linear objectives over one decision vector.
#[derive(Debug, Clone)]
pub struct LinearMultiObjective {
    objectives: Vec<LinearTerm>,
    bounds: Vec<(f64, f64)>,
}

impl LinearMultiObjective {
    /// Build from objectives and per-variable `(lower, upper)` bounds. All
    /// objectives must use the same number of variables as `bounds.len()`.
    pub fn new(objectives: Vec<LinearTerm>, bounds: Vec<(f64, f64)>) -> Self {
        assert!(!objectives.is_empty());
        let n = bounds.len();
        for o in &objectives {
            assert_eq!(o.coeffs.len(), n, "objective coefficient count != variable count");
        }
        Self { objectives, bounds }
    }

    /// Number of decision variables.
    pub fn num_vars(&self) -> usize {
        self.bounds.len()
    }

    /// Number of objectives.
    pub fn num_objectives(&self) -> usize {
        self.objectives.len()
    }

    /// Build the weighted-sum model `min Σ_i w_i (c_i·x + const_i)`.
    pub fn weighted_sum_model(&self, weights: &[f64]) -> Model {
        assert_eq!(weights.len(), self.objectives.len());
        let n = self.bounds.len();
        let mut model = Model::new(n);
        for (i, b) in self.bounds.iter().enumerate() {
            model.variables[i].bound = VarBound::continuous(b.0, b.1);
        }
        let mut obj_idx = Vec::with_capacity(n);
        let mut obj_coeff = Vec::with_capacity(n);
        let mut constant = 0.0f64;
        for (v, c) in self.objectives.iter().zip(weights) {
            for (j, &co) in v.coeffs.iter().enumerate() {
                obj_idx.push(j);
                obj_coeff.push(co * c);
            }
            constant += v.constant * c;
        }
        model.set_objective(Objective {
            sense: Sense::Minimize,
            indices: obj_idx,
            coeffs: obj_coeff,
            constant,
        });
        model
    }

    /// Build the ε-constraint model: minimise `objectives[primary]` subject to
    /// `objectives[j] <= eps[j]` for every `j != primary`.
    pub fn epsilon_constraint_model(&self, primary: usize, eps: &[f64]) -> Model {
        assert!(primary < self.objectives.len());
        assert_eq!(eps.len(), self.objectives.len());
        let n = self.bounds.len();
        let mut model = Model::new(n);
        for (i, b) in self.bounds.iter().enumerate() {
            model.variables[i].bound = VarBound::continuous(b.0, b.1);
        }
        // Primary objective as the minimisation target.
        let p = &self.objectives[primary];
        model.set_objective(Objective {
            sense: Sense::Minimize,
            indices: (0..n).collect(),
            coeffs: p.coeffs.clone(),
            constant: p.constant,
        });
        // Constraints objectives[j] <= eps[j] for j != primary.
        for (j, &limit) in eps.iter().enumerate() {
            if j == primary {
                continue;
            }
            let o = &self.objectives[j];
            let rhs = limit - o.constant;
            model.add_constraint(Constraint::le((0..n).collect(), o.coeffs.clone(), rhs));
        }
        model
    }

    /// Solve the weighted-sum problem, returning the optimal decision vector.
    pub fn solve_weighted_sum(&self, weights: &[f64]) -> Result<Vec<f64>, OptError> {
        let model = self.weighted_sum_model(weights);
        let mut solver = MilpSolver::new();
        let sol = solver.solve(&model)?;
        Ok(sol.primal)
    }

    /// Solve the ε-constraint problem, returning the optimal decision vector and
    /// the primary objective value.
    pub fn solve_epsilon_constraint(
        &self,
        primary: usize,
        eps: &[f64],
    ) -> Result<(Vec<f64>, f64), OptError> {
        let model = self.epsilon_constraint_model(primary, eps);
        let mut solver = MilpSolver::new();
        let sol = solver.solve(&model)?;
        Ok((sol.primal, sol.objective_value))
    }
}

/// Convenience constructor for a single linear objective term.
pub fn term(coeffs: Vec<f64>, constant: f64) -> LinearTerm {
    LinearTerm { coeffs, constant }
}

/// Solve a weighted-sum scalarisation and return the resulting objective vector.
pub fn weighted_sum(prob: &LinearMultiObjective, weights: &[f64]) -> Result<Vec<f64>, OptError> {
    let x = prob.solve_weighted_sum(weights)?;
    Ok(prob.objectives.iter().map(|o| o.constant + dot(&o.coeffs, &x)).collect())
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Marker trait alias kept for API symmetry with the other solver crates.
pub trait Scalarization {}
impl Scalarization for LinearMultiObjective {}

#[allow(dead_code)]
fn _assert_status(s: SolverStatus) -> bool {
    s.has_solution()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epsilon_constraint_caps_objective() {
        // min (x, y) s.t. x,y in [0,2]. ε-constraint: minimise f0=x subject to
        // f1=y <= 1.0. Optimal x = 0, y free in [0,1] -> y = 0.
        let prob = LinearMultiObjective::new(
            vec![term(vec![1.0, 0.0], 0.0), term(vec![0.0, 1.0], 0.0)],
            vec![(0.0, 2.0), (0.0, 2.0)],
        );
        let (sol, val) = prob.solve_epsilon_constraint(0, &[2.0, 1.0]).unwrap();
        assert!((val - 0.0).abs() < 1e-6, "primary objective x should be 0");
        assert!(sol[1] <= 1.0 + 1e-6, "y must respect the epsilon bound");
    }

    #[test]
    fn weighted_sum_minimises_sum() {
        let prob = LinearMultiObjective::new(
            vec![term(vec![1.0, 0.0], 0.0), term(vec![0.0, 1.0], 0.0)],
            vec![(0.0, 2.0), (0.0, 2.0)],
        );
        let x = prob.solve_weighted_sum(&[1.0, 1.0]).unwrap();
        assert!((x[0] - 0.0).abs() < 1e-6);
        assert!((x[1] - 0.0).abs() < 1e-6);
    }
}
