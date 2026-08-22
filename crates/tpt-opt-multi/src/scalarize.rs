//! Linear scalarisation helpers for multi-objective problems.
//!
//! A [`LinearMultiObjective`] is a set of linear objectives sharing one decision
//! vector `x`. It can be collapsed to a single-objective [`tpt_opt_core::Model`]
//! via the weighted-sum or ε-constraint method and solved with
//! [`tpt_opt_milp::MilpSolver`].

use std::vec::Vec;

use tpt_opt_core::bounds::Bound;
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

    /// Build the (plain) Tchebycheff model: minimise the epigraph variable `t`
    /// subject to `w_i * (f_i(x) - z_i) <= t` for every objective `i`.
    ///
    /// `weights` must be strictly positive and `ideal` (`z`) must be a valid
    /// utopia/ideal point with `z_i <= min f_i` over the feasible set — typically
    /// obtained from [`Self::ideal_point`]. Every weight vector paired with an
    /// ideal point yields at least one Pareto-optimal solution.
    ///
    /// The returned model has one extra free variable appended after the `n`
    /// decision variables: the epigraph variable `t` at index `n`.
    pub fn tchebycheff_model(&self, weights: &[f64], ideal: &[f64]) -> Model {
        self.tchebycheff_model_inner(weights, ideal, None)
    }

    /// Build the *augmented* Tchebycheff model: minimise
    /// `t + rho * sum_i (f_i(x) - z_i)` subject to the same epigraph rows.
    ///
    /// The augmentation term (`rho > 0`, commonly `1e-3 ..= 1e-1`) breaks ties
    /// among weakly-Pareto-optimal solutions so the returned point is guaranteed
    /// Pareto-efficient even when some weight is zero-ish or the front has flat
    /// pieces.
    pub fn augmented_tchebycheff_model(&self, weights: &[f64], ideal: &[f64], rho: f64) -> Model {
        assert!(rho > 0.0, "augmentation coefficient rho must be > 0");
        self.tchebycheff_model_inner(weights, ideal, Some(rho))
    }

    fn tchebycheff_model_inner(&self, weights: &[f64], ideal: &[f64], rho: Option<f64>) -> Model {
        assert_eq!(weights.len(), self.objectives.len());
        assert_eq!(ideal.len(), self.objectives.len());
        assert!(weights.iter().all(|&w| w > 0.0), "Tchebycheff weights must be strictly positive");
        let n = self.bounds.len();
        let mut model = Model::new(n);
        for (i, b) in self.bounds.iter().enumerate() {
            model.variables[i].bound = VarBound::continuous(b.0, b.1);
        }
        // Epigraph variable t (free).
        let t = model
            .add_variable(VarBound::continuous(Bound::UNBOUNDED_LOWER, Bound::UNBOUNDED_UPPER));
        // Epigraph rows: w_i * (c_i·x + const_i - z_i) <= t
        //            <=> w_i * c_i·x - t <= w_i * (z_i - const_i)
        for ((o, &w), &z) in self.objectives.iter().zip(weights).zip(ideal) {
            let mut idx: Vec<usize> = (0..n).collect();
            let mut coefs: Vec<f64> = o.coeffs.iter().map(|&c| w * c).collect();
            idx.push(t);
            coefs.push(-1.0);
            model.add_constraint(Constraint::le(idx, coefs, w * (z - o.constant)));
        }
        // Objective: min t (+ rho * sum_i (f_i(x) - z_i)) when augmented.
        match rho {
            None => {
                model.set_objective(Objective {
                    sense: Sense::Minimize,
                    indices: vec![t],
                    coeffs: vec![1.0],
                    constant: 0.0,
                });
            }
            Some(rho) => {
                // sum_i rho * (c_i·x + const_i - z_i): accumulate per variable.
                let mut acc = vec![0.0f64; n];
                let mut constant = 0.0f64;
                for (i, o) in self.objectives.iter().enumerate() {
                    for (j, &c) in o.coeffs.iter().enumerate() {
                        acc[j] += rho * c;
                    }
                    constant += rho * (o.constant - ideal[i]);
                }
                let mut idx: Vec<usize> = (0..n).collect();
                let mut coefs = acc;
                idx.push(t);
                coefs.push(1.0);
                model.set_objective(Objective {
                    sense: Sense::Minimize,
                    indices: idx,
                    coeffs: coefs,
                    constant,
                });
            }
        }
        model
    }

    /// Compute the ideal point `z*` by minimising every objective individually.
    /// Each entry satisfies `z*_i <= min f_i` over the box-feasible set, which is
    /// exactly what the Tchebycheff methods expect.
    pub fn ideal_point(&self) -> Result<Vec<f64>, OptError> {
        let mut z = Vec::with_capacity(self.objectives.len());
        for _ in 0..self.objectives.len() {
            z.push(f64::INFINITY);
        }
        for (i, _) in self.objectives.iter().enumerate() {
            // Minimise objective i alone via weighted-sum with unit weight on i.
            let mut w = vec![0.0; self.objectives.len()];
            w[i] = 1.0;
            let model = self.weighted_sum_model(&w);
            let mut solver = MilpSolver::new();
            let sol = solver.solve(&model)?;
            let fi = self.objectives[i].constant + dot(&self.objectives[i].coeffs, &sol.primal);
            z[i] = fi;
        }
        Ok(z)
    }

    /// Solve the Tchebycheff problem, returning the optimal decision vector and
    /// the full objective vector at that point.
    pub fn solve_tchebycheff(
        &self,
        weights: &[f64],
        ideal: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), OptError> {
        let model = self.tchebycheff_model(weights, ideal);
        let mut solver = MilpSolver::new();
        let sol = solver.solve(&model)?;
        let x = sol.primal[..self.num_vars()].to_vec();
        let f = self.evaluate(&x);
        Ok((x, f))
    }

    /// Solve the augmented Tchebycheff problem (see
    /// [`Self::augmented_tchebycheff_model`]), returning the optimal decision
    /// vector and the full objective vector at that point.
    pub fn solve_augmented_tchebycheff(
        &self,
        weights: &[f64],
        ideal: &[f64],
        rho: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), OptError> {
        let model = self.augmented_tchebycheff_model(weights, ideal, rho);
        let mut solver = MilpSolver::new();
        let sol = solver.solve(&model)?;
        let x = sol.primal[..self.num_vars()].to_vec();
        let f = self.evaluate(&x);
        Ok((x, f))
    }

    /// Evaluate every objective at `x`.
    pub fn evaluate(&self, x: &[f64]) -> Vec<f64> {
        self.objectives.iter().map(|o| o.constant + dot(&o.coeffs, x)).collect()
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

/// Solve a Tchebycheff scalarisation and return the resulting objective vector.
///
/// Convenience wrapper around [`LinearMultiObjective::solve_tchebycheff`].
pub fn tchebycheff(
    prob: &LinearMultiObjective,
    weights: &[f64],
    ideal: &[f64],
) -> Result<Vec<f64>, OptError> {
    let (_, f) = prob.solve_tchebycheff(weights, ideal)?;
    Ok(f)
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

    #[test]
    fn ideal_point_matches_individual_minima() {
        // f0 = x, f1 = y on [0,2]^2: ideal point is (0, 0).
        let prob = LinearMultiObjective::new(
            vec![term(vec![1.0, 0.0], 0.0), term(vec![0.0, 1.0], 0.0)],
            vec![(0.0, 2.0), (0.0, 2.0)],
        );
        let z = prob.ideal_point().unwrap();
        assert!((z[0] - 0.0).abs() < 1e-6);
        assert!((z[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn tchebycheff_achieves_weighted_deviation_optimum() {
        // f0 = x, f1 = y on [0,2]^2, ideal (0,0), weights (1, 2):
        // min max(x, 2y). Optimum t = 0 at x = y = 0.
        let prob = LinearMultiObjective::new(
            vec![term(vec![1.0, 0.0], 0.0), term(vec![0.0, 1.0], 0.0)],
            vec![(0.0, 2.0), (0.0, 2.0)],
        );
        let (x, f) = prob.solve_tchebycheff(&[1.0, 2.0], &[0.0, 0.0]).unwrap();
        assert!(x[0] >= -1e-6 && x[0] <= 1e-6, "x should be ~0, got {}", x[0]);
        assert!(x[1] >= -1e-6 && x[1] <= 1e-6, "y should be ~0, got {}", x[1]);
        assert!((f[0] - 0.0).abs() < 1e-6 && (f[1] - 0.0).abs() < 1e-6);

        // Shifted ideal forces a non-trivial trade-off: ideal (1, 1) means
        // deviations are (x-1, y-1); min max(x-1, 2(y-1)) over [0,2]^2 is
        // attained at any point with x-1 = 2(y-1) <= 0, e.g. (1,1) itself
        // giving t = 0. Verify the reported deviation equals the optimum.
        let (_, f) = prob.solve_tchebycheff(&[1.0, 2.0], &[1.0, 1.0]).unwrap();
        let dev = (f[0] - 1.0).max(2.0 * (f[1] - 1.0));
        assert!(dev <= 1e-6, "max weighted deviation should be <= 0, got {dev}");
    }

    #[test]
    fn tchebycheff_recovers_pareto_extremes() {
        // Coupled objectives over one effective dimension: f0 = a, f1 = 1 - a
        // on [0,1]. The Pareto front is the segment {(a, 1-a)} and the ideal
        // point is (0, 0). A heavy weight on f0 pulls the solution toward the
        // f0-ideal (a -> 0); a heavy weight on f1 toward the f1-ideal (a -> 1).
        let coupled = LinearMultiObjective::new(
            vec![term(vec![1.0, 0.0], 0.0), term(vec![-1.0, 0.0], 1.0)],
            vec![(0.0, 1.0), (0.0, 1.0)],
        );
        let z = vec![0.0, 0.0];
        let (_, f_a) = coupled.solve_tchebycheff(&[100.0, 1.0], &z).unwrap();
        assert!(f_a[0] < 0.05, "heavy f0 weight should minimise f0, got {}", f_a[0]);
        let (_, f_b) = coupled.solve_tchebycheff(&[1.0, 100.0], &z).unwrap();
        assert!(f_b[1] < 0.05, "heavy f1 weight should minimise f1, got {}", f_b[1]);
    }

    #[test]
    fn augmented_tchebycheff_excludes_weakly_dominated() {
        // Classic weak-dominance trap: f0 = x, f1 = y on [0,1]^2 with ideal
        // (-1, -1). Plain Tchebycheff with weights (1, 1) minimises
        // max(x+1, y+1) — any point on x = y is optimal, including weakly
        // dominated ones. The augmentation term rho*(x+y) pushes the solution
        // to the Pareto-efficient corner (0, 0).
        let prob = LinearMultiObjective::new(
            vec![term(vec![1.0, 0.0], 0.0), term(vec![0.0, 1.0], 0.0)],
            vec![(0.0, 1.0), (0.0, 1.0)],
        );
        let ideal = vec![-1.0, -1.0];
        let (_x_aug, f_aug) = prob.solve_augmented_tchebycheff(&[1.0, 1.0], &ideal, 1e-2).unwrap();
        assert!(
            f_aug[0] < 1e-6 && f_aug[1] < 1e-6,
            "augmented Tchebycheff should reach the efficient corner (0,0), got {f_aug:?}"
        );
        // Plain Tchebycheff may legitimately stop anywhere on x = y; both must
        // agree on the epigraph value.
        let (_, f_plain) = prob.solve_tchebycheff(&[1.0, 1.0], &ideal).unwrap();
        let d_plain = (f_plain[0] + 1.0).max(f_plain[1] + 1.0);
        let d_aug = (f_aug[0] + 1.0).max(f_aug[1] + 1.0);
        assert!((d_plain - d_aug).abs() < 1e-6, "epigraph optima must coincide");
    }

    #[test]
    #[should_panic(expected = "strictly positive")]
    fn tchebycheff_rejects_nonpositive_weights() {
        let prob = LinearMultiObjective::new(
            vec![term(vec![1.0, 0.0], 0.0), term(vec![0.0, 1.0], 0.0)],
            vec![(0.0, 2.0), (0.0, 2.0)],
        );
        let _ = prob.solve_tchebycheff(&[1.0, 0.0], &[0.0, 0.0]);
    }
}
