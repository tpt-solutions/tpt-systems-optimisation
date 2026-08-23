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

    /// Compute the pay-off table: minimise each objective individually and
    /// record the full objective vector at each single-objective optimum.
    ///
    /// Returns `(ideal, nadir)` where `ideal[j]` is the best value ever seen
    /// for objective `j` (a valid utopia point for the Tchebycheff methods)
    /// and `nadir[j]` is the worst value observed for `j` across those
    /// optima — the classic pay-off-table nadir estimate. For non-convex
    /// fronts the nadir estimate can under-approximate the true nadir; it is
    /// used here only to *normalise* deviations, so a mild underestimate is
    /// harmless.
    pub fn payoff_table(&self) -> Result<(Vec<f64>, Vec<f64>), OptError> {
        let m = self.objectives.len();
        let mut ideal = vec![f64::INFINITY; m];
        let mut nadir = vec![f64::NEG_INFINITY; m];
        for i in 0..m {
            let mut w = vec![0.0; m];
            w[i] = 1.0;
            let model = self.weighted_sum_model(&w);
            let mut solver = MilpSolver::new();
            let sol = solver.solve(&model)?;
            let f = self.evaluate(&sol.primal);
            for (j, &fv) in f.iter().enumerate() {
                ideal[j] = ideal[j].min(fv);
                nadir[j] = nadir[j].max(fv);
            }
        }
        Ok((ideal, nadir))
    }

    /// Adaptive weighted-Tchebycheff scalarisation.
    ///
    /// Repeatedly solves the augmented Tchebycheff problem while adapting the
    /// weights toward a balanced achievement: after every solve the normalised
    /// deviation of each objective at the current point,
    /// `d_i = (f_i(x) - z_i) / range_i` with `range_i = nadir_i - z_i`, is
    /// compared to its mean, and weights are multiplicatively updated as
    /// `w_i <- w_i * exp(eta * (d_i - mean(d)))` (then renormalised). Objectives
    /// that lag behind the others gain weight, so the iteration drifts away
    /// from extreme points toward the knee of the Pareto front — unlike a
    /// fixed weight vector, which commits to whatever region its skew selects.
    ///
    /// Arguments:
    /// - `weights0`: strictly positive initial weights (any scale; normalised
    ///   internally);
    /// - `iterations`: number of solve/adapt rounds (`>= 1`);
    /// - `eta`: adaptation rate (`0` disables adaptation; `0.25 ..= 1.0` is a
    ///   sensible range);
    /// - `rho`: augmentation coefficient passed to
    ///   [`Self::augmented_tchebycheff_model`] (excludes weakly dominated
    ///   points).
    ///
    /// The ideal/nadir reference points come from [`Self::payoff_table`].
    /// Among the visited points the one with the smallest maximum normalised
    /// deviation is returned together with its full objective vector — so the
    /// result is monotone in the balanced-achievement sense even if the final
    /// iterate overshoots.
    ///
    /// Deterministic: no randomness anywhere in the loop.
    pub fn solve_weighted_tchebycheff_adaptive(
        &self,
        weights0: &[f64],
        iterations: usize,
        eta: f64,
        rho: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), OptError> {
        assert_eq!(weights0.len(), self.objectives.len());
        assert!(weights0.iter().all(|&w| w > 0.0), "initial weights must be strictly positive");
        assert!(iterations >= 1, "need at least one iteration");
        assert!(eta >= 0.0, "adaptation rate must be non-negative");
        assert!(rho > 0.0, "augmentation coefficient must be positive");

        let m = self.objectives.len();
        let n = self.num_vars();
        let (z, nadir) = self.payoff_table()?;
        // Normalisation ranges, guarded against flat objectives.
        let range: Vec<f64> = (0..m).map(|j| (nadir[j] - z[j]).max(1e-12)).collect();

        // Start from a normalised copy of the initial weights.
        let total: f64 = weights0.iter().sum();
        let mut w: Vec<f64> = weights0.iter().map(|&wi| wi / total).collect();

        let mut best: Option<(f64, Vec<f64>, Vec<f64>)> = None;
        for _ in 0..iterations {
            let model = self.augmented_tchebycheff_model(&w, &z, rho);
            let mut solver = MilpSolver::new();
            let sol = solver.solve(&model)?;
            let x = sol.primal[..n].to_vec();
            let f = self.evaluate(&x);

            // Balanced-achievement score: max normalised deviation.
            let dev: Vec<f64> = (0..m).map(|j| (f[j] - z[j]) / range[j]).collect();
            let ach = dev.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if best.is_none() || ach < best.as_ref().unwrap().0 {
                best = Some((ach, x, f));
            }

            // Multiplicative weight adaptation toward balance.
            let mean = dev.iter().sum::<f64>() / m as f64;
            for j in 0..m {
                w[j] *= (eta * (dev[j] - mean)).exp();
            }
            let total: f64 = w.iter().sum();
            if total <= 0.0 || !total.is_finite() {
                w = vec![1.0 / m as f64; m]; // numerical safety net
            } else {
                for wj in w.iter_mut() {
                    *wj /= total;
                }
            }
        }

        let (_, x, f) = best.expect("at least one iteration ran");
        Ok((x, f))
    }

    /// Convenience wrapper for [`Self::solve_weighted_tchebycheff_adaptive`]
    /// with uniform initial weights and default dynamics (8 iterations,
    /// `eta = 0.5`, `rho = 1e-2`).
    pub fn solve_adaptive_tchebycheff(&self) -> Result<(Vec<f64>, Vec<f64>), OptError> {
        let m = self.objectives.len();
        self.solve_weighted_tchebycheff_adaptive(&vec![1.0; m], 8, 0.5, 1e-2)
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

/// Solve an adaptive weighted-Tchebycheff scalarisation and return the
/// resulting objective vector.
///
/// Convenience wrapper around
/// [`LinearMultiObjective::solve_weighted_tchebycheff_adaptive`] with uniform
/// initial weights.
pub fn adaptive_tchebycheff(prob: &LinearMultiObjective) -> Result<Vec<f64>, OptError> {
    let (_, f) = prob.solve_adaptive_tchebycheff()?;
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

    #[test]
    fn payoff_table_ideal_and_nadir_on_coupled_front() {
        // Coupled front f0 = a, f1 = 1 - a on [0,1]^2: minimising f0 pins
        // a = 0 giving the row (0, 1); minimising f1 pins a = 1 giving (1, 0).
        // Ideal is therefore (0, 0) and the pay-off-table nadir is (1, 1),
        // both uniquely determined (no degenerate vertex choice).
        let prob = LinearMultiObjective::new(
            vec![term(vec![1.0, 0.0], 0.0), term(vec![-1.0, 0.0], 1.0)],
            vec![(0.0, 1.0), (0.0, 1.0)],
        );
        let (z, nadir) = prob.payoff_table().unwrap();
        assert!(z[0].abs() < 1e-6 && z[1].abs() < 1e-6);
        assert!((nadir[0] - 1.0).abs() < 1e-6 && (nadir[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn adaptive_tchebycheff_escapes_skewed_extreme() {
        // Coupled front f0 = a, f1 = 1 - a on [0,1]^2. A heavily skewed fixed
        // Tchebycheff weight (999, 1) pins the solution near the f0-ideal
        // extreme (a ~ 0.001). The adaptive loop must rebalance and move
        // substantially toward the interior/knee.
        let coupled = LinearMultiObjective::new(
            vec![term(vec![1.0, 0.0], 0.0), term(vec![-1.0, 0.0], 1.0)],
            vec![(0.0, 1.0), (0.0, 1.0)],
        );
        let (_, f_fixed) = coupled.solve_tchebycheff(&[999.0, 1.0], &[0.0, 0.0]).unwrap();
        assert!(f_fixed[0] < 0.05, "fixed skewed weights should sit at the extreme");

        // The weight ratio must shrink by ~e^{-eta} per round while the point
        // sits at the extreme, so crossing to balance takes ~ln(999)/eta
        // rounds; 16 rounds at eta = 1 comfortably suffice, and the
        // best-achievement tracker holds the most balanced iterate visited.
        let (_, f_adaptive) =
            coupled.solve_weighted_tchebycheff_adaptive(&[999.0, 1.0], 16, 1.0, 1e-2).unwrap();
        assert!(
            f_adaptive[0] > 0.2 && f_adaptive[1] > 0.2,
            "adaptive run should leave the extreme, got {f_adaptive:?}"
        );
        // And it must remain on the Pareto front of this coupled problem.
        assert!((f_adaptive[0] + f_adaptive[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn adaptive_tchebycheff_is_deterministic() {
        let prob = LinearMultiObjective::new(
            vec![
                term(vec![1.0, 2.0, 0.5], 1.0),
                term(vec![3.0, 1.0, 1.0], 0.5),
                term(vec![0.5, 0.5, 2.0], 0.25),
            ],
            vec![(0.0, 4.0), (0.0, 4.0), (0.0, 4.0)],
        );
        let a = prob.solve_weighted_tchebycheff_adaptive(&[2.0, 1.0, 1.0], 8, 0.5, 1e-2).unwrap();
        let b = prob.solve_weighted_tchebycheff_adaptive(&[2.0, 1.0, 1.0], 8, 0.5, 1e-2).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn adaptive_tchebycheff_zero_eta_matches_plain_augmented() {
        // With eta = 0 the weights never change, so the adaptive loop reduces
        // to repeated augmented-Tchebycheff solves with fixed weights; the
        // returned point must coincide with a direct augmented solve using
        // the same (normalised) weights and payoff-table ideal.
        let prob = LinearMultiObjective::new(
            vec![term(vec![1.0, 0.0], 0.0), term(vec![0.0, 1.0], 0.0)],
            vec![(0.0, 2.0), (0.0, 2.0)],
        );
        let (x_ada, _) =
            prob.solve_weighted_tchebycheff_adaptive(&[3.0, 1.0], 4, 0.0, 1e-2).unwrap();
        let (z, _) = prob.payoff_table().unwrap();
        let (x_fix, _) = prob.solve_augmented_tchebycheff(&[0.75, 0.25], &z, 1e-2).unwrap();
        for (a, b) in x_ada.iter().zip(x_fix.iter()) {
            assert!((a - b).abs() < 1e-6, "eta=0 must reduce to the fixed-weight solve");
        }
    }

    #[test]
    #[should_panic(expected = "strictly positive")]
    fn adaptive_tchebycheff_rejects_nonpositive_initial_weights() {
        let prob = LinearMultiObjective::new(
            vec![term(vec![1.0, 0.0], 0.0), term(vec![0.0, 1.0], 0.0)],
            vec![(0.0, 2.0), (0.0, 2.0)],
        );
        let _ = prob.solve_weighted_tchebycheff_adaptive(&[1.0, 0.0], 4, 0.5, 1e-2);
    }
}
