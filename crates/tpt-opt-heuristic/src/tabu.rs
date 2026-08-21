//! Tabu search with adaptive tenure, aspiration, and diversification.
//!
//! Tabu search explores neighbours produced by a [`TabuNeighborhood`], which
//! also reports the *move* (coordinate) that was applied. Recently applied
//! moves are forbidden (tabu) for an adaptive tenure unless they satisfy the
//! aspiration criterion (beating the global best). Long stagnation triggers
//! diversification (random restart), realising intensification/diversification.

use tpt_math_prob::Xoshiro256;
use tpt_opt_core::{OptError, Sense};

use crate::history::ConvergenceHistory;
use crate::neighborhood::{CoordinateNeighborhood, TabuNeighborhood};
use crate::problem::{random_point, Objective};
use crate::result::HeuristicResult;
use crate::rng::Rng;

/// Tabu search solver.
///
/// Determinism: seeded via [`TabuSearch::with_seed`].
pub struct TabuSearch {
    objective: Box<dyn Objective>,
    neighborhood: Option<Box<dyn TabuNeighborhood>>,
    max_iter: usize,
    tenure_base: usize,
    tenure_spread: usize,
    diversification_after: usize,
    sample_size: usize,
    initial: Option<Vec<f64>>,
    target: Option<f64>,
    seed: u64,
    rng: Xoshiro256,
    history: ConvergenceHistory,
}

impl TabuSearch {
    /// Build a tabu search for `objective` with defaults.
    pub fn new(objective: impl Objective + 'static) -> Self {
        Self {
            objective: Box::new(objective),
            neighborhood: None,
            max_iter: 1000,
            tenure_base: 7,
            tenure_spread: 7,
            diversification_after: 50,
            sample_size: 20,
            initial: None,
            target: None,
            seed: 0,
            rng: Xoshiro256::new(0),
            history: ConvergenceHistory::new(),
        }
    }

    /// Set a custom tabu neighbourhood (otherwise a coordinate-step neighbourhood
    /// is built from the objective bounds).
    pub fn with_neighborhood(mut self, nb: impl TabuNeighborhood + 'static) -> Self {
        self.neighborhood = Some(Box::new(nb));
        self
    }

    /// Maximum number of iterations.
    pub fn with_iterations(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Base tabu tenure (iterations a move stays forbidden).
    pub fn with_tenure(mut self, base: usize) -> Self {
        self.tenure_base = base;
        self
    }

    /// Random extra tenure added per move (adaptive spread).
    pub fn with_tenure_spread(mut self, spread: usize) -> Self {
        self.tenure_spread = spread;
        self
    }

    /// Iterations without improvement before diversifying (random restart).
    pub fn with_diversification_after(mut self, n: usize) -> Self {
        self.diversification_after = n;
        self
    }

    /// Number of neighbour candidates sampled per iteration.
    pub fn with_sample_size(mut self, n: usize) -> Self {
        self.sample_size = n.max(1);
        self
    }

    /// Warm-start from an explicit initial point.
    pub fn with_initial(mut self, initial: Vec<f64>) -> Self {
        self.initial = Some(initial);
        self
    }

    /// Stop early (reporting [`SolverStatus::Optimal`](tpt_opt_core::SolverStatus::Optimal))
    /// once the incumbent reaches `target`.
    pub fn with_target(mut self, target: f64) -> Self {
        self.target = Some(target);
        self
    }

    /// Set the deterministic seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.rng = Xoshiro256::new(seed);
        self
    }

    /// Convergence history from the last [`solve`](Self::solve).
    pub fn history(&self) -> &ConvergenceHistory {
        &self.history
    }

    /// Run tabu search.
    pub fn solve(&mut self) -> Result<HeuristicResult, OptError> {
        if self.max_iter == 0 {
            return Err(OptError::invalid_model("tabu max_iter must be > 0"));
        }
        if self.objective.dim() == 0 {
            return Err(OptError::invalid_model("objective dimension is zero"));
        }
        let bounds: Vec<(f64, f64)> = (0..self.objective.dim())
            .map(|i| self.objective.bound(i))
            .collect();
        let default_nb = CoordinateNeighborhood::new(bounds, 0.2);
        let nb: &dyn TabuNeighborhood = self
            .neighborhood
            .as_deref()
            .unwrap_or(&default_nb);

        let eval = |x: &[f64]| self.objective.evaluate(x);
        let sense = self.objective.sense();
        let better = |a: f64, b: f64| match sense {
            Sense::Minimize => a < b,
            Sense::Maximize => a > b,
        };

        let mut current = self.initial.clone().unwrap_or_else(|| random_point(&*self.objective, &mut self.rng));
        let mut current_val = eval(&current);
        let mut best = current.clone();
        let mut best_val = current_val;
        let mut tabu: Vec<usize> = vec![0; current.len()];
        let mut since_improve = 0usize;
        let mut history = ConvergenceHistory::new();

        for iter in 0..self.max_iter {
            let sample = self.sample_size;
            let mut chosen: Option<(Vec<f64>, f64, usize)> = None;
            let mut abs_best: Option<(Vec<f64>, f64, usize)> = None;

            for _ in 0..sample {
                let (cand, mv) = nb.neighbor_move(&current, &mut self.rng);
                let v = eval(&cand);
                let is_tabu = tabu[mv] > iter;
                let aspiration = better(v, best_val);
                if (aspiration || !is_tabu)
                    && (chosen.is_none() || better(v, chosen.as_ref().unwrap().1))
                {
                    chosen = Some((cand.clone(), v, mv));
                }
                if abs_best.is_none() || better(v, abs_best.as_ref().unwrap().1) {
                    abs_best = Some((cand, v, mv));
                }
            }

            let (cand, cand_val, mv) = chosen.or(abs_best).expect("sample >= 1");

            current = cand;
            current_val = cand_val;
            if better(current_val, best_val) {
                best = current.clone();
                best_val = current_val;
                since_improve = 0;
            } else {
                since_improve += 1;
            }

            // Adaptive tenure: longer when stagnating, plus random spread.
            let extra = if since_improve > 0 {
                since_improve.min(self.tenure_spread)
            } else {
                0
            };
            let tenure = self.tenure_base + self.rng.index(self.tenure_spread + 1) + extra;
            tabu[mv] = iter + tenure;

            if since_improve >= self.diversification_after && self.diversification_after > 0 {
                current = random_point(&*self.objective, &mut self.rng);
                current_val = eval(&current);
                since_improve = 0;
                for t in tabu.iter_mut() {
                    *t = 0;
                }
            }

            history.push(iter, best_val, current_val);
            if let Some(t) = self.target {
                if better(best_val, t) || best_val == t {
                    break;
                }
            }
        }

        let status = match (self.target, sense) {
            (Some(t), Sense::Minimize) if best_val <= t => tpt_opt_core::SolverStatus::Optimal,
            (Some(t), Sense::Maximize) if best_val >= t => tpt_opt_core::SolverStatus::Optimal,
            _ => tpt_opt_core::SolverStatus::TimeLimit,
        };

        Ok(HeuristicResult {
            best_x: best,
            best_value: best_val,
            status,
            iterations: self.max_iter,
            seed: self.seed,
            history,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ObjectiveFn;

    #[test]
    fn determinism_same_seed() {
        let obj = ObjectiveFn::minimize(3, |x| x.iter().map(|v| v * v).sum::<f64>(), [(-3.0, 3.0); 3]);
        let build = || TabuSearch::new(obj.clone()).with_seed(321).with_iterations(400);
        let a = build().solve().unwrap();
        let b = build().solve().unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn reaches_optimum_on_small_problem() {
        // Minimum of (x-2)^2 + (y+1)^2 at (2,-1).
        let obj = ObjectiveFn::minimize(
            2,
            |x| (x[0] - 2.0).powi(2) + (x[1] + 1.0).powi(2),
            [(0.0, 4.0), (-3.0, 1.0)],
        );
        let mut ts = TabuSearch::new(obj)
            .with_seed(4)
            .with_iterations(1500)
            .with_sample_size(30);
        let res = ts.solve().unwrap();
        assert!(res.best_value < 0.1, "got {}", res.best_value);
        assert!((res.best_x[0] - 2.0).abs() < 0.2);
        assert!((res.best_x[1] + 1.0).abs() < 0.2);
    }

    #[test]
    fn rejects_zero_iterations() {
        let obj = ObjectiveFn::minimize(1, |x| x[0], [(-1.0, 1.0)]);
        let mut ts = TabuSearch::new(obj).with_iterations(0);
        assert!(ts.solve().is_err());
    }
}
