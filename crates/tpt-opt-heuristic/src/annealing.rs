//! Simulated annealing (SA) with geometric / adaptive / reheating schedules.
//!
//! SA minimises or maximises an [`Objective`](crate::Objective) by a Metropolis
//! acceptance rule over neighbours produced by a [`Neighborhood`](crate::Neighborhood).
//! Cooling schedules are configurable and the whole search is deterministic for
//! a fixed seed.

use tpt_math_prob::Xoshiro256;
use tpt_opt_core::{
    Model, OptError, Sense, Solution, SolveParameters, Solver, SolverStatus, WarmStart,
};

use crate::history::ConvergenceHistory;
use crate::neighborhood::{GaussianNeighborhood, Neighborhood};
use crate::problem::{random_point, ModelObjective, Objective};
use crate::result::HeuristicResult;
use crate::rng::Rng;

/// Cooling schedule for simulated annealing.
#[derive(Debug, Clone, Copy)]
pub enum CoolingSchedule {
    /// `T(k) = T0 * alpha^k`. `alpha` in `(0, 1)`.
    Geometric {
        /// Initial temperature.
        initial_temp: f64,
        /// Per-iteration multiplicative decay.
        alpha: f64,
    },
    /// Acceptance-ratio driven: raise temperature when too many moves are
    /// accepted (under-exploring), lower it when too few are accepted
    /// (over-accepting or stagnating).
    Adaptive {
        /// Initial temperature.
        initial_temp: f64,
        /// Floor temperature.
        min_temp: f64,
        /// Target acceptance ratio in a window.
        accept_target: f64,
        /// Multiplicative up/down factor (should be `> 1`).
        adapt_rate: f64,
    },
    /// Geometric cooling that periodically reheats back to `reheat_temp`.
    Reheating {
        /// Initial temperature.
        initial_temp: f64,
        /// Per-iteration multiplicative decay.
        alpha: f64,
        /// Temperature restored every `reheat_every` iterations.
        reheat_temp: f64,
        /// Reheat period (iterations).
        reheat_every: usize,
    },
}

impl CoolingSchedule {
    /// Geometric schedule with `T0 = 1.0`, `alpha = 0.95`.
    pub fn geometric(initial_temp: f64, alpha: f64) -> Self {
        CoolingSchedule::Geometric { initial_temp, alpha }
    }

    /// Adaptive schedule with sensible defaults.
    pub fn adaptive(initial_temp: f64) -> Self {
        CoolingSchedule::Adaptive {
            initial_temp,
            min_temp: 1e-6,
            accept_target: 0.44,
            adapt_rate: 1.02,
        }
    }

    /// Reheating schedule with sensible defaults.
    pub fn reheating(initial_temp: f64, alpha: f64, reheat_every: usize) -> Self {
        CoolingSchedule::Reheating { initial_temp, alpha, reheat_temp: initial_temp, reheat_every }
    }
}

/// Internal cooling-state machine driving [`CoolingSchedule`].
struct CoolingState {
    schedule: CoolingSchedule,
    iter: usize,
    temp: f64,
}

impl CoolingState {
    fn new(schedule: &CoolingSchedule) -> Self {
        let temp = match schedule {
            CoolingSchedule::Geometric { initial_temp, .. } => *initial_temp,
            CoolingSchedule::Adaptive { initial_temp, .. } => *initial_temp,
            CoolingSchedule::Reheating { initial_temp, .. } => *initial_temp,
        };
        Self { schedule: *schedule, iter: 0, temp }
    }

    fn next(&mut self, accept_ratio: f64) -> f64 {
        self.iter += 1;
        match self.schedule {
            CoolingSchedule::Geometric { initial_temp, alpha } => {
                self.temp = initial_temp * alpha.powi(self.iter as i32);
            }
            CoolingSchedule::Adaptive { min_temp, accept_target, adapt_rate, .. } => {
                if accept_ratio > accept_target {
                    self.temp = (self.temp / adapt_rate).max(min_temp);
                } else {
                    self.temp = (self.temp * adapt_rate).max(min_temp);
                }
            }
            CoolingSchedule::Reheating { initial_temp, alpha, reheat_temp, reheat_every } => {
                if reheat_every > 0 && self.iter % reheat_every == 0 {
                    self.temp = reheat_temp.max(initial_temp);
                } else {
                    self.temp = initial_temp * alpha.powi(self.iter as i32);
                }
            }
        }
        self.temp
    }
}

/// Simulated annealing solver.
///
/// Determinism: the RNG is seeded via [`SimulatedAnnealing::with_seed`]. Two
/// runs with the same seed produce identical [`HeuristicResult`]s.
pub struct SimulatedAnnealing {
    objective: Box<dyn Objective>,
    schedule: CoolingSchedule,
    neighborhood: Option<Box<dyn Neighborhood>>,
    iterations: usize,
    initial: Option<Vec<f64>>,
    target: Option<f64>,
    seed: u64,
    rng: Xoshiro256,
    history: ConvergenceHistory,
}

impl SimulatedAnnealing {
    /// Build an SA solver for `objective` with default schedule/neighbourhood.
    pub fn new(objective: impl Objective + 'static) -> Self {
        Self {
            objective: Box::new(objective),
            schedule: CoolingSchedule::geometric(1.0, 0.95),
            neighborhood: None,
            iterations: 1000,
            initial: None,
            target: None,
            seed: 0,
            rng: Xoshiro256::new(0),
            history: ConvergenceHistory::new(),
        }
    }

    /// Set the cooling schedule.
    pub fn with_cooling(mut self, schedule: CoolingSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    /// Set a custom neighbourhood (otherwise a Gaussian neighbourhood is built
    /// from the objective bounds).
    pub fn with_neighborhood(mut self, neighborhood: impl Neighborhood + 'static) -> Self {
        self.neighborhood = Some(Box::new(neighborhood));
        self
    }

    /// Number of Metropolis iterations.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Warm-start the search from an explicit initial point.
    pub fn with_initial(mut self, initial: Vec<f64>) -> Self {
        self.initial = Some(initial);
        self
    }

    /// Stop early (and report [`SolverStatus::Optimal`]) once the incumbent
    /// reaches `target` (≤ for minimise, ≥ for maximise).
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

    /// Retrieve the convergence history from the last [`solve`](Self::solve).
    pub fn history(&self) -> &ConvergenceHistory {
        &self.history
    }

    /// Run simulated annealing.
    pub fn solve(&mut self) -> Result<HeuristicResult, OptError> {
        if self.iterations == 0 {
            return Err(OptError::invalid_model("SA iterations must be > 0"));
        }
        if self.objective.dim() == 0 {
            return Err(OptError::invalid_model("objective dimension is zero"));
        }
        let bounds: Vec<(f64, f64)> =
            (0..self.objective.dim()).map(|i| self.objective.bound(i)).collect();
        let default_nb = GaussianNeighborhood::new(bounds, 0.1);
        let nb: &dyn Neighborhood = self.neighborhood.as_deref().unwrap_or(&default_nb);
        let result = anneal(
            &*self.objective,
            self.iterations,
            &self.schedule,
            nb,
            self.initial.clone(),
            self.target,
            self.seed,
            &mut self.rng,
        );
        self.history = result.history.clone();
        Ok(result)
    }
}

/// Core Metropolis loop, shared by the struct API and the [`Solver`] impl.
pub(crate) fn anneal(
    objective: &dyn Objective,
    iterations: usize,
    schedule: &CoolingSchedule,
    neighborhood: &dyn Neighborhood,
    initial: Option<Vec<f64>>,
    target: Option<f64>,
    seed: u64,
    rng: &mut Xoshiro256,
) -> HeuristicResult {
    let sense = objective.sense();
    let mut current = initial.unwrap_or_else(|| random_point(objective, rng));
    let mut current_val = objective.evaluate(&current);
    let mut best = current.clone();
    let mut best_val = current_val;

    let better = |a: f64, b: f64| match sense {
        Sense::Minimize => a < b,
        Sense::Maximize => a > b,
    };

    let mut cooler = CoolingState::new(schedule);
    let window = iterations.clamp(1, 100);
    let mut accepted_in_window: usize = 0;
    let mut history = ConvergenceHistory::new();

    for iter in 0..iterations {
        let candidate = neighborhood.neighbor(&current, rng);
        let cand_val = objective.evaluate(&candidate);
        let delta = match sense {
            Sense::Minimize => cand_val - current_val,
            Sense::Maximize => current_val - cand_val,
        };
        let accept =
            if delta <= 0.0 { true } else { rng.next_f64() < (-delta / cooler.temp).exp() };
        if accept {
            current = candidate;
            current_val = cand_val;
            accepted_in_window += 1;
            if better(current_val, best_val) {
                best = current.clone();
                best_val = current_val;
            }
        }
        let ratio = if iter < window {
            accepted_in_window as f64 / (iter as f64 + 1.0)
        } else {
            accepted_in_window as f64 / window as f64
        };
        cooler.next(ratio);
        if iter % window == window - 1 {
            accepted_in_window = 0;
        }
        history.push(iter, best_val, current_val);

        if let Some(t) = target {
            if better(best_val, t) || best_val == t {
                break;
            }
        }
    }

    let status = match (target, sense) {
        (Some(t), Sense::Minimize) if best_val <= t => SolverStatus::Optimal,
        (Some(t), Sense::Maximize) if best_val >= t => SolverStatus::Optimal,
        _ => SolverStatus::TimeLimit,
    };

    HeuristicResult { best_x: best, best_value: best_val, status, iterations, seed, history }
}

impl Solver<Model> for SimulatedAnnealing {
    fn solve(&mut self, model: &Model) -> Result<Solution, OptError> {
        if self.iterations == 0 {
            return Err(OptError::invalid_model("SA iterations must be > 0"));
        }
        let obj = ModelObjective::new(model.clone());
        if obj.dim() == 0 {
            return Err(OptError::invalid_model("model has zero variables"));
        }
        let bounds: Vec<(f64, f64)> = (0..obj.dim()).map(|i| obj.bound(i)).collect();
        let default_nb = GaussianNeighborhood::new(bounds, 0.1);
        let nb: &dyn Neighborhood = self.neighborhood.as_deref().unwrap_or(&default_nb);
        let result = anneal(
            &obj,
            self.iterations,
            &self.schedule,
            nb,
            self.initial.clone(),
            self.target,
            self.seed,
            &mut self.rng,
        );
        Ok(result.solution())
    }

    fn set_parameter(&mut self, param: &SolveParameters) -> Result<(), OptError> {
        if let Some(seed) = param.seed {
            self.seed = seed;
            self.rng = Xoshiro256::new(seed);
        }
        Ok(())
    }

    fn warm_start(&mut self, warm: WarmStart) -> Result<(), OptError> {
        if let Some(primal) = warm.primal {
            self.initial = Some(primal);
        }
        Ok(())
    }

    fn status(&self) -> SolverStatus {
        if self.history.is_empty() {
            SolverStatus::Error
        } else if self.target.is_some() {
            SolverStatus::Optimal
        } else {
            SolverStatus::TimeLimit
        }
    }

    fn solution(&self) -> Option<Solution> {
        if self.history.is_empty() {
            None
        } else {
            let best_val = self.history.best().unwrap_or(f64::INFINITY);
            Some(
                Solution::new(self.initial.clone().unwrap_or_default(), best_val, self.status())
                    .with_iterations(self.iterations),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ObjectiveFn;

    #[test]
    fn determinism_same_seed() {
        let build = || {
            let obj = ObjectiveFn::minimize(
                3,
                |x| x.iter().map(|v| v * v).sum::<f64>(),
                [(-5.0, 5.0); 3],
            );
            SimulatedAnnealing::new(obj).with_seed(123).with_iterations(500)
        };
        let a = build().solve().unwrap();
        let b = build().solve().unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn converges_on_sphere() {
        let obj =
            ObjectiveFn::minimize(5, |x| x.iter().map(|v| v * v).sum::<f64>(), [(-3.0, 3.0); 5]);
        let mut sa = SimulatedAnnealing::new(obj)
            .with_seed(1)
            .with_cooling(CoolingSchedule::geometric(10.0, 0.97))
            .with_iterations(4000);
        let res = sa.solve().unwrap();
        assert!(res.best_value < 0.5, "got {}", res.best_value);
    }

    #[test]
    fn reheating_schedule_works() {
        let obj = ObjectiveFn::minimize(
            2,
            |x| (x[0] - 1.0).powi(2) + (x[1] + 1.0).powi(2),
            [(-5.0, 5.0); 2],
        );
        let mut sa = SimulatedAnnealing::new(obj)
            .with_seed(9)
            .with_cooling(CoolingSchedule::reheating(5.0, 0.9, 200))
            .with_iterations(2000);
        let res = sa.solve().unwrap();
        assert!(res.best_value < 1.0);
    }

    #[test]
    fn rejects_zero_iterations() {
        let obj = ObjectiveFn::minimize(1, |x| x[0], [(-1.0, 1.0)]);
        let mut sa = SimulatedAnnealing::new(obj).with_iterations(0);
        assert!(sa.solve().is_err());
    }
}
