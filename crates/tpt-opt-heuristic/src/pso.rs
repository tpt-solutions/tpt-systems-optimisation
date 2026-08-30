//! Particle swarm optimization (PSO) over continuous space.
//!
//! Particles fly through `dim`-dimensional box-bounded space, guided by their
//! personal best and a neighbourhood best. Inertia weight adapts over time, and
//! the neighbourhood topology ([`Topology`]) can be global (gbest), a local ring
//! (lbest), or a Von Neumann lattice.

use tpt_math_prob::sampler::SplitMix64;
use tpt_opt_core::{
    Model, OptError, Sense, Solution, SolveParameters, Solver, SolverStatus, WarmStart,
};

use crate::history::ConvergenceHistory;
use crate::problem::{random_point, Objective};
use crate::result::HeuristicResult;
use crate::rng::Rng;
use crate::rng::RngExt;
use crate::ModelObjective;

/// Inertia-weight schedule for PSO.
#[derive(Debug, Clone, Copy)]
pub enum InertiaSchedule {
    /// Constant inertia weight `w`.
    Constant(f64),
    /// Linearly decrease from `start` to `end` across the run.
    Linear {
        /// Start weight.
        start: f64,
        /// End weight.
        end: f64,
    },
    /// Adapt `w` each iteration based on whether the global best improved.
    Adaptive {
        /// Start weight.
        start: f64,
        /// Minimum weight floor.
        min: f64,
        /// Multiplicative up/down factor (`> 1`).
        adapt: f64,
    },
}

impl InertiaSchedule {
    /// Linear schedule `1.0 -> 0.4`.
    pub fn linear(start: f64, end: f64) -> Self {
        InertiaSchedule::Linear { start, end }
    }

    /// Adaptive schedule with sensible defaults.
    pub fn adaptive(start: f64) -> Self {
        InertiaSchedule::Adaptive { start, min: 0.1, adapt: 1.02 }
    }
}

/// Neighbourhood topology connecting particles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Topology {
    /// Every particle sees the global best (gbest).
    Global,
    /// Each particle sees its two ring neighbours (lbest).
    Ring,
    /// Each particle sees its 4 Von Neumann lattice neighbours.
    VonNeumann,
}

/// Particle swarm optimization solver.
///
/// Determinism: seeded via [`ParticleSwarmOptimization::with_seed`].
pub struct ParticleSwarmOptimization {
    objective: Box<dyn Objective>,
    swarm_size: usize,
    iterations: usize,
    inertia: InertiaSchedule,
    c1: f64,
    c2: f64,
    vmax_frac: f64,
    topology: Topology,
    initial: Option<Vec<f64>>,
    target: Option<f64>,
    seed: u64,
    rng: SplitMix64,
    history: ConvergenceHistory,
}

impl ParticleSwarmOptimization {
    /// Build a PSO for `objective` with defaults.
    pub fn new(objective: impl Objective + 'static) -> Self {
        Self {
            objective: Box::new(objective),
            swarm_size: 40,
            iterations: 1000,
            inertia: InertiaSchedule::linear(0.9, 0.4),
            c1: 1.5,
            c2: 1.5,
            vmax_frac: 0.2,
            topology: Topology::Global,
            initial: None,
            target: None,
            seed: 0,
            rng: SplitMix64::seed_from_u64(0),
            history: ConvergenceHistory::new(),
        }
    }

    /// Set swarm size.
    pub fn with_swarm_size(mut self, n: usize) -> Self {
        self.swarm_size = n;
        self
    }

    /// Set iteration count.
    pub fn with_iterations(mut self, n: usize) -> Self {
        self.iterations = n;
        self
    }

    /// Set the inertia schedule.
    pub fn with_inertia(mut self, inertia: InertiaSchedule) -> Self {
        self.inertia = inertia;
        self
    }

    /// Set cognitive (`c1`) and social (`c2`) acceleration coefficients.
    pub fn with_acceleration(mut self, c1: f64, c2: f64) -> Self {
        self.c1 = c1;
        self.c2 = c2;
        self
    }

    /// Set the maximum velocity as a fraction of each coordinate's span.
    pub fn with_vmax_fraction(mut self, frac: f64) -> Self {
        self.vmax_frac = frac.clamp(0.0, 1.0);
        self
    }

    /// Set the neighbourhood topology.
    pub fn with_topology(mut self, topology: Topology) -> Self {
        self.topology = topology;
        self
    }

    /// Warm-start from an explicit initial point.
    pub fn with_initial(mut self, initial: Vec<f64>) -> Self {
        self.initial = Some(initial);
        self
    }

    /// Stop early (reporting [`SolverStatus::Optimal`])
    /// once the incumbent reaches `target`.
    pub fn with_target(mut self, target: f64) -> Self {
        self.target = Some(target);
        self
    }

    /// Set the deterministic seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.rng = SplitMix64::seed_from_u64(seed);
        self
    }

    /// Convergence history from the last [`solve`](Self::solve).
    pub fn history(&self) -> &ConvergenceHistory {
        &self.history
    }

    /// Run PSO.
    pub fn solve(&mut self) -> Result<HeuristicResult, OptError> {
        if self.swarm_size == 0 {
            return Err(OptError::invalid_model("swarm size must be > 0"));
        }
        if self.iterations == 0 {
            return Err(OptError::invalid_model("iterations must be > 0"));
        }
        let dim = self.objective.dim();
        if dim == 0 {
            return Err(OptError::invalid_model("objective dimension is zero"));
        }
        let bounds: Vec<(f64, f64)> = (0..dim).map(|i| self.objective.bound(i)).collect();
        let sense = self.objective.sense();
        let better = |a: f64, b: f64| match sense {
            Sense::Minimize => a < b,
            Sense::Maximize => a > b,
        };
        let eval = |x: &[f64]| self.objective.evaluate(x);

        let span = |i: usize| -> f64 {
            let (lo, hi) = bounds[i];
            if lo.is_finite() && hi.is_finite() {
                (hi - lo).abs()
            } else {
                1.0
            }
        };
        let vmax = |i: usize| -> f64 { self.vmax_frac * span(i) };

        let mut pos: Vec<Vec<f64>> = Vec::with_capacity(self.swarm_size);
        let mut vel: Vec<Vec<f64>> = Vec::with_capacity(self.swarm_size);
        for _ in 0..self.swarm_size {
            let p = self
                .initial
                .clone()
                .unwrap_or_else(|| random_point(&*self.objective, &mut self.rng));
            let v = (0..dim)
                .map(|i| {
                    let (lo, hi) = bounds[i];
                    let a = if lo.is_finite() { lo } else { -1.0 };
                    let b = if hi.is_finite() { hi } else { 1.0 };
                    self.rng.range(a, b) * self.vmax_frac
                })
                .collect();
            pos.push(p);
            vel.push(v);
        }

        let mut pbest_pos = pos.clone();
        let mut pbest_val: Vec<f64> = pos.iter().map(|p| eval(p)).collect();

        let neighbors = build_topology(self.topology, self.swarm_size);
        let mut nbest_pos = pbest_pos.clone();
        let mut nbest_val = pbest_val.clone();
        for i in 0..self.swarm_size {
            for &j in &neighbors[i] {
                if better(pbest_val[j], nbest_val[i]) {
                    nbest_val[i] = pbest_val[j];
                    nbest_pos[i] = pbest_pos[j].clone();
                }
            }
        }

        let mut best_i = 0;
        for i in 1..self.swarm_size {
            if better(nbest_val[i], nbest_val[best_i]) {
                best_i = i;
            }
        }
        let mut best_pos = nbest_pos[best_i].clone();
        let mut best_val = nbest_val[best_i];

        let mut w_state = match self.inertia {
            InertiaSchedule::Constant(w) => w,
            InertiaSchedule::Linear { start, .. } => start,
            InertiaSchedule::Adaptive { start, .. } => start,
        };

        let mut history = ConvergenceHistory::new();

        for iter in 0..self.iterations {
            let w = match self.inertia {
                InertiaSchedule::Constant(w) => w,
                InertiaSchedule::Linear { start, end } => {
                    if self.iterations <= 1 {
                        start
                    } else {
                        start + (end - start) * (iter as f64) / (self.iterations as f64 - 1.0)
                    }
                }
                InertiaSchedule::Adaptive { .. } => w_state,
            };

            let prev_best = best_val;
            for i in 0..self.swarm_size {
                for d in 0..dim {
                    let r1 = self.rng.next_f64();
                    let r2 = self.rng.next_f64();
                    let mut v = w * vel[i][d]
                        + self.c1 * r1 * (pbest_pos[i][d] - pos[i][d])
                        + self.c2 * r2 * (nbest_pos[i][d] - pos[i][d]);
                    let vm = vmax(d);
                    if v > vm {
                        v = vm;
                    } else if v < -vm {
                        v = -vm;
                    }
                    vel[i][d] = v;
                    let mut x = pos[i][d] + v;
                    let (lo, hi) = bounds[d];
                    if lo.is_finite() {
                        x = x.max(lo);
                    }
                    if hi.is_finite() {
                        x = x.min(hi);
                    }
                    pos[i][d] = x;
                }
                let v = eval(&pos[i]);
                if better(v, pbest_val[i]) {
                    pbest_val[i] = v;
                    pbest_pos[i] = pos[i].clone();
                }
            }

            for i in 0..self.swarm_size {
                for &j in &neighbors[i] {
                    if better(pbest_val[j], nbest_val[i]) {
                        nbest_val[i] = pbest_val[j];
                        nbest_pos[i] = pbest_pos[j].clone();
                    }
                }
            }

            for i in 0..self.swarm_size {
                if better(nbest_val[i], best_val) {
                    best_val = nbest_val[i];
                    best_pos = nbest_pos[i].clone();
                }
            }

            if let InertiaSchedule::Adaptive { min, adapt, .. } = self.inertia {
                if better(best_val, prev_best) {
                    w_state = (w_state * adapt).max(min);
                } else {
                    w_state = (w_state / adapt).min(self.inertia.start_val());
                }
            }

            history.push(iter, best_val, nbest_val[best_i]);
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
            best_x: best_pos,
            best_value: best_val,
            status,
            iterations: self.iterations,
            seed: self.seed,
            history,
        })
    }
}

/// Solver-agnostic adapter: run PSO over a canonical [`Model`] by temporarily
/// installing a [`ModelObjective`] view of the model.
impl Solver<Model> for ParticleSwarmOptimization {
    fn solve(&mut self, model: &Model) -> Result<Solution, OptError> {
        let original =
            std::mem::replace(&mut self.objective, Box::new(ModelObjective::new(model.clone())));
        let result = ParticleSwarmOptimization::solve(self);
        self.objective = original;
        result.map(|r| r.solution())
    }

    fn set_parameter(&mut self, param: &SolveParameters) -> Result<(), OptError> {
        if let Some(seed) = param.seed {
            self.seed = seed;
            self.rng = SplitMix64::seed_from_u64(seed);
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

impl InertiaSchedule {
    fn start_val(self) -> f64 {
        match self {
            InertiaSchedule::Constant(w) => w,
            InertiaSchedule::Linear { start, .. } => start,
            InertiaSchedule::Adaptive { start, .. } => start,
        }
    }
}

/// Build neighbour index lists for a topology.
fn build_topology(topology: Topology, swarm: usize) -> Vec<Vec<usize>> {
    match topology {
        Topology::Global => {
            let all: Vec<usize> = (0..swarm).collect();
            vec![all; swarm]
        }
        Topology::Ring => {
            (0..swarm)
                .map(|i| {
                    if swarm <= 1 {
                        vec![i]
                    } else {
                        vec![(i + swarm - 1) % swarm, (i + 1) % swarm]
                    }
                })
                .collect()
        }
        Topology::VonNeumann => {
            if swarm == 0 {
                return Vec::new();
            }
            let cols = (swarm as f64).sqrt().floor().max(1.0) as usize;
            let rows = swarm.div_ceil(cols);
            let at = |r: usize, c: usize| -> usize {
                let rr = r % rows;
                let cc = c % cols;
                rr * cols + cc
            };
            (0..swarm)
                .map(|p| {
                    let r = p / cols;
                    let c = p % cols;
                    let mut nbrs = Vec::with_capacity(4);
                    nbrs.push(at(r, c + 1));
                    nbrs.push(at(r, c + cols - 1));
                    nbrs.push(at(r + 1, c));
                    nbrs.push(at(r + rows - 1, c));
                    nbrs.sort_unstable();
                    nbrs.dedup();
                    nbrs.retain(|&x| x < swarm);
                    if nbrs.is_empty() {
                        nbrs.push(p);
                    }
                    nbrs
                })
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ObjectiveFn;

    #[test]
    fn determinism_same_seed() {
        let obj =
            ObjectiveFn::minimize(3, |x| x.iter().map(|v| v * v).sum::<f64>(), [(-3.0, 3.0); 3]);
        let build =
            || ParticleSwarmOptimization::new(obj.clone()).with_seed(55).with_iterations(500);
        let a = build().solve().unwrap();
        let b = build().solve().unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn converges_on_sphere() {
        let obj =
            ObjectiveFn::minimize(5, |x| x.iter().map(|v| v * v).sum::<f64>(), [(-2.0, 2.0); 5]);
        let mut pso = ParticleSwarmOptimization::new(obj)
            .with_seed(8)
            .with_swarm_size(50)
            .with_iterations(800)
            .with_inertia(InertiaSchedule::linear(0.9, 0.3));
        let res = pso.solve().unwrap();
        assert!(res.best_value < 0.1, "got {}", res.best_value);
    }

    #[test]
    fn ring_and_von_neumann_topologies() {
        let obj = ObjectiveFn::minimize(
            2,
            |x| (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2),
            [(0.0, 2.0); 2],
        );
        for topo in [Topology::Ring, Topology::VonNeumann] {
            let mut pso = ParticleSwarmOptimization::new(obj.clone())
                .with_seed(2)
                .with_topology(topo)
                .with_iterations(600);
            let res = pso.solve().unwrap();
            assert!(res.best_value < 0.5, "{topo:?} got {}", res.best_value);
        }
    }
}
