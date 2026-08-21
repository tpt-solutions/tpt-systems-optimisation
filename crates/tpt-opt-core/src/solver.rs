//! Solver trait contract, status, parameters, warm-start, and solution.
//!
//! [`Solver`] is the agnosticism boundary (spec §4): every solver crate's
//! primary solve type implements `Solver<M>` with a *consistent* signature —
//! `solve` / `set_parameter` / `warm_start` / `status` / `solution`. `M` is the
//! native model representation, conventionally [`crate::model::Model`].

use alloc::vec;
use alloc::vec::Vec;

use crate::error::OptError;
use crate::tolerance::Tolerances;

/// Terminal status reported by a solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverStatus {
    /// A proven optimal solution was found.
    Optimal,
    /// The model has no feasible solution.
    Infeasible,
    /// The feasible region is unbounded.
    Unbounded,
    /// Termination due to the time limit (solution may still be usable).
    TimeLimit,
    /// Numerical failure (cycling, stalling, ill-conditioning).
    NumericalIssue,
    /// Any other failure.
    Error,
}

impl SolverStatus {
    /// `true` if a usable solution is available (optimal or time-limited).
    pub fn has_solution(&self) -> bool {
        matches!(self, SolverStatus::Optimal | SolverStatus::TimeLimit)
    }
}

/// Verbosity level for solver logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    /// No output.
    Quiet,
    /// Summary output.
    Normal,
    /// Detailed output.
    Verbose,
}

/// Solver parameter bundle (time limit, gap, threads, seed, tolerances, ...).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveParameters {
    /// Wall-clock time limit in seconds (`None` = no limit).
    pub time_limit: Option<f64>,
    /// Absolute optimality gap tolerance.
    pub absolute_gap: f64,
    /// Relative optimality gap tolerance.
    pub relative_gap: f64,
    /// Worker thread count for parallel solvers (`0`/`1` = sequential).
    pub threads: usize,
    /// Verbosity level.
    pub verbosity: Verbosity,
    /// Deterministic seed for branching/heuristics/parallel work distribution.
    pub seed: Option<u64>,
    /// Numeric tolerances (see [`Tolerances`]).
    pub tolerances: Tolerances,
}

impl SolveParameters {
    /// Sensible defaults: no time limit, spec gap 1e-4, single-threaded, quiet.
    pub fn defaults() -> Self {
        Self {
            time_limit: None,
            absolute_gap: 1e-4,
            relative_gap: 1e-4,
            threads: 1,
            verbosity: Verbosity::Quiet,
            seed: None,
            tolerances: Tolerances::spec_default(),
        }
    }

    /// Set the wall-clock time limit (seconds).
    pub fn with_time_limit(mut self, seconds: f64) -> Self {
        self.time_limit = Some(seconds);
        self
    }

    /// Set the worker thread count.
    pub fn with_threads(mut self, n: usize) -> Self {
        self.threads = n;
        self
    }

    /// Set absolute and relative optimality gap tolerances.
    pub fn with_gap(mut self, absolute: f64, relative: f64) -> Self {
        self.absolute_gap = absolute;
        self.relative_gap = relative;
        self
    }

    /// Set the verbosity level.
    pub fn with_verbosity(mut self, v: Verbosity) -> Self {
        self.verbosity = v;
        self
    }

    /// Set the deterministic seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Replace the tolerance bundle.
    pub fn with_tolerances(mut self, t: Tolerances) -> Self {
        self.tolerances = t;
        self
    }
}

impl Default for SolveParameters {
    fn default() -> Self {
        Self::defaults()
    }
}

/// Warm-start data for reusing a previous solve.
#[derive(Debug, Clone, PartialEq)]
pub struct WarmStart {
    /// Previously known primal values (optionally partial).
    pub primal: Option<Vec<f64>>,
    /// Previously known dual values (one per constraint).
    pub dual: Option<Vec<f64>>,
    /// Previously known status hint.
    pub status: Option<SolverStatus>,
}

impl WarmStart {
    /// Warm-start from a primal point only.
    pub fn primal(values: Vec<f64>) -> Self {
        Self { primal: Some(values), dual: None, status: None }
    }

    /// Warm-start from dual values only.
    pub fn dual(values: Vec<f64>) -> Self {
        Self { primal: None, dual: Some(values), status: None }
    }

    /// An empty (no-op) warm start.
    pub fn empty() -> Self {
        Self { primal: None, dual: None, status: None }
    }

    /// `true` if there is nothing to reuse.
    pub fn is_empty(&self) -> bool {
        self.primal.is_none() && self.dual.is_none()
    }
}

/// Extracted solution: primal/dual/reduced-cost/slack vectors plus metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    /// Primal variable values.
    pub primal: Vec<f64>,
    /// Dual values, one per constraint.
    pub dual: Vec<f64>,
    /// Reduced costs, one per variable.
    pub reduced_costs: Vec<f64>,
    /// Constraint slacks, one per constraint.
    pub slacks: Vec<f64>,
    /// Objective value at `primal`.
    pub objective_value: f64,
    /// Terminal status.
    pub status: SolverStatus,
    /// Iteration / node count, if reported by the solver.
    pub iterations: Option<usize>,
    /// Elapsed solve time in seconds, if reported by the solver.
    pub solve_time: Option<f64>,
}

impl Solution {
    /// Construct a solution from a primal point and objective value.
    pub fn new(primal: Vec<f64>, objective_value: f64, status: SolverStatus) -> Self {
        let n = primal.len();
        Self {
            primal,
            dual: Vec::new(),
            reduced_costs: vec![0.0; n],
            slacks: Vec::new(),
            objective_value,
            status,
            iterations: None,
            solve_time: None,
        }
    }

    /// Attach dual values.
    pub fn with_dual(mut self, dual: Vec<f64>) -> Self {
        self.dual = dual;
        self
    }

    /// Attach reduced costs.
    pub fn with_reduced_costs(mut self, rc: Vec<f64>) -> Self {
        self.reduced_costs = rc;
        self
    }

    /// Attach constraint slacks.
    pub fn with_slacks(mut self, slacks: Vec<f64>) -> Self {
        self.slacks = slacks;
        self
    }

    /// Attach iteration/node count.
    pub fn with_iterations(mut self, iters: usize) -> Self {
        self.iterations = Some(iters);
        self
    }

    /// Attach solve time.
    pub fn with_solve_time(mut self, t: f64) -> Self {
        self.solve_time = Some(t);
        self
    }

    /// Primal value of variable `i`, if in range.
    pub fn primal_value(&self, i: usize) -> Option<f64> {
        self.primal.get(i).copied()
    }
}

/// The core solver contract. Implementors provide a consistent agnostic
/// interface over their native model type `M`.
pub trait Solver<M> {
    /// Solve `model`, returning the extracted [`Solution`].
    fn solve(&mut self, model: &M) -> Result<Solution, OptError>;

    /// Apply a parameter bundle (may be partially rejected by the solver).
    fn set_parameter(&mut self, param: &SolveParameters) -> Result<(), OptError>;

    /// Seed the solver with a previous solution for reuse.
    fn warm_start(&mut self, warm: WarmStart) -> Result<(), OptError>;

    /// Current terminal status (after `solve`).
    fn status(&self) -> SolverStatus;

    /// Last extracted solution, if any.
    fn solution(&self) -> Option<Solution>;
}
