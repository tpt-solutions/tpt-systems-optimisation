//! Metaheuristic optimization algorithms for the `tpt-systems-optimisation` workspace.
//!
//! `tpt-opt-heuristic` provides deterministic, seedable metaheuristics for
//! problems where exact methods are too slow or where the landscape is
//! non-convex with no good relaxation (spec.txt, section 3, "tpt-opt-heuristic").
//!
//! # Algorithms
//!
//! * **Simulated annealing** ([`annealing::SimulatedAnnealing`]) with geometric,
//!   adaptive, and reheating cooling schedules and configurable, user-supplied
//!   neighborhoods.
//! * **Genetic algorithms** ([`ga::GeneticAlgorithm`]) over continuous
//!   [`Vec<f64>`](std::vec::Vec) *and* permutation [`Vec<usize>`](std::vec::Vec)
//!   genomes, with single-point / two-point / uniform / order-based crossover,
//!   bit-flip / flip / swap / inversion / scramble mutation, and tournament /
//!   roulette / rank selection.
//! * **Tabu search** ([`tabu::TabuSearch`]) with adaptive tenure, aspiration,
//!   and diversification / intensification.
//! * **Particle swarm optimization** ([`pso::ParticleSwarmOptimization`]) with
//!   inertia-weight adaptation and global / ring / Von Neumann topologies.
//!
//! # Reproducibility (spec, section 4)
//!
//! Every heuristic accepts a deterministic seed via [`rng::SplitMix64`] (re-exported
//! from `tpt-math-prob`). Two runs with the same seed produce byte-identical
//! results. Custom neighborhoods and operator closures are supported through
//! trait objects.
//!
//! # Tying into `tpt-opt-core`
//!
//! Results are reported through [`result::HeuristicResult`], which can be turned
//! into a [`tpt_opt_core::Solution`] via [`result::HeuristicResult::solution`].
//! Invalid configurations surface as [`tpt_opt_core::OptError`], and the
//! terminal condition is summarized with a [`tpt_opt_core::SolverStatus`].
//! [`annealing::SimulatedAnnealing`] additionally implements
//! [`tpt_opt_core::Solver`] for [`tpt_opt_core::Model`] so it can be driven
//! through the common solver contract.
//!
//! # Example
//!
//! ```rust
//! use tpt_opt_heuristic::{SimulatedAnnealing, ObjectiveFn, CoolingSchedule};
//!
//! let obj = ObjectiveFn::minimize(2, |x| x[0]*x[0] + x[1]*x[1], [(-5.0, 5.0), (-5.0, 5.0)]);
//! let mut sa = SimulatedAnnealing::new(obj).with_seed(7);
//! let res = sa.solve().unwrap();
//! assert!(res.best_value < 1.0);
//! ```

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

pub mod annealing;
pub mod ga;
pub mod history;
pub mod neighborhood;
pub mod problem;
pub mod pso;
pub mod result;
pub mod rng;
pub mod tabu;

pub use annealing::{CoolingSchedule, SimulatedAnnealing};
pub use ga::{
    crossover, mutate, order_based, select_index, single_point, two_point, uniform, CrossoverKind,
    GaSetup, Gene, GeneticAlgorithm, Genome, MutationKind, SelectionKind,
};
pub use history::ConvergenceHistory;
pub use neighborhood::{GaussianNeighborhood, Neighborhood, NeighborhoodFn, TabuNeighborhood};
pub use problem::{ModelObjective, Objective, ObjectiveFn};
pub use pso::{InertiaSchedule, ParticleSwarmOptimization, Topology};
pub use result::HeuristicResult;
pub use rng::{rng_from_seed, Rng, SplitMix64};
pub use tabu::TabuSearch;
