//! Factor-graph least squares for SLAM-style problems: manifold variables
//! (SO(2)/SE(2)/SO(3)/SE(3) plus Euclidean), standard factors (prior,
//! between/odometry, range-bearing, pinhole reprojection), Gaussian noise
//! models with robust kernels, and sparse Levenberg-Marquardt with a
//! fill-reducing-ordered LDLᵀ inner solve.
//!
//! Jacobians are computed numerically **on the manifold** — each key is
//! perturbed through `retract` — so rotations and poses never leave their
//! group while stepping. The Hessian of the whitened system doubles as the
//! Gaussian information matrix, which [`marginal::marginal_covariance`]
//! exploits for uncertainty estimates. For bundle-adjustment-shaped
//! problems, [`solve_schur`] eliminates the point variables with a Schur
//! complement before solving the reduced camera system.
//!
//! # Example
//!
//! ```rust
//! use tpt_opt_factorgraph::factors::{BetweenFactor, PriorFactor};
//! use tpt_opt_factorgraph::graph::FactorGraph;
//! use tpt_opt_factorgraph::manifold::Manifold;
//! use tpt_opt_factorgraph::noise::NoiseModel;
//! use tpt_opt_factorgraph::solver::{solve, FactorSolverConfig};
//!
//! // Two SE(2) poses connected by a measured relative transform, anchored
//! // by a prior on the first pose.
//! let mut graph = FactorGraph::new();
//! graph.add_variable(0, Manifold::Se2, vec![0.0, 0.0, 0.0]).unwrap();
//! graph.add_variable(1, Manifold::Se2, vec![0.5, 0.6, 0.7]).unwrap();
//! graph.add_factor(
//!     Box::new(PriorFactor::new(0, Manifold::Se2, vec![0.0, 0.0, 0.0])),
//!     NoiseModel::Isotropic { sigma: 0.01 },
//!     None,
//! ).unwrap();
//! graph.add_factor(
//!     Box::new(BetweenFactor::new(0, 1, Manifold::Se2, vec![1.0, 0.0, 0.0])),
//!     NoiseModel::Isotropic { sigma: 0.05 },
//!     None,
//! ).unwrap();
//! let outcome = solve(&mut graph, &FactorSolverConfig::new()).unwrap();
//! assert_eq!(outcome.status, tpt_opt_nls::NlsStatus::Converged);
//! // Pose 1 snaps to (1, 0, 0) — the measured transform.
//! let p1 = graph.values.get(1).unwrap();
//! assert!((p1[0] - 1.0).abs() < 1e-6);
//! assert!(p1[1].abs() < 1e-6 && p1[2].abs() < 1e-6);
//! ```

pub mod factor;
pub mod factors;
pub mod graph;
pub mod linear;
pub mod manifold;
pub mod manifold_values;
pub mod marginal;
pub mod noise;
pub mod solver;

pub use factor::{Factor, Key, VarView};
pub use factors::{BetweenFactor, PriorFactor, RangeBearingFactor, ReprojectionFactor};
pub use graph::{FactorGraph, GraphError};
pub use manifold::Manifold;
pub use manifold_values::KeyedValues;
pub use noise::NoiseModel;
pub use solver::{solve, solve_incremental, solve_schur, FactorSolverConfig, SolveOutcome};
