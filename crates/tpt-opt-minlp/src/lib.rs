//! Mixed-integer nonlinear programming (MINLP).
//!
//! This crate provides deterministic algorithms for problems mixing integer
//! decisions with nonlinear functions:
//!
//! - [`model`] — the [`MinlpModel`] representation: boxed-closure objective
//!   and constraints (`g(x) <= 0`, `h(x) = 0`), optional analytic gradients
//!   with finite-difference fallback, and indicator-gated constraints
//!   (nonlinear consequents behind a binary switch).
//! - [`oa`] — **outer approximation** (Duran–Grossmann): MILP epigraph
//!   master accumulating tangent planes ↔ NLP subproblems with integers
//!   fixed; converges with duality-gap certificates on convex instances.
//! - [`gbd`] — **generalized Benders decomposition** over complicating
//!   integer variables, with validity-checked slope cuts and violation-based
//!   feasibility cuts.
//! - [`sqp`] — **SQP-style branch-and-bound** for (possibly non-convex)
//!   MINLPs: NLP relaxations per node, most-fractional branching, bound
//!   pruning.
//! - [`subproblem`] — the shared continuous NLP subproblem adapter over
//!   `tpt-math-optimize-general`'s augmented-Lagrangian solver.
//! - [`relax`] — **McCormick envelopes** for bilinear products.
//! - [`alphabb`] — **αBB convex underestimators** and tangent cuts for
//!   twice-differentiable terms with a curvature bound.
//! - [`logical`] — AND/OR/XOR/cardinality/implication constraints compiled
//!   to linear rows over binaries.
//! - [`complementarity`] — big-M linearisation of `x·y = 0` pairs.
//! - [`certificates`] — per-iteration lower/upper bounds and gap tracking.
//!
//! # Example
//!
//! ```
//! use tpt_opt_minlp::{model::{MinlpModel, VarKind}, oa::{outer_approximate, OaConfig}};
//!
//! // min x + y  s.t.  y >= (x−1)²,  x ∈ [0,2] continuous, y ∈ ℤ ∩ [0,4].
//! let mut m = MinlpModel::new(2, |x| x[0] + x[1]);
//! m.set_var(0, VarKind::Continuous, 0.0, 2.0);
//! m.set_var(1, VarKind::Integer, 0.0, 4.0);
//! m.add_le(
//!     |x| (x[0] - 1.0) * (x[0] - 1.0) - x[1],
//!     |x, g| { g[0] = 2.0 * (x[0] - 1.0); g[1] = -1.0; },
//! );
//! let res = outer_approximate(&m, &OaConfig::default());
//! assert_eq!(res.status, tpt_opt_minlp::oa::OaStatus::Optimal);
//! assert!((res.objective.unwrap() - 1.0).abs() < 1e-3);
//! ```

pub mod alphabb;
pub mod certificates;
pub mod complementarity;
pub mod gbd;
pub mod logical;
pub mod model;
pub mod oa;
pub mod relax;
pub mod sqp;
pub mod subproblem;

pub use certificates::{CertificateHistory, ConvergenceCertificate};
pub use model::MinlpModel;
pub use oa::{OaConfig, OaResult, OaStatus};
