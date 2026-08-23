//! {{description}}
//!
//! Built on [`tpt_opt_systems`], the umbrella crate of the
//! `tpt-systems-optimisation` workspace. Only the solver families enabled by
//! this crate's dependency features are compiled in; see the root README's
//! Tier 2 consumption table for the recommended feature set per domain.
//!
//! # Quick start (always available — core model types)
//!
//! ```
//! use tpt_opt_systems::core::{Constraint, Model, Objective, VarBound};
//!
//! // max 3x + 2y  s.t. x + y <= 4  (x, y integer in [0, 4])
//! let mut m = Model::new(2);
//! m.variables[0].bound = VarBound::integer(0.0, 4.0);
//! m.variables[1].bound = VarBound::integer(0.0, 4.0);
//! m.add_constraint(Constraint::le(vec![0, 1], vec![1.0, 1.0], 4.0));
//! m.set_objective(Objective {
//!     sense: tpt_opt_systems::core::Sense::Maximize,
//!     indices: vec![0, 1],
//!     coeffs: vec![3.0, 2.0],
//!     constant: 0.0,
//! });
//! assert_eq!(m.validate(), Ok(()));
//! ```
//!
//! With the `milp` feature enabled you can solve it through the fluent
//! builder:
//!
//! ```ignore
//! let sol = tpt_opt_systems::MilpBuilder::new(&m)?.maximize()?.solve()?;
//! assert_eq!(sol.objective_value, 14.0);
//! ```

// Implementation goes here. Conventions (see CONTRIBUTING.md upstream):
// - Result-returning APIs; never panic on bad input.
// - Route numeric tolerances through `tpt_opt_systems::core::Tolerances`.
// - Seedable determinism for anything randomised.