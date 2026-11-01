//! Problem classification and automatic solver selection.
//!
//! [`classify`] inspects a canonical [`Model`] and reports whether it is a
//! pure linear program or a mixed-integer program. [`solve_auto`] solves it
//! with the MILP backend when the `milp` feature is enabled.
//!
//! The canonical [`Model`] is linear, so the only structural distinction the
//! model itself carries is its variable kinds. Nonlinear, conic, stochastic,
//! and constraint-programming problems have their own types and solvers, which
//! callers choose directly; this module does not route them.

use tpt_opt_core::{Model, VarType};

/// Coarse structural class of a canonical [`Model`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemClass {
    /// Every variable is continuous, so the model is a linear program.
    LinearProgram,
    /// At least one variable is integer, binary, or semi-continuous.
    MixedInteger,
}

/// Classify `model` by the kinds of its variables.
///
/// ```
/// use tpt_opt_systems::{classify, ProblemClass};
/// use tpt_opt_systems::expr::ModelBuilder;
///
/// let mut b = ModelBuilder::new();
/// let x = b.continuous("x", 0.0, 4.0);
/// b.maximize(x);
/// assert_eq!(classify(&b.build().unwrap()), ProblemClass::LinearProgram);
///
/// let mut b = ModelBuilder::new();
/// let y = b.binary("y");
/// b.maximize(y);
/// assert_eq!(classify(&b.build().unwrap()), ProblemClass::MixedInteger);
/// ```
pub fn classify(model: &Model) -> ProblemClass {
    if model.variables.iter().all(|v| v.kind == VarType::Continuous) {
        ProblemClass::LinearProgram
    } else {
        ProblemClass::MixedInteger
    }
}

/// Solve a linear or mixed-integer [`Model`] with the MILP backend.
///
/// Both classes go to [`tpt_opt_milp::MilpSolver`], whose root node is the LP
/// relaxation, so a pure LP is solved as a branch-and-bound with no branching.
/// Call [`classify`] first if the caller needs to route the problem itself.
#[cfg(feature = "milp")]
pub fn solve_auto(model: &Model) -> Result<tpt_opt_core::Solution, tpt_opt_core::OptError> {
    use tpt_opt_core::Solver;
    #[cfg(feature = "tracing")]
    let _span = tracing::info_span!(
        "solve_auto",
        class = ?classify(model),
        variables = model.num_vars,
        constraints = model.num_constraints(),
    )
    .entered();

    let result = tpt_opt_milp::MilpSolver::new().solve(model);

    #[cfg(feature = "tracing")]
    match &result {
        Ok(sol) => tracing::info!(status = ?sol.status, objective = sol.objective_value, "solved"),
        Err(err) => tracing::warn!(error = %err, "solve failed"),
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_opt_core::expr::ModelBuilder;

    #[test]
    fn continuous_model_is_a_linear_program() {
        let mut b = ModelBuilder::new();
        let x = b.continuous("x", 0.0, 1.0);
        b.minimize(x);
        assert_eq!(classify(&b.build().unwrap()), ProblemClass::LinearProgram);
    }

    #[test]
    fn any_discrete_kind_makes_a_mixed_integer_program() {
        for kind in [VarType::Integer, VarType::Binary, VarType::SemiContinuous] {
            let mut b = ModelBuilder::new();
            let z = b.continuous("z", 0.0, 2.0);
            let mut m = b.build().unwrap();
            m.variables[z.index()].kind = kind;
            assert_eq!(classify(&m), ProblemClass::MixedInteger, "{kind:?}");
        }
    }

    #[cfg(feature = "milp")]
    #[test]
    fn solve_auto_solves_a_small_knapsack() {
        let mut b = ModelBuilder::new();
        let x = b.binary("x");
        let y = b.binary("y");
        b.constrain((3.0 * x + 4.0 * y).le(4.0));
        b.maximize(2.0 * x + 3.0 * y);
        let sol = solve_auto(&b.build().unwrap()).unwrap();
        assert!((sol.objective_value - 3.0).abs() < 1e-9);
    }
}
