//! Irreducible infeasible subsystem (IIS) by the deletion filter.
//!
//! An IIS is a set of constraints that is infeasible together with the
//! variable bounds, and becomes feasible when any one of its constraints is
//! removed. Those constraints are the usual starting point for diagnosing why a
//! model has no solution.
//!
//! The filter tries each candidate constraint in turn and drops it when the
//! remaining set is still infeasible. Variable bounds and custom constraints are
//! always kept, so the result is an IIS over the linear constraints only.

use tpt_opt_core::{Model, OptError, Solver, SolverStatus};

use crate::MilpSolver;

/// Constraint indices of an irreducible infeasible subsystem, in model order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Iis {
    /// Indices into `model.constraints`.
    pub constraints: Vec<usize>,
}

/// Find an IIS of `model`.
///
/// Returns an error when the model is not proven infeasible (it is feasible, or
/// the solver stopped without a verdict) or when a subproblem fails to solve.
///
/// ```
/// use tpt_opt_core::expr::ModelBuilder;
/// use tpt_opt_milp::find_iis;
///
/// let mut b = ModelBuilder::new();
/// let x = b.continuous("x", 0.0, 10.0);
/// let y = b.continuous("y", 0.0, 10.0);
/// b.constrain(x.ge(3.0)); // conflicts with the next row
/// b.constrain(x.le(1.0));
/// b.constrain((x + y).le(9.0)); // not part of the conflict
/// let model = b.build().unwrap();
///
/// assert_eq!(find_iis(&model).unwrap().constraints, vec![0, 1]);
/// ```
pub fn find_iis(model: &Model) -> Result<Iis, OptError> {
    if verdict(model)? != SolverStatus::Infeasible {
        return Err(OptError::invalid_model("model is not proven infeasible"));
    }

    let mut kept: Vec<usize> =
        (0..model.constraints.len()).filter(|&i| !model.constraints[i].is_custom).collect();

    let mut i = 0;
    while i < kept.len() {
        let mut trial = kept.clone();
        trial.remove(i);
        if verdict(&restrict(model, &trial))? == SolverStatus::Infeasible {
            kept = trial; // still infeasible without this row: drop it for good
        } else {
            i += 1; // this row is needed for the conflict
        }
    }
    Ok(Iis { constraints: kept })
}

/// Copy of `model` whose linear constraints are exactly `subset` (custom
/// constraints are always retained).
fn restrict(model: &Model, subset: &[usize]) -> Model {
    let mut copy = model.clone();
    copy.constraints = model
        .constraints
        .iter()
        .enumerate()
        .filter(|(i, c)| c.is_custom || subset.contains(i))
        .map(|(_, c)| c.clone())
        .collect();
    copy
}

fn verdict(model: &Model) -> Result<SolverStatus, OptError> {
    let sol = MilpSolver::new().solve(model)?;
    Ok(sol.status)
}
