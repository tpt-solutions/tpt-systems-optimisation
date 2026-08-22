//! Optional [HiGHS](https://highs.dev) binding (feature `highs`).
//!
//! [`HighsSolver`] is an alternate [`Solver`] implementation that translates a
//! canonical [`Model`] into HiGHS' column-wise form and delegates the actual
//! optimisation to the HiGHS library (via the MIT-licensed `highs`/`highs-sys`
//! bindings). It exists for benchmarking and cross-validation against the
//! bundled branch-and-bound engine ([`crate::MilpSolver`]); it is **not**
//! enabled by default because `highs-sys` compiles the HiGHS C++ sources,
//! which requires a C++ build toolchain (cmake plus an MSVC/gcc/clang
//! compiler) at build time.
//!
//! Status mapping: HiGHS `Optimal`/`ModelEmpty` → [`SolverStatus::Optimal`];
//! `Infeasible`/`UnboundedOrInfeasible` → [`SolverStatus::Infeasible`];
//! `Unbounded` → [`SolverStatus::Unbounded`]; `ReachedTimeLimit` and the other
//! early-termination statuses (iteration/solution/memory limits, objective
//! bound/target cutoffs) → [`SolverStatus::TimeLimit`]; solver/model errors →
//! [`OptError`]. The objective constant of the canonical model is added to the
//! reported objective value (HiGHS itself only sees the linear part).

use std::ops::Bound as RangeEnd;

use highs::{Col, HighsModelStatus, RowProblem, Sense};
use tpt_opt_core::bounds::VarType;
use tpt_opt_core::model::{Model, Sense as CoreSense};
use tpt_opt_core::solver::{Solution, SolveParameters, Solver, SolverStatus, Verbosity, WarmStart};
use tpt_opt_core::OptError;

/// Alternate [`Solver`] backed by the external HiGHS library.
///
/// The solver is stateless between solves apart from the stored parameter
/// bundle and warm-start hint; each [`Solver::solve`] rebuilds the HiGHS
/// problem from scratch.
#[derive(Debug, Clone, Default)]
pub struct HighsSolver {
    params: SolveParameters,
    warm: Option<WarmStart>,
    last: Option<Solution>,
}

impl HighsSolver {
    /// A solver with default parameters.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Translate a core `[lower, upper]` interval into HiGHS range ends.
fn row_ends(lower: f64, upper: f64) -> (RangeEnd<f64>, RangeEnd<f64>) {
    let lo = if lower.is_finite() { RangeEnd::Included(lower) } else { RangeEnd::Unbounded };
    let hi = if upper.is_finite() { RangeEnd::Included(upper) } else { RangeEnd::Unbounded };
    (lo, hi)
}

/// Map a HiGHS terminal status onto the core status enum.
fn map_status(status: HighsModelStatus) -> SolverStatus {
    match status {
        HighsModelStatus::Optimal | HighsModelStatus::ModelEmpty => SolverStatus::Optimal,
        HighsModelStatus::Infeasible | HighsModelStatus::UnboundedOrInfeasible => {
            SolverStatus::Infeasible
        }
        HighsModelStatus::Unbounded => SolverStatus::Unbounded,
        // Early termination with a possibly usable incumbent.
        HighsModelStatus::ReachedTimeLimit
        | HighsModelStatus::ReachedIterationLimit
        | HighsModelStatus::ReachedSolutionLimit
        | HighsModelStatus::ReachedMemoryLimit
        | HighsModelStatus::ReachedInterrupt
        | HighsModelStatus::ObjectiveBound
        | HighsModelStatus::ObjectiveTarget => SolverStatus::TimeLimit,
        // Everything else is a failure of some kind.
        _ => SolverStatus::NumericalIssue,
    }
}

/// Apply the stored parameter bundle to a freshly built HiGHS model.
fn apply_params(model: &mut highs::Model, params: &SolveParameters) -> Result<(), OptError> {
    if let Some(seconds) = params.time_limit {
        model.set_option("time_limit", seconds);
    }
    if params.threads > 1 {
        model.set_option("threads", params.threads.min(i32::MAX as usize) as i32);
    }
    match params.verbosity {
        Verbosity::Quiet => model.make_quiet(),
        // `Model::try_new` starts quiet; re-enable output for louder levels.
        Verbosity::Normal => {
            let _ = model.try_set_option("output_flag", true);
            let _ = model.try_set_option("log_to_console", true);
        }
        Verbosity::Verbose => {
            let _ = model.try_set_option("output_flag", true);
            let _ = model.try_set_option("log_to_console", true);
            let _ = model.try_set_option("log_dev_level", 2);
        }
    }
    model
        .try_set_option("mip_abs_gap", params.absolute_gap)
        .map_err(|_| OptError::invalid_model("HiGHS rejected mip_abs_gap"))?;
    model
        .try_set_option("mip_rel_gap", params.relative_gap)
        .map_err(|_| OptError::invalid_model("HiGHS rejected mip_rel_gap"))?;
    if let Some(seed) = params.seed {
        model
            .try_set_option("random_seed", seed as i32)
            .map_err(|_| OptError::invalid_model("HiGHS rejected random_seed"))?;
    }
    let t = &params.tolerances;
    let _ = model.try_set_option("primal_feasibility_tolerance", t.feasibility);
    let _ = model.try_set_option("dual_feasibility_tolerance", t.feasibility);
    let _ = model.try_set_option("integrality_tolerance", t.integrality);
    Ok(())
}

impl Solver<Model> for HighsSolver {
    fn solve(&mut self, model: &Model) -> Result<Solution, OptError> {
        model.validate()?;
        let n = model.num_vars;

        // Objective coefficients per column (0 where absent).
        let mut costs = vec![0.0f64; n];
        for (&i, &c) in model.objective.indices.iter().zip(&model.objective.coeffs) {
            costs[i] = c;
        }

        // Columns first (RowProblem shape matches the canonical model).
        let mut pb = RowProblem::new();
        let mut cols: Vec<Col> = Vec::with_capacity(n);
        for v in &model.variables {
            let b = v.bound.bound;
            let (lo, hi) = row_ends(b.lower, b.upper);
            let cost = costs[v.index];
            // Read the kind from `bound` (authoritative) rather than the
            // possibly-stale `Variable::kind` mirror.
            let col = match v.bound.kind {
                VarType::Continuous => pb.add_column(cost, (lo, hi)),
                VarType::Integer | VarType::Binary => pb.add_integer_column(cost, (lo, hi)),
                VarType::SemiContinuous => pb.add_semi_continuous_column(cost, (lo, hi)),
            };
            cols.push(col);
        }

        // Rows.
        for c in &model.constraints {
            let entries: Vec<(Col, f64)> =
                c.indices.iter().map(|&i| cols[i]).zip(c.coeffs.iter().copied()).collect();
            let (lo, hi) = row_ends(c.lower, c.upper);
            pb.add_row((lo, hi), entries);
        }

        let sense = match model.objective.sense {
            CoreSense::Minimize => Sense::Minimise,
            CoreSense::Maximize => Sense::Maximise,
        };
        let mut hmodel = pb.optimise(sense);
        apply_params(&mut hmodel, &self.params)?;

        // Warm start: pass a full-length primal guess when available.
        if let Some(warm) = &self.warm {
            if let Some(primal) = &warm.primal {
                if primal.len() == hmodel.num_cols() {
                    let _ = hmodel.try_set_solution(Some(primal), None, None, None);
                }
            }
        }

        let solved = hmodel
            .try_solve()
            .map_err(|e| OptError::numerical(format!("HiGHS run failed: {e:?}")))?;
        let status = map_status(solved.status());
        let hsol = solved.get_solution();

        let primal = hsol.columns().to_vec();
        // HiGHS reports the linear objective only; restore the constant.
        let objective_value = solved.objective_value() + model.objective.constant;
        let dual = hsol.dual_rows().to_vec();
        let reduced_costs = hsol.dual_columns().to_vec();
        // Slacks follow the canonical convention via the original rows.
        let slacks: Vec<f64> = model.constraints.iter().map(|c| c.slack(&primal)).collect();

        let sol = Solution {
            primal,
            dual,
            reduced_costs,
            slacks,
            objective_value,
            status,
            iterations: None,
            solve_time: None,
        };
        self.last = Some(sol.clone());
        Ok(sol)
    }

    fn set_parameter(&mut self, param: &SolveParameters) -> Result<(), OptError> {
        self.params = *param;
        Ok(())
    }

    fn warm_start(&mut self, warm: WarmStart) -> Result<(), OptError> {
        self.warm = Some(warm);
        Ok(())
    }

    fn status(&self) -> SolverStatus {
        self.last.as_ref().map_or(SolverStatus::Error, |s| s.status)
    }

    fn solution(&self) -> Option<Solution> {
        self.last.clone()
    }
}
