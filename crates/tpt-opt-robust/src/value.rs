//! Value of the stochastic solution (VSS) and expected value of perfect
//! information (EVPI).
//!
//! Given a two-stage program (see [`crate::scenario`]):
//!
//! - **RP** — the recourse-problem optimum (the stochastic solution).
//! - **WS** — wait-and-see: per-scenario perfect-information optima
//!   `z_s*`; `E[WS] = Σ p_s z_s*`.
//! - **EEV** — expected-value solution: solve with expected data, then
//!   evaluate its expected true cost.
//!
//! Then `VSS = EEV − RP ≥ 0` (what stochastic optimisation buys over using
//! average data) and `EVPI = RP − E[WS] ≥ 0` (what perfect foresight would
//! buy).

use std::vec::Vec;

use tpt_opt_core::solver::Solver;
use tpt_opt_core::{OptError, SolverStatus};
use tpt_opt_milp::MilpSolver;

use crate::scenario::{RowSense, StageData, StageRow, TwoStageProblem};

/// VSS/EVPI metrics for a two-stage program.
#[derive(Debug, Clone, Copy)]
pub struct ValueMetrics {
    /// Recourse-problem optimum (stochastic solution value).
    pub rp: f64,
    /// Expected wait-and-see value `Σ p_s z_s*`.
    pub ws: f64,
    /// Expected cost of the expected-value solution.
    pub eev: f64,
    /// `EEV − RP ≥ 0`.
    pub vss: f64,
    /// `RP − E[WS] ≥ 0`.
    pub evpi: f64,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Solve one fixed-scenario problem: minimise `c1·x + c2·y` subject to the
/// scenario rows, returning the optimal objective.
fn solve_fixed_scenario(problem: &TwoStageProblem, s: usize) -> Result<f64, OptError> {
    let n1 = problem.first_bounds.len();
    let n2 = problem.second_bounds.len();
    let mut model = tpt_opt_core::model::Model::new(n1 + n2);
    for (i, b) in problem.first_bounds.iter().enumerate() {
        model.variables[i].bound = tpt_opt_core::VarBound::continuous(b.0, b.1);
    }
    for (j, b) in problem.second_bounds.iter().enumerate() {
        model.variables[n1 + j].bound = tpt_opt_core::VarBound::continuous(b.0, b.1);
    }
    let (_, data): &(crate::scenario::Scenario, StageData) = &problem.scenarios[s];
    let mut idx = Vec::with_capacity(n1 + n2);
    let mut coeffs = Vec::with_capacity(n1 + n2);
    for (j, &c) in problem.first_cost.iter().chain(data.cost.iter()).enumerate() {
        if c != 0.0 {
            idx.push(j);
            coeffs.push(c);
        }
    }
    model.set_objective(tpt_opt_core::model::Objective {
        sense: tpt_opt_core::model::Sense::Minimize,
        indices: idx,
        coeffs,
        constant: 0.0,
    });
    for row in &data.rows {
        let mut ridx = Vec::new();
        let mut rcoeffs = Vec::new();
        for (j, &w) in row.w.iter().enumerate() {
            if w != 0.0 {
                ridx.push(j);
                rcoeffs.push(w);
            }
        }
        for (j, &t) in row.t.iter().enumerate() {
            if t != 0.0 {
                ridx.push(n1 + j);
                rcoeffs.push(t);
            }
        }
        let con = match row.sense {
            RowSense::Le => tpt_opt_core::model::Constraint::le(ridx, rcoeffs, row.h),
            RowSense::Ge => tpt_opt_core::model::Constraint::ge(ridx, rcoeffs, row.h),
            RowSense::Eq => tpt_opt_core::model::Constraint::equality(ridx, rcoeffs, row.h),
        };
        model.add_constraint(con);
    }
    let mut solver = MilpSolver::new();
    let sol = solver.solve(&model)?;
    debug_assert_eq!(sol.status, SolverStatus::Optimal);
    Ok(sol.objective_value)
}

/// Evaluate the expected true cost of a first-stage decision `x` under the
/// two-stage problem's scenarios (re-optimising recourse per scenario).
fn expected_cost_of(problem: &TwoStageProblem, x: &[f64]) -> Result<f64, OptError> {
    let total_p: f64 = problem.scenarios.iter().map(|(s, _)| s.probability).sum();
    let mut total = dot(&problem.first_cost, x);
    for ((scen, data), _) in problem.scenarios.iter().zip(std::iter::repeat(())) {
        // Recourse subproblem with x fixed: rows become T·y ⋈ h − W·x.
        let n2 = problem.second_bounds.len();
        let mut model = tpt_opt_core::model::Model::new(n2);
        for (j, b) in problem.second_bounds.iter().enumerate() {
            model.variables[j].bound = tpt_opt_core::VarBound::continuous(b.0, b.1);
        }
        let mut idx = Vec::new();
        let mut coeffs = Vec::new();
        for (j, &c) in data.cost.iter().enumerate() {
            if c != 0.0 {
                idx.push(j);
                coeffs.push(c);
            }
        }
        model.set_objective(tpt_opt_core::model::Objective {
            sense: tpt_opt_core::model::Sense::Minimize,
            indices: idx,
            coeffs,
            constant: 0.0,
        });
        for row in &data.rows {
            let rhs = row.h - dot(&row.w, x);
            let ridx: Vec<usize> =
                row.t.iter().enumerate().filter(|&(_, &t)| t != 0.0).map(|(j, _)| j).collect();
            let rcoeffs: Vec<f64> = row.t.iter().copied().filter(|&t| t != 0.0).collect();
            let con = match row.sense {
                RowSense::Le => tpt_opt_core::model::Constraint::le(ridx, rcoeffs, rhs),
                RowSense::Ge => tpt_opt_core::model::Constraint::ge(ridx, rcoeffs, rhs),
                RowSense::Eq => tpt_opt_core::model::Constraint::equality(ridx, rcoeffs, rhs),
            };
            model.add_constraint(con);
        }
        let mut solver = MilpSolver::new();
        let sol = solver.solve(&model)?;
        total += (scen.probability / total_p) * sol.objective_value;
    }
    Ok(total)
}

/// Compute VSS/EVPI metrics for a two-stage program.
pub fn value_metrics(problem: &TwoStageProblem) -> Result<ValueMetrics, OptError> {
    // RP: the extensive-form optimum.
    let rp_sol = problem.solve()?;
    let rp = rp_sol.objective;

    // WS: per-scenario perfect information.
    let total_p: f64 = problem.scenarios.iter().map(|(s, _)| s.probability).sum();
    let mut ws = 0.0;
    for s in 0..problem.scenarios.len() {
        let p = problem.scenarios[s].0.probability / total_p;
        ws += p * solve_fixed_scenario(problem, s)?;
    }

    // EEV: solve with expected RHS data, then evaluate truly.
    // Build the expected-data problem: h̄ = Σ p_s h_s per row position.
    let n_rows = problem.scenarios[0].1.rows.len();
    let mut exp_data =
        StageData { cost: problem.scenarios[0].1.cost.clone(), rows: Vec::with_capacity(n_rows) };
    for r in 0..n_rows {
        let template = &problem.scenarios[0].1.rows[r];
        let mut h_bar = 0.0;
        for ((scen, data), _) in problem.scenarios.iter().zip(std::iter::repeat(())) {
            h_bar += (scen.probability / total_p) * data.rows[r].h;
        }
        exp_data.rows.push(StageRow {
            w: template.w.clone(),
            t: template.t.clone(),
            h: h_bar,
            sense: template.sense,
        });
    }
    let ev_problem = TwoStageProblem {
        first_cost: problem.first_cost.clone(),
        first_bounds: problem.first_bounds.clone(),
        second_bounds: problem.second_bounds.clone(),
        scenarios: vec![(
            crate::scenario::Scenario { probability: 1.0, data: Vec::new() },
            exp_data,
        )],
    };
    let ev_x = ev_problem.solve()?.x;
    let eev = expected_cost_of(problem, &ev_x)?;

    Ok(ValueMetrics { rp, ws, eev, vss: eev - rp, evpi: rp - ws })
}

/// Convenience wrapper computing only VSS.
pub fn vss(problem: &TwoStageProblem) -> Result<f64, OptError> {
    Ok(value_metrics(problem)?.vss)
}

/// Convenience wrapper computing only EVPI.
pub fn evpi(problem: &TwoStageProblem) -> Result<f64, OptError> {
    Ok(value_metrics(problem)?.evpi)
}
