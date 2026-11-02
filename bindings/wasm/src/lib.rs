//! Browser entry point: solve an MPS or CPLEX-LP model given as text and return
//! the result as JSON.

use std::time::Duration;

use tpt_opt_core::{OptError, Solver, SolverStatus};
use tpt_opt_milp::{read_lp, read_mps, MilpSolver};
use wasm_bindgen::prelude::*;

fn status_key(s: SolverStatus) -> &'static str {
    match s {
        SolverStatus::Optimal => "optimal",
        SolverStatus::Infeasible => "infeasible",
        SolverStatus::Unbounded => "unbounded",
        SolverStatus::TimeLimit => "time_limit",
        SolverStatus::NumericalIssue => "numerical_issue",
        SolverStatus::Error => "error",
    }
}

/// JSON number: `null` for non-finite values, which JSON cannot represent.
fn json_num(v: f64) -> String {
    if v.is_finite() {
        format!("{v}")
    } else {
        "null".into()
    }
}

fn fail(message: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&message.to_string())
}

/// Solve `text` (`format` is `"mps"` or `"lp"`) and return a JSON object with
/// `status`, `objective`, `primal`, and `variables`. An optional `time_limit`
/// is in seconds. Parse and solver failures are returned as a thrown string.
#[wasm_bindgen]
pub fn solve(text: &str, format: &str, time_limit: Option<f64>) -> Result<String, JsValue> {
    let model = match format {
        "mps" => read_mps(text).map_err(|e: OptError| fail(format!("MPS: {e}")))?,
        "lp" => read_lp(text).map_err(|e: OptError| fail(format!("LP: {e}")))?,
        other => return Err(fail(format!("unknown format {other:?} (mps or lp)"))),
    };
    let mut solver = MilpSolver::new();
    if let Some(secs) = time_limit {
        if !(secs > 0.0 && secs.is_finite()) {
            return Err(fail("time_limit must be a positive number"));
        }
        solver = solver.with_time_limit(Duration::from_secs_f64(secs));
    }
    let sol = solver.solve(&model).map_err(|e| fail(format!("solve: {e}")))?;

    let primal: Vec<String> = sol.primal.iter().map(|v| json_num(*v)).collect();
    Ok(format!(
        "{{\"status\":\"{}\",\"objective\":{},\"primal\":[{}],\"variables\":{}}}",
        status_key(sol.status),
        json_num(sol.objective_value),
        primal.join(","),
        model.num_vars,
    ))
}

/// Crate version, so the page can show which build it is running.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
