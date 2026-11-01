//! Rendering of models, solutions, and reports as text or JSON.
//!
//! JSON is written by hand (no serde dependency) and only needs strings,
//! numbers, booleans, arrays, and objects, so the escaping is small.

use tpt_opt_core::bounds::VarType;
use tpt_opt_core::{Model, Sense, SolverStatus};
use tpt_opt_milp::PresolveReport;

/// Compact number: `3` for `3.0`, `-10.5` for `-10.5`.
pub fn fmt_num(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// Quote and escape a string as a JSON string literal.
pub fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// JSON value for a float: `null` for infinities and NaN.
pub fn json_num(v: f64) -> String {
    if v.is_finite() {
        fmt_num(v)
    } else {
        "null".into()
    }
}

/// Human-readable status.
pub fn status_text(s: SolverStatus) -> &'static str {
    match s {
        SolverStatus::Optimal => "optimal",
        SolverStatus::Infeasible => "infeasible",
        SolverStatus::Unbounded => "unbounded",
        SolverStatus::TimeLimit => "time limit (best solution shown)",
        SolverStatus::NumericalIssue => "numerical issue",
        SolverStatus::Error => "error",
    }
}

/// Machine-readable status used in JSON output.
pub fn status_key(s: SolverStatus) -> &'static str {
    match s {
        SolverStatus::Optimal => "optimal",
        SolverStatus::Infeasible => "infeasible",
        SolverStatus::Unbounded => "unbounded",
        SolverStatus::TimeLimit => "time_limit",
        SolverStatus::NumericalIssue => "numerical_issue",
        SolverStatus::Error => "error",
    }
}

/// Structural summary of a model.
pub struct Summary {
    /// Continuous variable count.
    pub continuous: usize,
    /// Integer variable count.
    pub integer: usize,
    /// Binary variable count.
    pub binary: usize,
    /// Semi-continuous variable count.
    pub semi_continuous: usize,
    /// Nonzero coefficients over all constraints.
    pub nonzeros: usize,
}

/// Count variable kinds and constraint nonzeros.
pub fn summarise(model: &Model) -> Summary {
    let count = |k: VarType| model.variables.iter().filter(|v| v.kind == k).count();
    Summary {
        continuous: count(VarType::Continuous),
        integer: count(VarType::Integer),
        binary: count(VarType::Binary),
        semi_continuous: count(VarType::SemiContinuous),
        nonzeros: model
            .constraints
            .iter()
            .map(|c| c.coeffs.iter().filter(|a| **a != 0.0).count())
            .sum(),
    }
}

/// `"linear program"` or `"mixed-integer program"`.
pub fn class_text(model: &Model) -> &'static str {
    if model.variables.iter().all(|v| v.kind == VarType::Continuous) {
        "linear program"
    } else {
        "mixed-integer program"
    }
}

/// One constraint as text, e.g. `2*x0 + x1 in [1, 4]`.
pub fn row_text(indices: &[usize], coeffs: &[f64], lower: f64, upper: f64) -> String {
    let mut expr = String::new();
    for (k, (&j, &a)) in indices.iter().zip(coeffs).enumerate() {
        if k > 0 {
            expr.push_str(if a < 0.0 { " - " } else { " + " });
        } else if a < 0.0 {
            expr.push('-');
        }
        let mag = a.abs();
        if mag == 1.0 {
            expr.push_str(&format!("x{j}"));
        } else {
            expr.push_str(&format!("{}*x{j}", fmt_num(mag)));
        }
    }
    if expr.is_empty() {
        expr.push('0');
    }
    let rel = if lower == upper {
        format!("= {}", fmt_num(lower))
    } else if lower == f64::NEG_INFINITY {
        format!("<= {}", fmt_num(upper))
    } else if upper == f64::INFINITY {
        format!(">= {}", fmt_num(lower))
    } else {
        format!("in [{}, {}]", fmt_num(lower), fmt_num(upper))
    };
    format!("{expr} {rel}")
}

/// Text block describing a presolve run.
pub fn presolve_text(r: &PresolveReport) -> String {
    format!(
        "removed rows:      {}\ntightened bounds:  {}\nfixed variables:   {}\ninfeasible:        {}",
        r.removed_rows, r.tightened_bounds, r.fixed_variables, r.infeasible
    )
}

/// JSON object describing a presolve run.
pub fn presolve_json(r: &PresolveReport) -> String {
    format!(
        "{{\"removed_rows\":{},\"tightened_bounds\":{},\"fixed_variables\":{},\"infeasible\":{}}}",
        r.removed_rows, r.tightened_bounds, r.fixed_variables, r.infeasible
    )
}

/// Sense name for display.
pub fn sense_text(s: Sense) -> &'static str {
    match s {
        Sense::Minimize => "minimize",
        Sense::Maximize => "maximize",
    }
}
