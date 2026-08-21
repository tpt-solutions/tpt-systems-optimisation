#![no_std]
//! Local dev shim mirroring `tpt-math-numeric`: tolerance helpers and finite
//! arithmetic utilities shared across the optimisation crates.

extern crate alloc;

/// Default relative/absolute comparison tolerance used across the workspace.
pub const DEFAULT_TOL: f64 = 1e-9;

/// Returns `true` if `a` and `b` are equal within absolute `tol`.
pub fn approx_eq_abs(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// Returns `true` if `a` and `b` are equal within a combined relative/absolute
/// tolerance (ULP-free, robust for mixed magnitudes).
pub fn approx_eq(a: f64, b: f64, abs_tol: f64, rel_tol: f64) -> bool {
    let diff = (a - b).abs();
    if diff <= abs_tol {
        return true;
    }
    let scale = a.abs().max(b.abs());
    diff <= rel_tol * scale
}

/// Returns `true` if `x` is finite (not NaN or infinite).
pub fn is_finite(x: f64) -> bool {
    x.is_finite()
}

/// Clamp `x` into `[lo, hi]`.
pub fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Returns the fractional part distance to the nearest integer, used for
/// integrality and most-fractional branching decisions.
pub fn distance_to_integer(x: f64) -> f64 {
    let trunc = (x as i64) as f64;
    let d = (x - trunc).abs();
    // distance to the *nearest* integer (handles the x.5 case)
    d.min(1.0 - d)
}

/// `true` if `x` is within `eps` of an integer.
pub fn is_integer(x: f64, eps: f64) -> bool {
    distance_to_integer(x) <= eps
}
