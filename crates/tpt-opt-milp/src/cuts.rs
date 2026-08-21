//! Gomory mixed-integer (fractional) cut generation for the MILP solver.
//!
//! Cuts are generated directly in the optimal simplex tableau space and
//! appended as new `<=` rows (with slack columns) to a [`LpState`], after which
//! the tableau is re-optimised. The classic Gomory fractional cut assumes the
//! row's variables are integer; for mixed-integer problems it is approximate.
//! The MILP solver therefore only enables cuts behind `with_cuts(true)` and
//! reverts any cut that would worsen (or invalidate) the relaxation bound, so
//! overall solve correctness is preserved.

use tpt_opt_core::tolerance::Tolerances;

use crate::lp::{reoptimize, LpState};

/// Add up to `max_cuts` Gomory fractional cuts to the tableau of `state` for
/// the fractional basic integer variables, re-optimising after each. Returns
/// the number of cuts actually added.
pub fn add_gomory_cuts(
    state: &mut LpState,
    int_vars: &[usize],
    tol: Tolerances,
    max_cuts: usize,
) -> usize {
    let mut added = 0;
    for _ in 0..max_cuts {
        if !add_one_cut(state, int_vars, tol) {
            break;
        }
        added += 1;
    }
    added
}

/// Add a single Gomory cut from the most fractional basic integer variable.
/// Returns `false` when no suitable fractional integer row exists.
fn add_one_cut(state: &mut LpState, int_vars: &[usize], tol: Tolerances) -> bool {
    let mrows = state.mrows;
    let n_struct = state.n_struct;
    let mut target: Option<(usize, f64)> = None;
    for ri in 0..mrows {
        let c = state.basis[ri];
        if c >= n_struct {
            continue;
        }
        let is_int = match state.col_to_orig[c] {
            Some(o) => int_vars.contains(&o),
            None => false,
        };
        if !is_int {
            continue;
        }
        let v = state.b[ri];
        let frac = v - v.floor();
        if frac > tol.integrality && (1.0 - frac) > tol.integrality {
            target = Some((ri, v));
            break;
        }
    }
    let (ri, v) = match target {
        Some(t) => t,
        None => return false,
    };

    let n_cols = state.n_cols;
    let f0 = v - v.floor();
    let cut: Vec<f64> = (0..n_cols).map(|k| frac_part(state.a[ri][k])).collect();
    // Convert sum_k fk x_k >= f0  to  sum_k (-fk) x_k + s = -f0  (s >= 0 slack).
    let slack = n_cols;
    state.n_cols += 1;
    for r in 0..mrows {
        state.a[r].push(0.0);
    }
    let mut new_row = vec![-1.0f64; state.n_cols];
    for (k, c) in cut.iter().enumerate().take(n_cols) {
        new_row[k] = -c;
    }
    new_row[slack] = 1.0;
    state.a.push(new_row);
    state.b.push(-f0);
    state.basis.push(slack);
    state.mrows += 1;

    reoptimize(state, tol);
    true
}

/// Fractional part of `x` in `[0, 1)`.
fn frac_part(x: f64) -> f64 {
    let f = x - x.floor();
    if f < 0.0 {
        f + 1.0
    } else {
        f
    }
}
