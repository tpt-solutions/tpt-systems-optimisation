//! Tableau-space cut generation: Gomory mixed-integer cuts and
//! lift-and-project intersection cuts.
//!
//! Both families are derived from the final simplex tableau of an LP
//! relaxation ([`LpState`]) and translated back into model space, so they can
//! be appended to the [`Model`] and tighten every node LP of the
//! branch-and-bound tree.
//!
//! - **Gomory mixed-integer cuts** ([`add_gomory_cuts`]): for each tableau
//!   row whose basic variable is an integer-constrained structural column
//!   with a fractional value, the row `sum_j ᾱ_j y_j = b̄` (all columns
//!   non-negative, nonbasics at zero) yields the valid inequality
//!   `sum_j π_j y_j >= 1` with the classic Gomory mixed-integer coefficients:
//!   integer nonbasics are rounded by fractionality, continuous nonbasics are
//!   amplified by `1/f` (positive ᾱ) or `1/(1-f)` (negative ᾱ), where
//!   `f = frac(b̄)`.
//! - **Lift-and-project intersection cuts** ([`add_lift_and_project_cuts`]):
//!   the same derivation restricted to rows whose basic variable is a
//!   *binary* column — the intersection cut for the disjunction
//!   `x_j <= 0 ∨ x_j >= 1`, i.e. the first round of a Balas-style
//!   lift-and-project scheme.
//!
//! Every cut produced by these generators is violated by the LP point the
//! tableau was solved from (nonbasic columns sit at zero, so the cut reads
//! `0 >= 1` there), and is valid for every mixed-integer point of the model
//! the relaxation came from. Both properties are guarded by exhaustive unit
//! tests on small enumerated instances.

use std::vec::Vec;

use tpt_opt_core::model::{Constraint, Model};

use crate::lp::LpState;

/// A cut `sum coefs[i] * x[vars[i]] <= rhs` in original variable space.
#[derive(Debug, Clone, PartialEq)]
pub struct TableauCut {
    /// Variable indices (sorted ascending).
    pub vars: Vec<usize>,
    /// Coefficients aligned with `vars`.
    pub coefs: Vec<f64>,
    /// Right-hand side of the `<=` inequality.
    pub rhs: f64,
}

impl TableauCut {
    /// Evaluate the left-hand side activity at `x`.
    pub fn activity(&self, x: &[f64]) -> f64 {
        self.vars.iter().zip(self.coefs.iter()).map(|(&v, &c)| c * x[v]).sum()
    }

    /// Whether the cut is violated at `x` by more than `tol`.
    pub fn is_violated(&self, x: &[f64], tol: f64) -> bool {
        self.activity(x) > self.rhs + tol
    }
}

/// Generate up to `max_cuts` Gomory mixed-integer cuts from the final
/// tableau of an LP relaxation. `int_vars` lists the integer-constrained
/// original variables of the model.
pub fn gomory_cuts(state: &LpState, int_vars: &[usize], max_cuts: usize) -> Vec<TableauCut> {
    let ints: std::collections::BTreeSet<usize> = int_vars.iter().copied().collect();
    split_cuts(state, &ints, &ints, max_cuts)
}

/// Generate up to `max_cuts` lift-and-project intersection cuts from the
/// final tableau: only rows whose basic variable is one of `binaries` are
/// used (the `x_j ∈ {0,1}` disjunction). `int_vars` lists all
/// integer-constrained original variables (needed to classify columns).
pub fn lift_and_project_cuts(
    state: &LpState,
    binaries: &[usize],
    int_vars: &[usize],
    max_cuts: usize,
) -> Vec<TableauCut> {
    let ints: std::collections::BTreeSet<usize> = int_vars.iter().copied().collect();
    let targets: std::collections::BTreeSet<usize> = binaries.iter().copied().collect();
    split_cuts(state, &targets, &ints, max_cuts)
}

/// Append `cuts` to `model` as `<=` rows, skipping cuts identical to an
/// existing row. Returns the number of rows added.
pub fn add_cuts(model: &mut Model, cuts: Vec<TableauCut>) -> usize {
    let mut added = 0;
    for cut in cuts {
        let dup = model.constraints.iter().any(|r| {
            r.lower.is_infinite()
                && r.upper == cut.rhs
                && r.indices.len() == cut.vars.len()
                && r.indices.iter().zip(cut.vars.iter()).all(|(a, b)| a == b)
                && r.coeffs.iter().zip(cut.coefs.iter()).all(|(a, b)| a.to_bits() == b.to_bits())
        });
        if dup {
            continue;
        }
        model.add_constraint(Constraint::le(cut.vars.clone(), cut.coefs.clone(), cut.rhs));
        added += 1;
    }
    added
}

/// Convenience wrapper: generate Gomory cuts and append them to `model`.
pub fn add_gomory_cuts(
    model: &mut Model,
    state: &LpState,
    int_vars: &[usize],
    max_cuts: usize,
) -> usize {
    add_cuts(model, gomory_cuts(state, int_vars, max_cuts))
}

/// Convenience wrapper: generate lift-and-project cuts and append them to
/// `model`.
pub fn add_lift_and_project_cuts(
    model: &mut Model,
    state: &LpState,
    binaries: &[usize],
    int_vars: &[usize],
    max_cuts: usize,
) -> usize {
    add_cuts(model, lift_and_project_cuts(state, binaries, int_vars, max_cuts))
}

/// Shared derivation: for each tableau row whose basic variable is a
/// structural column of one of `targets` with a fractional value, build the
/// mixed-integer rounding inequality of that row and translate it into
/// original variable space.
fn split_cuts(
    state: &LpState,
    targets: &std::collections::BTreeSet<usize>,
    ints: &std::collections::BTreeSet<usize>,
    max_cuts: usize,
) -> Vec<TableauCut> {
    let mut cuts = Vec::new();
    if state.mrows == 0 || max_cuts == 0 {
        return cuts;
    }

    // Columns that map 1:1 to an original variable. Free-variable split
    // columns (yp/ym) map two columns onto one variable; neither split column
    // is individually integer-constrained, so they must be treated as
    // continuous and cannot serve as cut source rows.
    let n_orig = state.var_lb.len();
    let mut multiplicity = vec![0usize; n_orig];
    for orig in state.col_to_orig.iter().flatten() {
        if *orig < n_orig {
            multiplicity[*orig] += 1;
        }
    }
    let clean = |col: usize| -> Option<usize> {
        let j = *state.col_to_orig.get(col)?.as_ref()?;
        (j < n_orig && multiplicity[j] == 1).then_some(j)
    };

    let mut in_basis = vec![false; state.n_cols];
    for &b in &state.basis {
        if b < state.n_cols {
            in_basis[b] = true;
        }
    }

    let mut seen: std::collections::BTreeSet<(Vec<(usize, u64)>, u64)> =
        std::collections::BTreeSet::new();

    for ri in 0..state.mrows {
        if cuts.len() >= max_cuts {
            break;
        }
        let basic = state.basis[ri];
        if basic >= state.n_struct {
            continue; // slack/artificial basic: no integrality to exploit
        }
        let Some(bj) = clean(basic) else { continue };
        if !targets.contains(&bj) {
            continue;
        }
        let val = state.b[ri];
        if !val.is_finite() || val.abs() > 1e12 {
            continue;
        }
        let f = val - val.floor();
        if f <= 1e-7 || f >= 1.0 - 1e-7 {
            continue;
        }

        // Gomory mixed-integer coefficients over the nonbasic columns, then
        // translate `sum pi_j y_j >= 1` into original space via col_expr.
        let mut acc: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
        let mut constant = 0.0f64;
        let mut usable = false;
        for (col, in_b) in in_basis.iter().enumerate() {
            if *in_b {
                continue;
            }
            let abar = state.a[ri][col];
            if !abar.is_finite() || abar.abs() < 1e-11 {
                continue;
            }
            let Some(expr) = state.col_expr.get(col).and_then(|e| e.as_ref()) else {
                continue; // artificial column: no original-space meaning
            };
            let is_int_col = match clean(col) {
                Some(j) => ints.contains(&j),
                None => false,
            };
            let pi = if is_int_col {
                let fj = abar - abar.floor();
                if fj <= f {
                    fj / f
                } else {
                    (1.0 - fj) / (1.0 - f)
                }
            } else if abar > 0.0 {
                abar / f
            } else {
                -abar / (1.0 - f)
            };
            if !pi.is_finite() || pi.abs() < 1e-12 {
                continue;
            }
            usable = true;
            let (terms, k0) = expr;
            for &(vj, cf) in terms {
                *acc.entry(vj).or_insert(0.0) += pi * cf;
            }
            constant += pi * k0;
        }
        if !usable {
            continue;
        }

        // `sum acc_v x_v + constant >= 1`  →  `sum (-acc_v) x_v <= constant - 1`.
        let mut vars = Vec::new();
        let mut coefs = Vec::new();
        let mut ok = true;
        for (v, c) in acc {
            let c = -c;
            if c.abs() < 1e-11 {
                continue;
            }
            if !c.is_finite() || c.abs() > 1e9 {
                ok = false;
                break;
            }
            vars.push(v);
            coefs.push(c);
        }
        if !ok || vars.is_empty() {
            continue;
        }
        let rhs = constant - 1.0;
        if !rhs.is_finite() {
            continue;
        }
        let sig = (
            vars.iter().zip(coefs.iter()).map(|(&v, &c)| (v, c.to_bits())).collect::<Vec<_>>(),
            rhs.to_bits(),
        );
        if seen.insert(sig) {
            cuts.push(TableauCut { vars, coefs, rhs });
        }
    }
    cuts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::solve_lp_state;
    use tpt_opt_core::bounds::VarBound;
    use tpt_opt_core::model::{Model, Objective};
    use tpt_opt_core::tolerance::Tolerances;
    use tpt_opt_core::Solver;

    fn tol() -> Tolerances {
        Tolerances::default()
    }

    /// Enumerate all binary points satisfying `model` (n <= 12).
    fn feasible_points(model: &Model) -> Vec<Vec<f64>> {
        let n = model.num_vars;
        assert!(n <= 12);
        let mut out = Vec::new();
        for mask in 0u32..(1 << n) {
            let x: Vec<f64> = (0..n).map(|i| ((mask >> i) & 1) as f64).collect();
            if model.constraints.iter().all(|c| c.is_satisfied(&x, 1e-9))
                && model.variables.iter().enumerate().all(|(i, v)| v.bound.feasible(x[i], 1e-9))
            {
                out.push(x);
            }
        }
        out
    }

    #[test]
    fn gomory_cut_valid_and_violated() {
        // maximise x0 s.t. x0 + x1 <= 2.5, x0 - x1 <= 0.5, x0/x1 integer.
        // LP optimum x0 = 1.5; the Gomory cut from the fractional row is
        // x0 <= 1 (up to scaling), valid for every integer point and
        // violated at the LP optimum.
        let mut m = Model::new(2);
        m.set_objective(Objective::maximize(vec![0], vec![1.0]));
        m.add_constraint(Constraint::le(vec![0, 1], vec![1.0, 1.0], 2.5));
        m.add_constraint(Constraint::le(vec![0, 1], vec![1.0, -1.0], 0.5));
        m.variables[0].bound = VarBound::integer(0.0, 10.0);
        m.variables[1].bound = VarBound::integer(0.0, 10.0);

        let lb = vec![0.0, 0.0];
        let ub = vec![10.0, 10.0];
        let state = solve_lp_state(&m, &lb, &ub, tol());
        assert_eq!(state.sol.status, crate::lp::LpStatus::Optimal);
        assert!((state.sol.objective - 1.5).abs() < 1e-6, "lp obj {}", state.sol.objective);

        let cuts = gomory_cuts(&state, &[0, 1], 10);
        assert!(!cuts.is_empty(), "expected at least one Gomory cut");
        let pts = feasible_points(&m);
        assert!(!pts.is_empty());
        for cut in &cuts {
            // Validity: every integer-feasible point satisfies the cut.
            for x in &pts {
                assert!(cut.activity(x) <= cut.rhs + 1e-7, "Gomory cut {cut:?} cut off {x:?}");
            }
            // Violation: the cut excludes the fractional LP optimum.
            assert!(
                cut.is_violated(&state.sol.x, 1e-6),
                "Gomory cut {cut:?} not violated at {:?}",
                state.sol.x
            );
        }
    }

    #[test]
    fn gomory_cut_strengthens_lp_bound() {
        // maximise x0 s.t. 2.5 x0 <= 3.5, x0 integer in [0, 5]: the LP bound
        // is 1.4; adding the Gomory cut and re-solving must not increase the
        // LP objective beyond the integer optimum 1.
        let mut m = Model::new(1);
        m.set_objective(Objective::maximize(vec![0], vec![1.0]));
        m.add_constraint(Constraint::le(vec![0], vec![2.5], 3.5));
        m.variables[0].bound = VarBound::integer(0.0, 5.0);

        let lb = vec![0.0];
        let ub = vec![5.0];
        let state = solve_lp_state(&m, &lb, &ub, tol());
        assert_eq!(state.sol.status, crate::lp::LpStatus::Optimal);

        let mut work = m.clone();
        let added = add_gomory_cuts(&mut work, &state, &[0], 10);
        assert!(added >= 1);
        let strengthened = solve_lp_state(&work, &lb, &ub, tol());
        assert_eq!(strengthened.sol.status, crate::lp::LpStatus::Optimal);
        assert!(
            strengthened.sol.objective <= state.sol.objective + 1e-9,
            "cuts must not weaken the relaxation"
        );
        assert!(
            strengthened.sol.objective <= 1.0 + 1e-6,
            "strengthened bound {} exceeds the integer optimum",
            strengthened.sol.objective
        );
        // The integer optimum must remain feasible for the strengthened LP.
        let x1 = vec![1.0];
        for c in &work.constraints {
            assert!(c.is_satisfied(&x1, 1e-9), "cut removed the integer optimum");
        }
    }

    #[test]
    fn lift_and_project_cut_valid_and_violated() {
        // maximise x0 + x1 s.t. x0 + x1 <= 1.5, x0 + 2x1 <= 2.5, binaries.
        // LP optimum (1, 0.5); the intersection cut for the fractional
        // binary disjunction is 2x0 + x1 <= 2 (up to scaling): valid for all
        // binary points, violated at the LP optimum.
        let mut m = Model::new(2);
        m.set_objective(Objective::maximize(vec![0, 1], vec![1.0, 1.0]));
        m.add_constraint(Constraint::le(vec![0, 1], vec![1.0, 1.0], 1.5));
        m.add_constraint(Constraint::le(vec![0, 1], vec![1.0, 2.0], 2.5));
        m.variables[0].bound = VarBound::binary();
        m.variables[1].bound = VarBound::binary();

        let lb = vec![0.0, 0.0];
        let ub = vec![1.0, 1.0];
        let state = solve_lp_state(&m, &lb, &ub, tol());
        assert_eq!(state.sol.status, crate::lp::LpStatus::Optimal);
        assert!((state.sol.objective - 1.5).abs() < 1e-6, "lp obj {}", state.sol.objective);

        let cuts = lift_and_project_cuts(&state, &[0, 1], &[0, 1], 10);
        assert!(!cuts.is_empty(), "expected at least one lift-and-project cut");
        let pts = feasible_points(&m);
        assert!(!pts.is_empty());
        for cut in &cuts {
            for x in &pts {
                assert!(
                    cut.activity(x) <= cut.rhs + 1e-7,
                    "lift-and-project cut {cut:?} cut off {x:?}"
                );
            }
            assert!(
                cut.is_violated(&state.sol.x, 1e-6),
                "lift-and-project cut {cut:?} not violated at {:?}",
                state.sol.x
            );
        }
    }

    #[test]
    fn no_cuts_when_root_lp_integral() {
        // All-integer LP optimum: no fractional source rows, no cuts.
        let mut m = Model::new(2);
        m.set_objective(Objective::maximize(vec![0, 1], vec![1.0, 1.0]));
        m.add_constraint(Constraint::le(vec![0, 1], vec![1.0, 1.0], 2.0));
        m.variables[0].bound = VarBound::binary();
        m.variables[1].bound = VarBound::binary();

        let lb = vec![0.0, 0.0];
        let ub = vec![1.0, 1.0];
        let state = solve_lp_state(&m, &lb, &ub, tol());
        assert_eq!(state.sol.status, crate::lp::LpStatus::Optimal);
        let cuts = gomory_cuts(&state, &[0, 1], 10);
        assert!(cuts.is_empty(), "integral LP must not yield cuts, got {cuts:?}");
    }

    #[test]
    fn solver_end_to_end_with_cuts() {
        // The full cut suite (model-space + tableau-space) must still solve
        // a small MILP to its true optimum.
        let mut m = Model::new(3);
        m.set_objective(Objective::maximize(vec![0, 1, 2], vec![5.0, 4.0, 3.0]));
        m.add_constraint(Constraint::le(vec![0, 1, 2], vec![2.0, 3.0, 1.0], 5.0));
        m.add_constraint(Constraint::le(vec![0, 1, 2], vec![4.0, 1.0, 2.0], 11.0));
        m.add_constraint(Constraint::le(vec![0, 1, 2], vec![3.0, 4.0, 2.0], 12.0));
        for v in m.variables.iter_mut() {
            v.bound = VarBound::binary();
        }
        // Brute-force the true optimum.
        let mut best = f64::NEG_INFINITY;
        for mask in 0..8u32 {
            let x: Vec<f64> = (0..3).map(|i| ((mask >> i) & 1) as f64).collect();
            if m.constraints.iter().all(|c| c.is_satisfied(&x, 1e-9)) {
                best = best.max(m.objective.eval(&x));
            }
        }
        let mut solver = crate::MilpSolver::new().with_cuts(true).with_parallel_cuts(2);
        let sol = solver.solve(&m).unwrap();
        assert_eq!(sol.status, tpt_opt_core::solver::SolverStatus::Optimal);
        assert!(
            (sol.objective_value - best).abs() < 1e-6,
            "obj {} != brute force {best}",
            sol.objective_value
        );
    }
}
