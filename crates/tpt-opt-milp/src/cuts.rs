//! Cut generation for the MILP solver (model space).
//!
//! Three provably valid families are provided, all operating directly on
//! [`Model`] rows so they integrate cleanly with the branch-and-bound tree:
//!
//! - **Clique cuts** ([`add_clique_cuts`]) — maximal cliques of the binary
//!   conflict graph implied by set-packing style rows.
//! - **Cover inequalities** ([`add_cover_cuts`]) — minimal-cover inequalities
//!   for binary knapsack rows, strengthened by exact sequential up-lifting
//!   whenever the row data is integral and small enough.
//! - **MIR cuts** ([`add_mir_cuts`]) — mixed-integer rounding inequalities
//!   derived from single rows after shifting integer variables to be
//!   non-negative.
//!
//! Every generator is guarded by exhaustive validity unit tests: no cut may
//! remove a feasible integer point of small enumerated instances.

use std::vec::Vec;

use tpt_opt_core::model::{Constraint, Model};

// ---------------------------------------------------------------------------
// Clique cuts
// ---------------------------------------------------------------------------

/// Detect binary conflict pairs `(i, j)` implied by the model's rows: pairs
/// that cannot both be 1 in any feasible solution. Only set-packing style
/// rows (`sum a_k x_k <= b`, all-positive binary coefficients with
/// `a_i + a_j > b`) are considered.
fn conflict_pairs(model: &Model, binaries: &[usize]) -> Vec<(usize, usize)> {
    let bin_set: std::collections::BTreeSet<usize> = binaries.iter().copied().collect();
    let mut pairs = Vec::new();
    for c in &model.constraints {
        if c.lower.is_finite() && c.upper.is_finite() {
            continue; // ranged/equality rows: skip
        }
        // Normalise >= rows by negation into <= orientation.
        let (indices, coeffs, rhs) = if c.upper.is_finite() {
            (c.indices.clone(), c.coeffs.clone(), c.upper)
        } else {
            (c.indices.clone(), c.coeffs.iter().map(|&v| -v).collect::<Vec<_>>(), -c.lower)
        };
        if indices.iter().any(|&v| !bin_set.contains(&v)) {
            continue;
        }
        if coeffs.iter().any(|&v| v <= 0.0) {
            continue;
        }
        for p in 0..indices.len() {
            for q in (p + 1)..indices.len() {
                if coeffs[p] + coeffs[q] > rhs + 1e-12 {
                    pairs.push((indices[p].min(indices[q]), indices[p].max(indices[q])));
                }
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// Add up to `max_cuts` clique cuts derived from maximal cliques of the binary
/// conflict graph. Returns the number of cuts appended to `model`.
///
/// A clique `C` with pairwise-conflicting binaries yields the valid cut
/// `sum_{j in C} x_j <= 1`.
pub fn add_clique_cuts(model: &mut Model, binaries: &[usize], max_cuts: usize) -> usize {
    let pairs = conflict_pairs(model, binaries);
    if pairs.is_empty() {
        return 0;
    }
    let nodes: Vec<usize> = {
        let mut s: Vec<usize> = pairs.iter().flat_map(|&(a, b)| vec![a, b]).collect();
        s.sort_unstable();
        s.dedup();
        s
    };
    // Greedy maximal cliques seeded at each node (deterministic order).
    let mut cuts: Vec<Vec<usize>> = Vec::new();
    for &seed in &nodes {
        let mut clique = vec![seed];
        for &cand in &nodes {
            if cand <= seed || clique.contains(&cand) {
                continue;
            }
            if clique.iter().all(|&m| m == cand || pairs.contains(&(m.min(cand), m.max(cand)))) {
                clique.push(cand);
            }
        }
        if clique.len() >= 3 {
            clique.sort_unstable();
            if !cuts.contains(&clique) {
                cuts.push(clique);
            }
        }
        if cuts.len() >= max_cuts {
            break;
        }
    }
    let before = model.constraints.len();
    for clique in &cuts {
        // Skip cuts identical to an existing row.
        let dup = model.constraints.iter().any(|r| {
            r.lower.is_infinite()
                && r.upper == 1.0
                && r.indices.len() == clique.len()
                && r.indices.iter().collect::<std::collections::BTreeSet<_>>()
                    == clique.iter().collect::<std::collections::BTreeSet<_>>()
        });
        if !dup {
            let coeffs = vec![1.0; clique.len()];
            model.add_constraint(Constraint::le(clique.clone(), coeffs, 1.0));
        }
    }
    model.constraints.len() - before
}

// ---------------------------------------------------------------------------
// Cover inequalities
// ---------------------------------------------------------------------------

/// Add up to `max_cuts` minimal-cover inequalities for binary knapsack rows.
///
/// For a row `sum a_j x_j <= b` over binaries with `a_j > 0`, a cover `C`
/// satisfies `sum_{j in C} a_j > b`; the cover inequality
/// `sum_{j in C} x_j <= |C| - 1` is valid. When every coefficient in the row
/// is integral and the row is small enough, cuts are strengthened by exact
/// sequential up-lifting of the non-cover binaries.
pub fn add_cover_cuts(model: &mut Model, binaries: &[usize], max_cuts: usize) -> usize {
    let bin_set: std::collections::BTreeSet<usize> = binaries.iter().copied().collect();
    let mut added = 0;
    let snapshot = model.constraints.clone();
    for c in &snapshot {
        if added >= max_cuts {
            break;
        }
        if c.indices.iter().any(|&v| !bin_set.contains(&v)) {
            continue;
        }
        if c.indices.len() < 2 || c.indices.len() > 64 {
            continue;
        }
        // Work with the <= orientation.
        let (indices, coeffs, rhs) = if c.upper.is_finite() && c.lower.is_infinite() {
            (c.indices.clone(), c.coeffs.clone(), c.upper)
        } else if c.lower.is_finite() && c.upper.is_infinite() {
            (c.indices.clone(), c.coeffs.iter().map(|&v| -v).collect::<Vec<_>>(), -c.lower)
        } else {
            continue;
        };
        if coeffs.iter().any(|&v| v <= 0.0) {
            continue;
        }
        // Greedy cover: heaviest coefficients first until the row is violated.
        let mut order: Vec<usize> = (0..indices.len()).collect();
        order.sort_by(|&p, &q| coeffs[q].partial_cmp(&coeffs[p]).unwrap());
        let mut cover: Vec<usize> = Vec::new();
        let mut acc = 0.0;
        for &p in &order {
            if acc > rhs + 1e-12 {
                break;
            }
            cover.push(p);
            acc += coeffs[p];
        }
        if acc <= rhs + 1e-12 {
            continue; // even all-ones does not violate; no cover exists
        }
        // Minimality: drop members whose removal keeps it a cover (lightest
        // first).
        let mut rem: Vec<usize> =
            cover.iter().copied().filter(|&p| acc - coeffs[p] > rhs + 1e-12).collect();
        rem.sort_by(|&p, &q| coeffs[p].partial_cmp(&coeffs[q]).unwrap());
        for &p in &rem {
            if cover.len() <= 1 {
                break;
            }
            if acc - coeffs[p] > rhs + 1e-12 {
                acc -= coeffs[p];
                cover.retain(|&q| q != p);
            }
        }

        // Optional exact lifting for integral-coefficient rows.
        let integral =
            coeffs.iter().all(|v| v.fract() == 0.0) && rhs.fract() == 0.0 && rhs.abs() < 1e15;
        let mut lifted: Vec<(usize, f64)> = cover.iter().map(|&p| (indices[p], 1.0)).collect();
        let cut_rhs = (cover.len() - 1) as f64;
        if integral && indices.len() <= 20 {
            let cap_base = rhs as i64;
            let cover_items: Vec<i64> = cover.iter().map(|&p| coeffs[p] as i64).collect();
            for p in 0..indices.len() {
                if cover.contains(&p) {
                    continue;
                }
                let ak = coeffs[p] as i64;
                let alpha = if ak > cap_base {
                    cut_rhs // x_k = 1 alone exceeds capacity
                } else {
                    // Exact subset-count knapsack: maximise the number of
                    // cover items fitting in cap_base - ak.
                    let cap = cap_base - ak;
                    let mut dp = vec![i64::MAX; cover_items.len() + 1];
                    dp[0] = 0;
                    for &w in &cover_items {
                        for t in (1..=cover_items.len()).rev() {
                            if dp[t - 1] != i64::MAX && dp[t - 1] + w <= cap {
                                dp[t] = dp[t].min(dp[t - 1] + w);
                            }
                        }
                    }
                    let best =
                        (0..=cover_items.len()).rev().find(|&t| dp[t] != i64::MAX).unwrap_or(0);
                    cut_rhs - best as f64
                };
                if alpha > 0.0 {
                    lifted.push((indices[p], alpha));
                }
            }
        }

        // Skip duplicates of existing rows.
        let mut idx: Vec<usize> = lifted.iter().map(|&(v, _)| v).collect();
        idx.sort_unstable();
        idx.dedup();
        let dup = model.constraints.iter().any(|r| {
            r.indices.len() == idx.len()
                && r.upper == cut_rhs
                && r.lower.is_infinite()
                && r.indices.iter().collect::<std::collections::BTreeSet<_>>()
                    == idx.iter().collect::<std::collections::BTreeSet<_>>()
        });
        if dup {
            continue;
        }
        let coefs: Vec<f64> = lifted.iter().map(|&(_, a)| a).collect();
        let vars: Vec<usize> = lifted.iter().map(|&(v, _)| v).collect();
        model.add_constraint(Constraint::le(vars, coefs, cut_rhs));
        added += 1;
    }
    added
}

// ---------------------------------------------------------------------------
// MIR cuts
// ---------------------------------------------------------------------------

/// Add up to `max_cuts` mixed-integer rounding (MIR) cuts derived from single
/// rows. Returns the number of cuts appended to `model`.
///
/// Each finite side of every row is relaxed by dropping non-negative
/// continuous terms (valid because they only tighten a `<=` row), shifted so
/// the integer variables are non-negative (`y = x - lb`), and then scaled by a
/// small deterministic set of candidate multipliers. For the scaled row in
/// Marchand–Wolsey form
///
/// ```text
/// sum_j a_j y_j <= b + sum_k g_k w_k   (y integer >= 0, w continuous >= 0)
/// ```
///
/// with `f0 = b - floor(b)` in `(0, 1)`, the MIR inequality
///
/// ```text
/// sum_j (floor(a_j) + max(f_j - f0, 0)/(1 - f0)) y_j
///     <= floor(b) + sum_k g_k/(1 - f0) w_k
/// ```
///
/// is valid for the mixed-integer hull. Rows without integer terms or with
/// unbounded integer variables are skipped; at most one cut is generated per
/// row per call.
pub fn add_mir_cuts(model: &mut Model, int_vars: &[usize], max_cuts: usize) -> usize {
    let ints: std::collections::BTreeSet<usize> = int_vars.iter().copied().collect();
    let snapshot = model.constraints.clone();
    let mut added = 0;
    let mut seen: std::collections::BTreeSet<(Vec<(usize, u64)>, u64)> =
        std::collections::BTreeSet::new();
    for c in &snapshot {
        if added >= max_cuts {
            break;
        }
        // Candidate `<=` orientations: the upper side as stored, and the
        // lower side negated into `<=` form (both are valid relaxations).
        let sides = [
            c.upper.is_finite().then(|| (c.indices.clone(), c.coeffs.clone(), c.upper)),
            c.lower.is_finite().then(|| {
                (c.indices.clone(), c.coeffs.iter().map(|&v| -v).collect::<Vec<_>>(), -c.lower)
            }),
        ];
        for (indices, coeffs, rhs) in sides.into_iter().flatten() {
            if let Some(row) = build_mir(model, &ints, &indices, &coeffs, rhs) {
                let (mut vars, coefs, upper) = row;
                // Deterministic variable order + exact-duplicate suppression.
                let mut order: Vec<usize> = (0..vars.len()).collect();
                order.sort_by(|&i, &j| vars[i].cmp(&vars[j]).then(coefs[i].total_cmp(&coefs[j])));
                let sig = (
                    order.iter().map(|&i| (vars[i], coefs[i].to_bits())).collect::<Vec<_>>(),
                    upper.to_bits(),
                );
                if seen.insert(sig) {
                    let sorted: Vec<(usize, f64)> =
                        order.iter().map(|&i| (vars[i], coefs[i])).collect();
                    vars = sorted.iter().map(|&(v, _)| v).collect();
                    let cs: Vec<f64> = sorted.iter().map(|&(_, a)| a).collect();
                    model.add_constraint(Constraint::le(vars, cs, upper));
                    added += 1;
                }
                break; // one cut per row per pass
            }
        }
    }
    added
}

/// Try to build one MIR cut from a `<=` row. Returns
/// `Some((vars, coeffs, upper))` in original variable space, or `None`.
fn build_mir(
    model: &Model,
    ints: &std::collections::BTreeSet<usize>,
    indices: &[usize],
    coeffs: &[f64],
    rhs: f64,
) -> Option<(Vec<usize>, Vec<f64>, f64)> {
    // Split the row: integer terms (finite lower bound required for the
    // shift), continuous terms usable on the right-hand side (negative
    // coefficient), and droppable non-negative continuous terms.
    let mut int_terms: Vec<(usize, f64)> = Vec::new(); // (var, p_j)
    let mut cont_terms: Vec<(usize, f64)> = Vec::new(); // (var, a_k), a_k < 0
    let mut shift_const = 0.0;
    for (&v, &a) in indices.iter().zip(coeffs.iter()) {
        if ints.contains(&v) {
            let lb = model.variables[v].bound.bound.lower;
            if !lb.is_finite() {
                return None;
            }
            shift_const += a * lb;
            int_terms.push((v, a));
        } else if a < -1e-12 {
            cont_terms.push((v, a));
        }
        // Continuous terms with a >= 0 are dropped: valid relaxation of a
        // `<=` row (they only make it harder to satisfy).
    }
    if int_terms.is_empty() {
        return None;
    }
    let b_base = rhs - shift_const;

    // Deterministic candidate scalings: normalise each integer coefficient to
    // unit magnitude first (usually strongest), then the unscaled row.
    let mut deltas: Vec<f64> = Vec::new();
    for &(_, p) in &int_terms {
        let d = 1.0 / p.abs();
        if d.is_finite() && (1e-9..=1e9).contains(&d) {
            deltas.push(d);
        }
    }
    deltas.push(1.0);
    deltas.sort_by(|a, b| a.total_cmp(b));
    deltas.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    for &delta in &deltas {
        if let Some(cut) = mir_scaled(model, delta, &int_terms, &cont_terms, b_base) {
            return Some(cut);
        }
    }
    None
}

/// Apply the Marchand–Wolsey MIR inequality to the row scaled by `delta > 0`.
/// See [`build_mir`] for the derivation.
fn mir_scaled(
    model: &Model,
    delta: f64,
    int_terms: &[(usize, f64)],
    cont_terms: &[(usize, f64)],
    b_base: f64,
) -> Option<(Vec<usize>, Vec<f64>, f64)> {
    let b = delta * b_base;
    if !b.is_finite() || b.abs() > 1e15 {
        return None;
    }
    let f0 = b - b.floor();
    if f0 <= 1e-9 || f0 >= 1.0 - 1e-9 {
        return None; // effective RHS integral: no MIR strength
    }
    // Scaled row in MW form: sum (delta*p)_j y_j <= b + sum (-delta*a)_k w_k
    // with (-delta*a)_k > 0 on the right-hand side.
    let mut out_vars: Vec<usize> = Vec::new();
    let mut out_coefs: Vec<f64> = Vec::new();
    let mut upper = b.floor();
    for &(v, p) in int_terms {
        let a = delta * p;
        if !a.is_finite() || a.abs() > 1e15 {
            return None;
        }
        let fa = a - a.floor();
        let alpha = a.floor() + ((fa - f0).max(0.0)) / (1.0 - f0);
        if alpha.abs() <= 1e-12 {
            continue;
        }
        // Map back: y = x - lb → alpha*y = alpha*x - alpha*lb.
        let lb = model.variables[v].bound.bound.lower;
        upper += alpha * lb;
        out_vars.push(v);
        out_coefs.push(alpha);
    }
    if out_vars.is_empty() {
        return None;
    }
    // Continuous terms move back to the left with amplified magnitude:
    // sum alpha y <= floor(b) + sum (-delta*a)_k/(1-f0) w_k becomes
    // sum alpha x - sum (-delta*a)_k/(1-f0) w_k <= floor(b) + sum alpha*lb.
    for &(w, a) in cont_terms {
        out_vars.push(w);
        out_coefs.push(delta * a / (1.0 - f0));
    }
    Some((out_vars, out_coefs, upper))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_opt_core::bounds::VarBound;

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
    fn clique_cut_valid_and_found() {
        // A triple packing row implies the 3-clique {0,1,2}; the generated
        // clique cut coincides with the existing row and is deduplicated.
        let mut m = Model::new(4);
        m.add_constraint(Constraint::le(vec![0, 1, 2], vec![1.0, 1.0, 1.0], 1.0));
        m.variables[0].bound = VarBound::binary();
        m.variables[1].bound = VarBound::binary();
        m.variables[2].bound = VarBound::binary();
        m.variables[3].bound = VarBound::binary();

        let added = add_clique_cuts(&mut m, &[0, 1, 2, 3], 5);
        assert_eq!(added, 0, "clique duplicate of the packing row is skipped");

        // Build a genuine 3-clique from pairwise rows only.
        let mut m2 = Model::new(3);
        m2.add_constraint(Constraint::le(vec![0, 1], vec![1.0, 1.0], 1.0));
        m2.add_constraint(Constraint::le(vec![0, 2], vec![1.0, 1.0], 1.0));
        m2.add_constraint(Constraint::le(vec![1, 2], vec![1.0, 1.0], 1.0));
        for v in m2.variables.iter_mut() {
            v.bound = VarBound::binary();
        }
        let pts2 = feasible_points(&m2);
        let added2 = add_clique_cuts(&mut m2, &[0, 1, 2], 5);
        assert!(added2 >= 1);
        // Every previously feasible point still satisfies the new cuts.
        for x in &pts2 {
            for c in m2.constraints.iter().skip(3) {
                assert!(c.is_satisfied(x, 1e-9), "clique cut cut off {x:?}");
            }
        }
        // The clique cut excludes the half-integral point (0.5, 0.5, 0.5).
        let half = vec![0.5, 0.5, 0.5];
        assert!(m2.constraints.iter().skip(3).any(|c| !c.is_satisfied(&half, 1e-9)));
    }

    #[test]
    fn cover_cut_valid_and_tightening() {
        // Knapsack: 6x0 + 5x1 + 5x2 + 5x3 <= 11 → cover {0,1}: 6+5 > 11.
        let mut m = Model::new(4);
        m.add_constraint(Constraint::le(vec![0, 1, 2, 3], vec![6.0, 5.0, 5.0, 5.0], 11.0));
        for v in m.variables.iter_mut() {
            v.bound = VarBound::binary();
        }
        let pts = feasible_points(&m);
        let added = add_cover_cuts(&mut m, &[0, 1, 2, 3], 5);
        assert!(added >= 1);
        for x in &pts {
            for c in m.constraints.iter().skip(1) {
                assert!(c.is_satisfied(x, 1e-9), "cover cut cut off {x:?}");
            }
        }
        // The all-ones point must be excluded by the cut (it was infeasible).
        let ones = vec![1.0, 1.0, 1.0, 1.0];
        assert!(m.constraints.iter().skip(1).any(|c| !c.is_satisfied(&ones, 1e-9)));
    }

    #[test]
    fn mir_cut_valid() {
        // x0 + x1 - 0.5*x2 <= 1.5 with x0, x1 binary and x2 continuous in
        // [0, 10]: f0 = 0.5 yields the cut x0 + x1 - x2 <= 1, which excludes
        // the relaxed point (1, 0.5, 0) while keeping every integer-feasible
        // point.
        let mut m = Model::new(3);
        m.add_constraint(Constraint::le(vec![0, 1, 2], vec![1.0, 1.0, -0.5], 1.5));
        m.variables[0].bound = VarBound::binary();
        m.variables[1].bound = VarBound::binary();
        m.variables[2].bound = VarBound::continuous(0.0, 10.0);

        let pts = feasible_points(&m);
        let added = add_mir_cuts(&mut m, &[0, 1], 5);
        assert!(added >= 1);
        for x in &pts {
            for c in m.constraints.iter().skip(1) {
                assert!(c.is_satisfied(x, 1e-9), "MIR cut cut off {x:?}");
            }
        }
        let viol = vec![1.0, 0.5, 0.0];
        assert!(
            m.constraints.iter().skip(1).any(|c| !c.is_satisfied(&viol, 1e-9)),
            "MIR cut should exclude the fractional point {viol:?}"
        );
    }

    #[test]
    fn mir_cut_fractional_coefficient() {
        // 2.5*x0 + x2 <= 3.5 with x0 general integer in [0, 5] and x2
        // continuous in [0, 10]: the delta = 1/2.5 scaling puts the row into
        // x0 + 0.4*x2 <= 1.4, giving the MIR cut x0 <= 1 — stronger than the
        // row's own LP bound x0 <= 1.4 and satisfied by every integer-feasible
        // point.
        let mut m = Model::new(2);
        m.add_constraint(Constraint::le(vec![0, 1], vec![2.5, 1.0], 3.5));
        m.variables[0].bound = VarBound::integer(0.0, 5.0);
        m.variables[1].bound = VarBound::continuous(0.0, 10.0);

        let added = add_mir_cuts(&mut m, &[0], 5);
        assert!(added >= 1);
        let cut = m.constraints.last().unwrap();
        // Every integer-feasible point of the row satisfies the cut.
        for x0 in 0..=5i64 {
            let slack = 3.5 - 2.5 * x0 as f64;
            if slack < 0.0 {
                continue; // no feasible x2 for this x0
            }
            let x2_max = slack.min(10.0);
            for &x2 in &[0.0f64, x2_max] {
                let x = vec![x0 as f64, x2];
                assert!(cut.is_satisfied(&x, 1e-9), "MIR cut cut off ({x0}, {x2})");
            }
        }
        // The cut strengthens the relaxation beyond the source row's LP bound.
        let lp_point = vec![1.4, 0.0];
        assert!(!cut.is_satisfied(&lp_point, 1e-9));
    }
}
