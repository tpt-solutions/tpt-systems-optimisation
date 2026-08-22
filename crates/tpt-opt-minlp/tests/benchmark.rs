//! Integration benchmarks for `tpt-opt-minlp`.
//!
//! Two end-to-end checks mirroring the validation style of MINLPLib
//! benchmarking (a full MINLPLib corpus runner remains future work — see
//! todo.md):
//!
//! 1. **Cross-solver agreement** on a convex MINLP with a known analytic
//!    optimum: outer approximation and generalized Benders decomposition
//!    must independently converge to the same optimal value.
//! 2. **Enumeration cross-check** on a non-convex instance: SQP-style
//!    branch-and-bound must match the best value found by fixing every
//!    integer assignment and solving the resulting NLPs directly.

use tpt_math_optimize_general::NlpParams;
use tpt_opt_minlp::gbd::{generalized_benders, GbdConfig};
use tpt_opt_minlp::model::{MinlpModel, VarKind};
use tpt_opt_minlp::oa::{outer_approximate, OaConfig};
use tpt_opt_minlp::sqp::{sqp_branch_and_bound, SqpConfig};
use tpt_opt_minlp::subproblem::solve_subproblem;

/// Convex MINLP with a unique analytic optimum:
///
/// ```text
/// min   x² + 5y     s.t.  x + y = 4,
/// x ∈ [0,4] continuous, y ∈ ℤ ∩ [0,4].
/// ```
///
/// Substituting x = 4 − y: objective (4−y)² + 5y over y ∈ {0,…,4} gives
/// 16, 16, **14**, 16, 20 → optimum **14** at (x, y) = (2, 2).
fn convex_tradeoff_model() -> MinlpModel {
    let mut m = MinlpModel::new(2, |x| x[0] * x[0] + 5.0 * x[1]);
    m.set_var(0, VarKind::Continuous, 0.0, 4.0);
    m.set_var(1, VarKind::Integer, 0.0, 4.0);
    m.add_eq(
        |x| x[0] + x[1] - 4.0,
        |_x, g| {
            g[0] = 1.0;
            g[1] = 1.0;
        },
    );
    m
}

#[test]
fn oa_and_gbd_agree_on_convex_instance() {
    const OPT: f64 = 14.0;

    let oa = outer_approximate(
        &convex_tradeoff_model(),
        &OaConfig { max_iter: 80, ..OaConfig::default() },
    );
    assert_eq!(oa.status, tpt_opt_minlp::OaStatus::Optimal, "OA history: {:?}", oa.history);
    let oa_obj = oa.objective.expect("OA incumbent");
    assert!((oa_obj - OPT).abs() < 1e-3, "OA obj {oa_obj}");
    let oa_x = oa.x.unwrap();
    assert!((oa_x[0] - 2.0).abs() < 1e-2 && (oa_x[1] - 2.0).abs() < 1e-6, "OA x {oa_x:?}");

    let gbd = generalized_benders(
        &convex_tradeoff_model(),
        &GbdConfig { max_iter: 100, ..GbdConfig::default() },
    );
    assert_eq!(gbd.status, tpt_opt_minlp::OaStatus::Optimal, "GBD history: {:?}", gbd.history);
    let gbd_obj = gbd.objective.expect("GBD incumbent");
    assert!((gbd_obj - OPT).abs() < 1e-3, "GBD obj {gbd_obj}");

    // Independent solvers must agree on the optimal value.
    assert!((oa_obj - gbd_obj).abs() < 1e-3, "OA {oa_obj} vs GBD {gbd_obj}");
}

/// Non-convex (bilinear-objective) MINLP:
///
/// ```text
/// min   x·y        s.t.  x + y >= 3,
/// x ∈ [0.5,3] continuous, y ∈ ℤ ∩ [1,3].
/// ```
///
/// Enumerating integer assignments: y=1 forces x>=2 (obj ≥ 2), y=2 forces
/// x>=1 (obj ≥ 2), y=3 allows x=0.5 (obj = 1.5) → optimum **1.5**.
fn nonconvex_model() -> MinlpModel {
    let mut m = MinlpModel::new(2, |x| x[0] * x[1]);
    m.set_var(0, VarKind::Continuous, 0.5, 3.0);
    m.set_var(1, VarKind::Integer, 1.0, 3.0);
    m.add_le(
        |x| 3.0 - x[0] - x[1],
        |_x, g| {
            g[0] = -1.0;
            g[1] = -1.0;
        },
    );
    m
}

#[test]
fn sqp_bb_matches_integer_enumeration_on_nonconvex_instance() {
    // Brute force: fix each integer assignment, solve the continuous NLP.
    let model = nonconvex_model();
    let nlp_cfg = NlpParams { tol: 1e-8, ..NlpParams::default() };
    let mut best_enum = f64::INFINITY;
    for y in 1..=3 {
        let yfix = [0.0, y as f64];
        let res = solve_subproblem(&model, &yfix, &nlp_cfg);
        // Judge feasibility by violation (as the solvers do), not solely by
        // the NLP status flag.
        let v = tpt_opt_minlp::subproblem::max_violation(&model, &yfix, &res.x);
        eprintln!("y={y} status={:?} obj={} x={:?} viol={v:.3e}", res.status, res.objective, res.x);
        if v < 1e-6 {
            best_enum = best_enum.min(res.objective);
        }
    }
    assert!((best_enum - 1.5).abs() < 1e-3, "enumeration best {best_enum}");

    let bb = sqp_branch_and_bound(&model, &SqpConfig { max_nodes: 500, ..SqpConfig::default() });
    assert_eq!(bb.status, tpt_opt_minlp::OaStatus::Optimal);
    let bb_obj = bb.objective.expect("B&B incumbent");
    assert!(
        (bb_obj - best_enum).abs() < 1e-3,
        "branch-and-bound {bb_obj} vs enumeration {best_enum}"
    );
    assert!((bb_obj - 1.5).abs() < 1e-3, "B&B obj {bb_obj}");
}
