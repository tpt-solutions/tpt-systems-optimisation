//! Mixed-integer nonlinear programming through the umbrella crate:
//!
//! 1. a **convex** MINLP solved by outer approximation (MILP master ↔ NLP
//!    subproblems with duality-gap certificates),
//! 2. a **non-convex** MINLP solved by SQP branch-and-bound (no convexity
//!    assumption needed),
//! 3. a **McCormick envelope** relaxation of a bilinear term, showing the
//!    relaxation gap that branch-and-bound closes by tightening bounds.
//!
//! Run with: cargo run --manifest-path examples/Cargo.toml --example minlp_design

use tpt_opt_systems::core::Solver;
use tpt_opt_systems::minlp::{
    model::{MinlpModel, VarKind},
    oa::{outer_approximate, OaConfig, OaStatus},
    relax::{mccormick_envelope, FactorBounds},
    sqp::{sqp_branch_and_bound, SqpConfig},
};
use tpt_opt_systems::{milp, Constraint, Model, Objective, Sense, VarBound};

fn main() {
    // --- convex MINLP: outer approximation -----------------------------------
    // Process sizing: minimise material x plus whole units y of an auxiliary
    // module, where the module must absorb the square-deviation (x−1)².
    // Optimum 1.0 — attained at (x, y) = (0, 1) or (1, 0).
    let mut m = MinlpModel::new(2, |x| x[0] + x[1]);
    m.set_var(0, VarKind::Continuous, 0.0, 2.0);
    m.set_var(1, VarKind::Integer, 0.0, 4.0);
    m.add_le(
        |x| (x[0] - 1.0) * (x[0] - 1.0) - x[1],
        |x, g| {
            g[0] = 2.0 * (x[0] - 1.0);
            g[1] = -1.0;
        },
    );
    let res = outer_approximate(&m, &OaConfig::default());
    println!("outer approximation: {:?}, objective = {:?}", res.status, res.objective);
    assert_eq!(res.status, OaStatus::Optimal);
    assert!((res.objective.unwrap() - 1.0).abs() < 1e-3);

    // --- non-convex MINLP: SQP branch-and-bound ------------------------------
    // Double-well cost (x² − 4)² + 0.1·x over integer x ∈ [0, 3]: the global
    // minimum sits at x = 2 (value 0.2), not at the nearest well wall — a
    // shape outer approximation must not be trusted on without convexity.
    let mut nc = MinlpModel::new(1, |x| (x[0] * x[0] - 4.0).powi(2) + 0.1 * x[0]);
    nc.set_var(0, VarKind::Integer, 0.0, 3.0);
    let res = sqp_branch_and_bound(&nc, &SqpConfig::default());
    println!(
        "SQP branch-and-bound: {} nodes, objective = {:?}",
        res.nodes_explored, res.objective
    );
    assert!((res.objective.unwrap() - 0.2).abs() < 1e-3);

    // --- McCormick envelope: see the relaxation gap ---------------------------
    // Maximise w = x·y subject to x + y = 1 over [0, 1]². The true optimum is
    // 0.25 at x = y = 0.5; the four bilinear tangent planes alone admit w up
    // to 0.5. Branch-and-bound closes exactly this gap by splitting the
    // factor boxes and re-relaxing.
    let mut lp = Model::new(3);
    for v in lp.variables.iter_mut() {
        v.bound = VarBound::continuous(0.0, 1.0);
    }
    for row in mccormick_envelope(
        0,
        1,
        2,
        FactorBounds { lx: 0.0, ux: 1.0, ly: 0.0, uy: 1.0 },
    ) {
        lp.add_constraint(row);
    }
    lp.add_constraint(Constraint::equality(vec![0, 1], vec![1.0, 1.0], 1.0));
    lp.set_objective(Objective::maximize(vec![2], vec![1.0]));
    let sol = milp::MilpSolver::new().solve(&lp).expect("relaxation LP solves");
    println!(
        "McCormick relaxation bound = {:.4} (true bilinear optimum = 0.25)",
        sol.objective_value
    );
    assert!((sol.objective_value - 0.5).abs() < 1e-6);
    assert!(sol.objective_value >= 0.25); // relaxation never under-estimates max
    let _ = Sense::Minimize;
}