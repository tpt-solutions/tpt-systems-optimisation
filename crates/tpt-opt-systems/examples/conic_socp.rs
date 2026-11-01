//! Maximise `x + y` over the unit disc, written as a second-order cone.
//!
//! The constraint `‖(x, y)‖₂ ≤ 1` is the cone `‖q(x)‖₂ ≤ r(x)` with
//! `r(x) = 1` and `q(x) = (x, y)`. The optimum is `x = y = 1/√2` with value `√2`.
//!
//! Run with: `cargo run -p tpt-opt-systems --example conic_socp --features conic`

use tpt_opt_systems::conic::{solve_socp, ConeProgram, ConicStatus, SocRow};
use tpt_opt_systems::core::Sense;

fn main() {
    let prog = ConeProgram {
        n: 2,
        c: vec![1.0, 1.0],
        sense: Sense::Maximize,
        bounds: vec![(-2.0, 2.0), (-2.0, 2.0)],
        eq_a: vec![],
        eq_b: vec![],
        soc_rows: vec![SocRow {
            // Row 0 is r(x) = 1; rows 1 and 2 are q(x) = (x, y).
            q_mat: vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
            q_rhs: vec![1.0, 0.0, 0.0],
        }],
        sdp_blocks: vec![],
    };

    let sol = solve_socp(&prog, 1e-8, 200);
    println!("status: {:?}", sol.status);
    println!("x = {:?}", sol.x);
    println!("objective = {:.9} (expected √2 = {:.9})", sol.objective, 2f64.sqrt());

    assert_eq!(sol.status, ConicStatus::Optimal);
    assert!((sol.objective - 2f64.sqrt()).abs() < 1e-4);
}
