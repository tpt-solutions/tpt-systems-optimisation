//! Recover four poses on a unit square from relative (odometry) measurements.
//!
//! The poses form a closed loop. The solver starts from a deliberately wrong
//! guess, and the loop closure plus a prior at the origin pull every pose back
//! onto the truth. With noise-free measurements the solution is exact.
//!
//! Run with: `cargo run -p tpt-opt-systems --example factorgraph_square_loop --features factorgraph`

use std::f64::consts::{FRAC_PI_2, PI};

use tpt_opt_systems::factorgraph::factors::{BetweenFactor, PriorFactor};
use tpt_opt_systems::factorgraph::graph::FactorGraph;
use tpt_opt_systems::factorgraph::manifold::Manifold;
use tpt_opt_systems::factorgraph::noise::NoiseModel;
use tpt_opt_systems::factorgraph::solver::{solve, FactorSolverConfig};

fn main() {
    // SE(2) poses as (x, y, θ).
    let truth = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, FRAC_PI_2], [0.0, 1.0, PI]];
    let se2 = Manifold::Se2;

    let mut graph = FactorGraph::new();
    for (i, t) in truth.iter().enumerate() {
        // Start every pose 0.3 away from its true position.
        let guess = vec![t[0] + 0.3, t[1] - 0.3, t[2] + 0.1];
        graph.add_variable(i, Manifold::Se2, guess).expect("unique key");
    }

    graph
        .add_factor(
            Box::new(PriorFactor::new(0, Manifold::Se2, vec![0.0, 0.0, 0.0])),
            NoiseModel::Isotropic { sigma: 0.01 },
            None,
        )
        .expect("prior");

    // Odometry along the square, plus the closing edge back to pose 0.
    for i in 0..truth.len() {
        let j = (i + 1) % truth.len();
        let measured = se2.compose(&se2.inverse(&truth[i]), &truth[j]);
        graph
            .add_factor(
                Box::new(BetweenFactor::new(i, j, Manifold::Se2, measured.to_vec())),
                NoiseModel::Isotropic { sigma: 0.05 },
                None,
            )
            .expect("between factor");
    }

    let outcome = solve(&mut graph, &FactorSolverConfig::new()).expect("solvable graph");
    println!("status: {:?}, cost: {:e}", outcome.status, outcome.cost);

    let mut worst: f64 = 0.0;
    for (i, t) in truth.iter().enumerate() {
        let v = graph.values.get(i).expect("pose present");
        println!("pose {i}: ({:.5}, {:.5}, {:.5})", v[0], v[1], v[2]);
        worst = worst.max((v[0] - t[0]).hypot(v[1] - t[1]));
    }
    assert!(worst < 1e-3, "worst position error {worst}");
}
