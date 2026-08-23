//! Multi-objective optimisation through the umbrella crate: trace the exact
//! Pareto front of a linear production-planning problem by sweeping an
//! ε-constraint scalarisation, then apply the decision utilities —
//! hypervolume, knee-point detection and solution clustering — to pick and
//! summarise a compromise.
//!
//! Run with: cargo run --manifest-path examples/Cargo.toml --example multi_scalarize

use tpt_opt_systems::core::Solver;
use tpt_opt_systems::multi::{
    cluster_solutions, hypervolume, knee_point, pareto_front, LinearMultiObjective,
};
use tpt_opt_systems::multi::scalarize::term;
use tpt_opt_systems::{milp, Constraint};

fn main() {
    // Two products on one production line (capacity a + b <= 8):
    //   product A: profit 3/unit, pollution 1/unit
    //   product B: profit 2/unit, pollution 3/unit
    // Objectives (both minimised): f1 = -profit, f2 = pollution.
    let prob = LinearMultiObjective::new(
        vec![term(vec![-3.0, -2.0], 0.0), term(vec![1.0, 3.0], 0.0)],
        vec![(0.0, 8.0), (0.0, 8.0)],
    );

    // --- trace the Pareto front by ε-constraint sweeps ------------------------
    // `LinearMultiObjective` carries box bounds only, so the shared capacity
    // row is added to each generated ε-model before solving — the builders
    // are deliberately composable building blocks.
    let caps = [2.0f64, 4.0, 6.0, 8.0];
    let mut points: Vec<Vec<f64>> = Vec::new();
    println!("epsilon-constraint sweep (pollution cap -> max profit):");
    for cap in caps {
        let mut model = prob.epsilon_constraint_model(0, &[f64::INFINITY, cap]);
        model.add_constraint(Constraint::le(vec![0, 1], vec![1.0, 1.0], 8.0));
        let sol = milp::MilpSolver::new().solve(&model).expect("epsilon-model solves");
        let f = prob.evaluate(&sol.primal[..2]);
        println!("  cap {:>4.1} -> profit {:>5.1}, pollution {:>4.1}", cap, -f[0], f[1]);
        points.push(f);
    }

    // --- Pareto extraction -----------------------------------------------------
    let front = pareto_front(&points);
    assert_eq!(front.len(), caps.len(), "every swept point is Pareto-optimal");

    // --- hypervolume against a worse-than-worst reference ----------------------
    let reference = vec![0.0, 10.0];
    let hv = hypervolume(&points, &reference);
    println!("hypervolume w.r.t. {reference:?}: {hv:.2}");
    assert!(hv > 0.0);

    // --- knee point: the balanced compromise ------------------------------------
    let knee = knee_point(&points).expect("non-empty front");
    println!(
        "knee point: profit {:.1}, pollution {:.1}",
        -points[knee][0],
        points[knee][1]
    );

    // --- clustering: group the front into low/high-pollution strategies --------
    let assign = cluster_solutions(&points, 2);
    let groups: Vec<Vec<usize>> =
        (0..2).map(|c| assign.iter().enumerate().filter(|(_, &g)| g == c).map(|(i, _)| i).collect()).collect();
    for (c, members) in groups.iter().enumerate() {
        let pts: Vec<String> =
            members.iter().map(|&i| format!("({:.0},{:.0})", -points[i][0], points[i][1])).collect();
        println!("cluster {c}: {}", pts.join(" "));
    }
    assert_eq!(groups.iter().map(|g| g.len()).sum::<usize>(), caps.len());
}