//! Performance regression report for the bundled MILP engine.
//!
//! Ignored by default (`cargo test -- --ignored`) so ordinary test runs and
//! CI gates stay timing-free. Run it manually or from the non-blocking
//! `perf-report` CI job:
//!
//! ```text
//! cargo test -p tpt-opt-milp --test perf_regression -- --ignored --nocapture
//! ```
//!
//! The suite solves a fixed set of instances, prints a wall-time table for
//! eyeballing regressions across commits, and asserts only *correctness*
//! (optimal status) — never timings — so it can run as a non-blocking report
//! without flaking.

use std::time::Instant;

use tpt_opt_core::{
    bounds::VarBound,
    model::{Constraint, Model, Objective},
    Solver, SolverStatus,
};

/// Build a 0/1 knapsack with `n` items (deterministic pseudo-random data).
fn knapsack(n: usize) -> Model {
    let mut m = Model::with_name(n, "knapsack");
    let vars: Vec<usize> = (0..n).collect();
    let weights: Vec<f64> = (0..n).map(|i| 1.0 + ((i * 37 + 11) % 23) as f64).collect();
    let values: Vec<f64> = (0..n).map(|i| 1.0 + ((i * 53 + 7) % 29) as f64).collect();
    let capacity: f64 = weights.iter().sum::<f64>() * 0.4;
    m.add_constraint(Constraint::le(vars.clone(), weights.clone(), capacity));
    m.set_objective(Objective::maximize(vars, values));
    for i in 0..n {
        m.variables[i].bound = VarBound::binary();
    }
    m
}

/// Set-covering: min sum y_j s.t. every element covered by >= 1 chosen set.
fn covering(rows: usize, cols: usize) -> Model {
    let mut m = Model::with_name(cols, "covering");
    let ys: Vec<usize> = (0..cols).collect();
    for r in 0..rows {
        let members: Vec<usize> = (0..cols).filter(|c| (r * 31 + c * 17 + 5) % 3 == 0).collect();
        let members = if members.is_empty() { vec![r % cols] } else { members };
        let coefs = vec![1.0; members.len()];
        m.add_constraint(Constraint::ge(members, coefs, 1.0));
    }
    let ones = vec![1.0; cols];
    m.set_objective(Objective::minimize(ys, ones));
    for j in 0..cols {
        m.variables[j].bound = VarBound::binary();
    }
    m
}

#[test]
#[ignore = "performance report; run explicitly with --ignored"]
fn perf_report_fixed_instances() {
    let cases: Vec<(&str, Model)> = vec![
        ("knapsack-15", knapsack(15)),
        ("knapsack-30", knapsack(30)),
        ("knapsack-60", knapsack(60)),
        ("covering-20x25", covering(20, 25)),
        ("covering-40x50", covering(40, 50)),
    ];

    println!("{:<18} {:>10} {:>12} {:>10}", "instance", "status", "objective", "time_ms");
    println!("{}", "-".repeat(54));

    for (name, model) in &cases {
        let mut solver = tpt_opt_milp::MilpSolver::new();
        let start = Instant::now();
        let sol = solver.solve(model).expect("benchmark instance should solve");
        let elapsed = start.elapsed();

        assert_eq!(sol.status, SolverStatus::Optimal, "{name} must be optimal");
        println!(
            "{:<18} {:>10} {:>12.4} {:>10.1}",
            name,
            format!("{:?}", sol.status),
            sol.objective_value,
            elapsed.as_secs_f64() * 1000.0
        );
    }

    println!(
        "
(note: timings are informational only — no assertion is made on them)"
    );
}
