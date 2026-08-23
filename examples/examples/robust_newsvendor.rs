//! Optimisation under uncertainty through the umbrella crate: a two-stage
//! news-vendor problem solved as an extensive-form MILP, followed by the
//! classic value-of-information metrics — VSS (what stochastic optimisation
//! buys over planning with average demand) and EVPI (what perfect demand
//! foresight would buy).
//!
//! Run with: cargo run --manifest-path examples/Cargo.toml --example robust_newsvendor

use tpt_opt_systems::robust::scenario::{RowSense, Scenario, StageData, StageRow, TwoStageProblem};
use tpt_opt_systems::robust;

fn main() {
    // Buy-to-stock: order x units upfront at cost 2/unit; face demand d and
    // cover any shortfall y at spot cost 3/unit. Demand scenarios:
    //   3 units w.p. 0.5, 5 units w.p. 0.3, 10 units w.p. 0.2.
    let mk_scenario = |d: f64| {
        (
            Scenario { probability: match d as u32 { 3 => 0.5, 5 => 0.3, _ => 0.2 }, data: vec![] },
            StageData {
                cost: vec![3.0],
                rows: vec![StageRow {
                    w: vec![-1.0], // -x - y <= -d   <=>   x + y >= d
                    t: vec![-1.0],
                    h: -d,
                    sense: RowSense::Le,
                }],
            },
        )
    };
    let problem = TwoStageProblem {
        first_cost: vec![2.0],
        first_bounds: vec![(0.0, 10.0)],
        second_bounds: vec![(0.0, 10.0)],
        scenarios: vec![mk_scenario(3.0), mk_scenario(5.0), mk_scenario(10.0)],
    };

    // --- extensive form (the stochastic solution) ------------------------------
    let sol = problem.solve().expect("extensive form solves");
    println!("RP* (stochastic optimum) = {:.3} at order x = {:.1}", sol.objective, sol.x[0]);

    // --- value-of-information metrics ------------------------------------------
    let m = robust::value::value_metrics(&problem).expect("metrics computable");
    println!("WS  (wait-and-see, perfect foresight) = {:.3}", m.ws);
    println!("EEV (expected-value solution, evaluated truly) = {:.3}", m.eev);
    println!("VSS = EEV - RP = {:.3}", m.vss);
    println!("EVPI = RP - WS = {:.3}", m.evpi);

    // Hand-computed optima for this instance:
    //   RP* = 12 at x = 3, E[WS] = 10, EEV = 13  =>  VSS = 1, EVPI = 2.
    assert!((m.rp - 12.0).abs() < 1e-6);
    assert!((sol.x[0] - 3.0).abs() < 1e-6);
    assert!((m.ws - 10.0).abs() < 1e-6);
    assert!((m.eev - 13.0).abs() < 1e-6);
    assert!((m.vss - 1.0).abs() < 1e-9);
    assert!((m.evpi - 2.0).abs() < 1e-9);

    // Reading: planning with average demand (order 5) wastes money on excess
    // stock when the high-demand scenario hits; hedging at x = 3 costs less
    // expected spot purchases overall. Perfect foresight would save another 2.
}