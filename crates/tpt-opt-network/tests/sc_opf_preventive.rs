//! Co-optimised preventive SC-OPF on a three-bus triangle.
//!
//! Bus 0 (slack) has a cheap generator, bus 1 an expensive one, and bus 2 a
//! 100 MW load. Line 0 is the direct 0–2 path, line 1 is 1–2, and line 2 is 0–1.

use tpt_opt_core::SolverStatus;
use tpt_opt_network::{dc_opf, sc_opf_preventive, Bus, Generator, Line, Network};

fn triangle(cap_direct: f64, cap_to_load: f64, cap_middle: f64) -> Network {
    let bus = |id, is_slack, demand_p| Bus {
        id,
        is_slack,
        demand_p,
        demand_q: 0.0,
        v_min: 0.9,
        v_max: 1.1,
    };
    Network {
        buses: vec![bus(0, true, 0.0), bus(1, false, 0.0), bus(2, false, 100.0)],
        generators: vec![
            Generator { bus: 0, p_min: 0.0, p_max: 150.0, c0: 0.0, c1: 1.0, c2: 0.0 },
            Generator { bus: 1, p_min: 0.0, p_max: 150.0, c0: 0.0, c1: 10.0, c2: 0.0 },
        ],
        lines: vec![
            Line { from: 0, to: 2, reactance: 0.1, capacity: cap_direct },
            Line { from: 1, to: 2, reactance: 0.1, capacity: cap_to_load },
            Line { from: 0, to: 1, reactance: 0.1, capacity: cap_middle },
        ],
    }
}

#[test]
fn unconstrained_network_costs_the_same_as_base_case() {
    let net = triangle(1000.0, 1000.0, 1000.0);
    let base = dc_opf(&net);
    let co = sc_opf_preventive(&net);
    assert_eq!(co.status, SolverStatus::Optimal);
    assert!(
        (co.total_cost - base.total_cost).abs() < 1e-6,
        "{} vs {}",
        co.total_cost,
        base.total_cost
    );
}

#[test]
fn outage_that_cannot_be_survived_is_reported_infeasible() {
    // With line 0 out, all 100 MW must reach the load over line 1, so a 60 MW
    // limit makes N-1 security impossible under any dispatch.
    let net = triangle(1000.0, 60.0, 1000.0);
    let co = sc_opf_preventive(&net);
    assert_eq!(co.status, SolverStatus::Infeasible);
}

#[test]
fn secure_dispatch_respects_every_scenario_and_is_no_cheaper_than_base() {
    // Each load line alone must carry all 100 MW when the other is out, so the
    // limits are 100 MW. The middle line at 50 MW forces the cheap generator to
    // back off in the intact network, costing 550 against a base cost of 100.
    let net = triangle(100.0, 100.0, 50.0);
    let base = dc_opf(&net);
    let co = sc_opf_preventive(&net);
    assert_eq!(co.status, SolverStatus::Optimal);
    assert!((base.total_cost - 100.0).abs() < 1e-6, "base {}", base.total_cost);
    assert!((co.total_cost - 550.0).abs() < 1e-6, "co-optimised {}", co.total_cost);

    let total: f64 = co.dispatch.iter().sum();
    assert!((total - 100.0).abs() < 1e-6, "dispatch sums to {total}");

    for (j, line) in net.lines.iter().enumerate() {
        assert!(co.base_flows[j].abs() <= line.capacity + 1e-6, "base line {j}");
    }
    for (k, flows) in co.contingency_flows.iter().enumerate() {
        assert!(flows[k].abs() < 1e-9, "outaged line {k} must carry no flow");
        for (j, line) in net.lines.iter().enumerate() {
            assert!(flows[j].abs() <= line.capacity + 1e-6, "outage {k}, line {j}");
        }
    }
}
