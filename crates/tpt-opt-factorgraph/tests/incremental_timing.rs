//! Timing report: incremental versus batch re-solve on a growing pose chain.
//!
//! Each step appends one pose and one odometry factor. The incremental graph is
//! updated with [`solve_incremental`] and the batch graph with [`solve`]. The
//! test asserts only that both reach the same cost and estimates. Timings are
//! printed, never asserted, and the test is ignored by default so that ordinary
//! runs stay timing-free. Run it manually with:
//!
//! ```text
//! cargo test -p tpt-opt-factorgraph --test incremental_timing -- --ignored --nocapture
//! ```

use std::time::Instant;

use tpt_opt_factorgraph::factors::{BetweenFactor, PriorFactor};
use tpt_opt_factorgraph::graph::FactorGraph;
use tpt_opt_factorgraph::manifold::Manifold;
use tpt_opt_factorgraph::noise::NoiseModel;
use tpt_opt_factorgraph::solver::{solve, solve_incremental, FactorSolverConfig};

/// Chain of `steps` poses along x with noisy-free odometry, grown one pose per step.
fn grow(g: &mut FactorGraph, k: usize) {
    g.add_variable(k, Manifold::Se2, vec![k as f64, 0.0, 0.0]).unwrap();
    g.add_factor(
        Box::new(BetweenFactor::new(k - 1, k, Manifold::Se2, vec![1.0, 0.0, 0.0])),
        NoiseModel::Isotropic { sigma: 0.1 },
        None,
    )
    .unwrap();
}

fn anchored_chain() -> FactorGraph {
    let mut g = FactorGraph::new();
    g.add_variable(0, Manifold::Se2, vec![0.0, 0.0, 0.0]).unwrap();
    g.add_factor(
        Box::new(PriorFactor::new(0, Manifold::Se2, vec![0.0, 0.0, 0.0])),
        NoiseModel::Isotropic { sigma: 0.01 },
        None,
    )
    .unwrap();
    g
}

#[test]
#[ignore = "timing report; run manually with --ignored --nocapture"]
fn incremental_vs_batch_growing_chain() {
    const STEPS: usize = 200;
    let cfg = FactorSolverConfig::new();

    let mut incremental = anchored_chain();
    let mut batch = anchored_chain();
    solve(&mut incremental, &cfg).unwrap();

    let (mut t_inc, mut t_batch) = (0.0f64, 0.0f64);
    let mut last = None;
    for k in 1..STEPS {
        grow(&mut incremental, k);
        grow(&mut batch, k);

        let start = Instant::now();
        let inc = solve_incremental(&mut incremental, &cfg, &[k]).unwrap();
        t_inc += start.elapsed().as_secs_f64();

        let start = Instant::now();
        let out = solve(&mut batch, &cfg).unwrap();
        t_batch += start.elapsed().as_secs_f64();

        assert!(inc.status.has_solution() && out.status.has_solution(), "step {k}");
        last = Some((inc.cost, out.cost));
    }

    let (cost_inc, cost_batch) = last.unwrap();
    println!(
        "incremental: {:.3} s total, batch: {:.3} s total over {} steps (final costs {:.6} / {:.6})",
        t_inc,
        t_batch,
        STEPS - 1,
        cost_inc,
        cost_batch
    );
    assert!((cost_inc - cost_batch).abs() < 1e-6, "incremental {cost_inc} vs batch {cost_batch}");
    for k in 0..STEPS {
        let a = incremental.values.get(k).unwrap();
        let b = batch.values.get(k).unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6, "pose {k}: {a:?} vs {b:?}");
        }
    }
}
