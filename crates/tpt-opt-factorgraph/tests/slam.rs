//! Integration tests: 2-D/3-D pose graphs, small bundle adjustment,
//! Schur-vs-batch agreement, marginal covariance against an analytic
//! Gaussian posterior, incremental-vs-batch agreement, and robust kernels
//! on a corrupted loop closure.
//!
//! Measurement conventions exercised here: `BetweenFactor` measurements are
//! **group elements** in storage layout (`(dx, dy, dθ)` for SE(2),
//! `(t, q)` for SE(3)), and initial estimates come from dead-reckoning the
//! odometry chain — the standard SLAM setup.

use tpt_opt_factorgraph::factors::{BetweenFactor, PriorFactor, ReprojectionFactor};
use tpt_opt_factorgraph::graph::FactorGraph;
use tpt_opt_factorgraph::manifold::Manifold;
use tpt_opt_factorgraph::marginal::marginal_covariance;
use tpt_opt_factorgraph::noise::NoiseModel;
use tpt_opt_factorgraph::solver::{solve, solve_incremental, solve_schur, FactorSolverConfig};
use tpt_opt_nls::Loss;

/// Deterministic LCG uniform noise in [−1, 1).
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    }
}

/// 2-D Manhattan-world pose graph: a grid walk with noisy odometry edges
/// (relative group elements) plus loop closures back to the origin pose,
/// initialised by dead-reckoning.
fn build_2d_pose_graph(
    side: usize,
    sigma_odom: f64,
    with_outlier: bool,
    robust: bool,
) -> (FactorGraph, Vec<[f64; 3]>) {
    let mut rng = Lcg(0x5EED);
    let mut truth: Vec<[f64; 3]> = Vec::new();
    let mut p = [0.0f64, 0.0, 0.0];
    for _i in 0..(side * side) {
        truth.push(p);
        let forward = [p[0] + 1.0, p[1], 0.0];
        let turn = [p[0], p[1] + 1.0, std::f64::consts::FRAC_PI_2];
        p = if (truth.len()) % side == 0 { turn } else { forward };
    }
    let se2 = Manifold::Se2;
    let mut g = FactorGraph::new();
    let mut est = vec![0.0f64, 0.0, 0.0];
    for (i, _t) in truth.iter().enumerate() {
        if i > 0 {
            let rel = se2.compose(&se2.inverse(&truth[i - 1]), &truth[i]);
            est = se2.compose(
                &est,
                &[
                    rel[0] + sigma_odom * rng.next_f64(),
                    rel[1] + sigma_odom * rng.next_f64(),
                    rel[2] + sigma_odom * rng.next_f64(),
                ],
            );
        }
        g.add_variable(i, Manifold::Se2, est.clone()).unwrap();
    }
    g.add_factor(
        Box::new(PriorFactor::new(0, Manifold::Se2, vec![0.0, 0.0, 0.0])),
        NoiseModel::Isotropic { sigma: 0.01 },
        None,
    )
    .unwrap();
    for i in 0..truth.len() - 1 {
        let rel = se2.compose(&se2.inverse(&truth[i]), &truth[i + 1]);
        let meas = vec![
            rel[0] + sigma_odom * rng.next_f64(),
            rel[1] + sigma_odom * rng.next_f64(),
            rel[2] + sigma_odom * rng.next_f64(),
        ];
        g.add_factor(
            Box::new(BetweenFactor::new(i, i + 1, Manifold::Se2, meas)),
            NoiseModel::Isotropic { sigma: sigma_odom },
            None,
        )
        .unwrap();
    }
    for i in (4..truth.len()).step_by(7) {
        let rel = se2.compose(&se2.inverse(&truth[0]), &truth[i]);
        // Corrupt exactly one closure (i = 11) so the robust kernel has
        // clean siblings to fall back on.
        let blow = if with_outlier && i == 11 { 40.0 } else { 1.0 };
        let meas = vec![
            rel[0] + sigma_odom * blow * rng.next_f64(),
            rel[1] + sigma_odom * blow * rng.next_f64(),
            rel[2] + sigma_odom * blow * rng.next_f64(),
        ];
        g.add_factor(
            Box::new(BetweenFactor::new(0, i, Manifold::Se2, meas)),
            NoiseModel::Isotropic { sigma: sigma_odom },
            // δ is in whitened-residual units: 3σ → 3.
            if robust { Some(Loss::Huber { delta: 3.0 }) } else { None },
        )
        .unwrap();
    }
    (g, truth)
}

#[test]
fn pose_graph_2d_converges_to_truth() {
    let side = 5;
    let (mut graph, truth) = build_2d_pose_graph(side, 0.02, false, false);
    let outcome = solve(&mut graph, &FactorSolverConfig::new()).unwrap();
    assert!(outcome.status.has_solution(), "{:?}", outcome.status);
    // 84 whitened unit-variance residuals ⇒ optimum cost ≈ 42.
    assert!(outcome.cost < 60.0, "cost {}", outcome.cost);
    let mut rms = 0.0;
    for (i, t) in truth.iter().enumerate() {
        let v = graph.values.get(i).unwrap();
        rms += (v[0] - t[0]).powi(2) + (v[1] - t[1]).powi(2);
    }
    let rms = (rms / truth.len() as f64).sqrt();
    assert!(rms < 0.1, "position RMS {rms}");
}

#[test]
fn pose_graph_2d_huber_survives_outlier() {
    let side = 5;
    // Robust solve stays near truth despite a grossly corrupted closure.
    // Solve L2 first, then enable the kernel and re-solve (warm IRLS).
    let (mut graph, truth) = build_2d_pose_graph(side, 0.02, true, false);
    solve(&mut graph, &FactorSolverConfig::new()).unwrap();
    for f in 0..graph.n_factors() {
        graph.set_factor_robust(f, Some(Loss::Huber { delta: 3.0 })).unwrap();
    }
    let outcome = solve(&mut graph, &FactorSolverConfig::new()).unwrap();
    assert!(outcome.status.has_solution());
    let mut rms = 0.0;
    for (i, t) in truth.iter().enumerate() {
        let v = graph.values.get(i).unwrap();
        rms += (v[0] - t[0]).powi(2) + (v[1] - t[1]).powi(2);
    }
    let rms_robust = (rms / truth.len() as f64).sqrt();
    // Reference: plain L2 on the same corrupted data gets dragged.
    let (mut graph_l2, truth_l2) = build_2d_pose_graph(side, 0.02, true, false);
    let _ = solve(&mut graph_l2, &FactorSolverConfig::new());
    let mut rms_l2 = 0.0;
    for (i, t) in truth_l2.iter().enumerate() {
        let v = graph_l2.values.get(i).unwrap();
        rms_l2 += (v[0] - t[0]).powi(2) + (v[1] - t[1]).powi(2);
    }
    let rms_l2 = (rms_l2 / truth_l2.len() as f64).sqrt();
    assert!(rms_robust < rms_l2, "robust rms {rms_robust} should beat l2 rms {rms_l2}");
    assert!(rms_robust < 0.25, "robust rms {rms_robust}");
}

#[test]
fn pose_graph_2d_two_stage_robust() {
    // The standard robust workflow: batch-solve L2 (the IRLS weights are
    // meaningless from a drifted cold start), then enable Huber kernels and
    // re-solve from the warm start.
    let side = 5;
    let (mut graph, truth) = build_2d_pose_graph(side, 0.02, true, false);
    solve(&mut graph, &FactorSolverConfig::new()).unwrap();
    for f in 0..graph.n_factors() {
        graph.set_factor_robust(f, Some(Loss::Huber { delta: 3.0 })).unwrap();
    }
    let outcome = solve(&mut graph, &FactorSolverConfig::new()).unwrap();
    assert!(outcome.status.has_solution(), "{:?}", outcome.status);
    let mut rms = 0.0;
    for (i, t) in truth.iter().enumerate() {
        let v = graph.values.get(i).unwrap();
        rms += (v[0] - t[0]).powi(2) + (v[1] - t[1]).powi(2);
    }
    let rms = (rms / truth.len() as f64).sqrt();
    assert!(rms < 0.25, "two-stage robust rms {rms}");
}

#[test]
fn pose_graph_3d_loop_closure() {
    // SE(3) chain of 8 poses along +x with identity-rotation odometry and a
    // loop closure between last and first.
    let n = 8;
    let mut g = FactorGraph::new();
    let mut rng = Lcg(99);
    g.add_variable(0, Manifold::Se3, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
    for i in 1..n {
        let est = vec![
            i as f64 + 0.3 * rng.next_f64(),
            0.4 * rng.next_f64(),
            0.3 * rng.next_f64(),
            0.0,
            0.0,
            0.0,
            1.0,
        ];
        g.add_variable(i, Manifold::Se3, est).unwrap();
        // Between measurements use storage layout: (t, q) — 7 values.
        let meas = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        g.add_factor(
            Box::new(BetweenFactor::new(i - 1, i, Manifold::Se3, meas)),
            NoiseModel::Isotropic { sigma: 0.05 },
            None,
        )
        .unwrap();
    }
    // Identity SE(3) prior — note the quaternion component must be a valid
    // rotation (0,0,0,1), not all-zero.
    g.add_factor(
        Box::new(PriorFactor::new(0, Manifold::Se3, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])),
        NoiseModel::Isotropic { sigma: 0.01 },
        None,
    )
    .unwrap();
    g.add_factor(
        Box::new(BetweenFactor::new(
            0,
            n - 1,
            Manifold::Se3,
            vec![(n - 1) as f64, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        )),
        NoiseModel::Isotropic { sigma: 0.05 },
        None,
    )
    .unwrap();
    let outcome = solve(&mut g, &FactorSolverConfig::new()).unwrap();
    assert!(outcome.status.has_solution(), "{:?}", outcome.status);
    for i in 0..n {
        let v = g.values.get(i).unwrap();
        assert!((v[0] - i as f64).abs() < 0.05, "pose {i} x = {}", v[0]);
        assert!(v[1].abs() < 0.05 && v[2].abs() < 0.05, "pose {i} = {v:?}");
    }
}

/// The small BA fixture (two anchored SE(3) cameras, three points) shared
/// by the recovery and Schur-agreement tests.
fn build_ba() -> FactorGraph {
    let f = 100.0;
    let points_truth = [[0.1, 0.0, 3.0], [-0.2, 0.15, 2.8], [0.0, -0.1, 3.2]];
    let cams: Vec<[f64; 7]> =
        vec![[-0.5, 0.0, -2.0, 0.0, 0.0, 0.0, 1.0], [0.5, 0.0, -2.0, 0.0, 0.0, 0.0, 1.0]];
    let project = |t: &[f64; 7], p: &[f64; 3]| -> [f64; 2] {
        let x = p[0] + t[0];
        let y = p[1] + t[1];
        let z = (p[2] + t[2]).max(1e-9);
        [f * x / z, f * y / z]
    };
    let mut g = FactorGraph::new();
    for (ci, c) in cams.iter().enumerate() {
        g.add_variable(ci, Manifold::Se3, c.to_vec()).unwrap();
    }
    let mut rng = Lcg(7);
    for (pi, p) in points_truth.iter().enumerate() {
        let noisy = vec![
            p[0] + 0.05 * rng.next_f64(),
            p[1] + 0.05 * rng.next_f64(),
            p[2] + 0.05 * rng.next_f64(),
        ];
        g.add_variable(10 + pi, Manifold::Euclidean(3), noisy).unwrap();
        for (ci, c) in cams.iter().enumerate() {
            let uv = project(c, p);
            g.add_factor(
                Box::new(ReprojectionFactor::new(ci, 10 + pi, uv, [f, f], [0.0, 0.0])),
                NoiseModel::Isotropic { sigma: 0.5 },
                None,
            )
            .unwrap();
        }
    }
    for (ci, c) in cams.iter().enumerate() {
        g.add_factor(
            Box::new(PriorFactor::new(ci, Manifold::Se3, c.to_vec())),
            NoiseModel::Isotropic { sigma: 1e-3 },
            None,
        )
        .unwrap();
    }
    g
}

#[test]
fn bundle_adjustment_recovers_points() {
    let mut g = build_ba();
    let outcome = solve(&mut g, &FactorSolverConfig::new()).unwrap();
    assert!(outcome.status.has_solution(), "{:?}", outcome.status);
    assert!(outcome.cost < 1e-10, "cost {}", outcome.cost);
    for (pi, p) in [[0.1, 0.0, 3.0], [-0.2, 0.15, 2.8], [0.0, -0.1, 3.2]].iter().enumerate() {
        let v = g.values.get(10 + pi).unwrap();
        let err = ((v[0] - p[0]).powi(2) + (v[1] - p[1]).powi(2) + (v[2] - p[2]).powi(2)).sqrt();
        assert!(err < 1e-4, "point {pi}: {v:?} vs {p:?} (err {err})");
    }
}

#[test]
fn schur_solve_matches_batch_solve() {
    let mut batch = build_ba();
    let out_batch = solve(&mut batch, &FactorSolverConfig::new()).unwrap();
    let mut schur = build_ba();
    // Keys 10..13 are the points.
    let out_schur = solve_schur(&mut schur, &FactorSolverConfig::new(), &[10, 11, 12]).unwrap();
    assert!(out_batch.status.has_solution() && out_schur.status.has_solution());
    assert!(
        (out_batch.cost - out_schur.cost).abs() < 1e-9,
        "batch {} vs schur {}",
        out_batch.cost,
        out_schur.cost
    );
    for k in 10..13 {
        let a = batch.values.get(k).unwrap();
        let b = schur.values.get(k).unwrap();
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 1e-4, "key {k}[{i}]: {a:?} vs {b:?}");
        }
    }
}

#[test]
fn marginal_covariance_matches_analytic_linear_case() {
    // x0 with a tight prior, x1 connected by a noisy between edge. The
    // posterior information matrix is
    // I = [[1/sp²+1/sm², −1/sm²], [−1/sm², 1/sm²]], so Var(x1) = I₀₀/det(I).
    let sp = 0.1;
    let sm = 0.5;
    let mut g = FactorGraph::new();
    g.add_variable(0, Manifold::Euclidean(1), vec![0.0]).unwrap();
    g.add_variable(1, Manifold::Euclidean(1), vec![0.3]).unwrap();
    g.add_factor(
        Box::new(PriorFactor::new(0, Manifold::Euclidean(1), vec![0.0])),
        NoiseModel::Isotropic { sigma: sp },
        None,
    )
    .unwrap();
    g.add_factor(
        Box::new(BetweenFactor::new(0, 1, Manifold::Euclidean(1), vec![1.0])),
        NoiseModel::Isotropic { sigma: sm },
        None,
    )
    .unwrap();
    let cov1 = marginal_covariance(&g, 1, 1e-7).unwrap();
    let i00 = 1.0 / (sp * sp) + 1.0 / (sm * sm);
    let i11 = 1.0 / (sm * sm);
    let i01 = -1.0 / (sm * sm);
    let det = i00 * i11 - i01 * i01;
    assert!((cov1[0] - i00 / det).abs() < 1e-8, "var {} vs {}", cov1[0], i00 / det);
    let cov0 = marginal_covariance(&g, 0, 1e-7).unwrap();
    assert!((cov0[0] - i11 / det).abs() < 1e-8, "var0 {} vs {}", cov0[0], i11 / det);
}

#[test]
fn incremental_solve_matches_batch_solve() {
    // Batch-solve a chain; then add one more edge and compare an
    // incremental solve (only new-adjacent factors re-linearised) with a
    // fresh batch solve.
    let build = |extra: bool| {
        let mut g = FactorGraph::new();
        g.add_variable(0, Manifold::Se2, vec![0.0, 0.0, 0.0]).unwrap();
        g.add_variable(1, Manifold::Se2, vec![0.9, 0.2, 0.1]).unwrap();
        g.add_variable(2, Manifold::Se2, vec![1.8, 0.4, -0.1]).unwrap();
        g.add_factor(
            Box::new(PriorFactor::new(0, Manifold::Se2, vec![0.0, 0.0, 0.0])),
            NoiseModel::Isotropic { sigma: 0.05 },
            None,
        )
        .unwrap();
        g.add_factor(
            Box::new(BetweenFactor::new(0, 1, Manifold::Se2, vec![1.0, 0.0, 0.0])),
            NoiseModel::Isotropic { sigma: 0.1 },
            None,
        )
        .unwrap();
        g.add_factor(
            Box::new(BetweenFactor::new(1, 2, Manifold::Se2, vec![1.0, 0.0, 0.0])),
            NoiseModel::Isotropic { sigma: 0.1 },
            None,
        )
        .unwrap();
        if extra {
            g.add_factor(
                Box::new(BetweenFactor::new(0, 2, Manifold::Se2, vec![2.0, 0.0, 0.0])),
                NoiseModel::Isotropic { sigma: 0.1 },
                None,
            )
            .unwrap();
        }
        g
    };
    let cfg = FactorSolverConfig::new();
    let mut incremental = build(false);
    solve(&mut incremental, &cfg).unwrap();
    incremental
        .add_factor(
            Box::new(BetweenFactor::new(0, 2, Manifold::Se2, vec![2.0, 0.0, 0.0])),
            NoiseModel::Isotropic { sigma: 0.1 },
            None,
        )
        .unwrap();
    let out_inc = solve_incremental(&mut incremental, &cfg, &[2]).unwrap();
    let mut batch = build(true);
    let out_batch = solve(&mut batch, &cfg).unwrap();
    assert!(out_inc.status.has_solution() && out_batch.status.has_solution());
    assert!(
        (out_inc.cost - out_batch.cost).abs() < 1e-8,
        "incremental {} vs batch {}",
        out_inc.cost,
        out_batch.cost
    );
    for k in 0..3 {
        let a = incremental.values.get(k).unwrap();
        let b = batch.values.get(k).unwrap();
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 1e-5, "key {k}[{i}]: {a:?} vs {b:?}");
        }
    }
}
