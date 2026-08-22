//! End-to-end tests for `tpt-opt-robust` against hand-computed optima.

use std::vec::Vec;

use tpt_math_prob::Rng;
use tpt_opt_core::solver::Solver;
use tpt_opt_core::SolverStatus;
use tpt_opt_milp::MilpSolver;
use tpt_opt_robust::chance::{gaussian_chance_row, normal_quantile, scenario_chance_model};
use tpt_opt_robust::dro::{
    worst_case_box_expectation, worst_case_linear_wasserstein, DroCuttingPlane,
};
use tpt_opt_robust::robust::{budgeted_reformulation, ellipsoid_reformulation, UncertainRow};
use tpt_opt_robust::saa::{SaaConfig, SaaSolver};
use tpt_opt_robust::scenario::{
    multi_stage_model, RowSense, Scenario, ScenarioPath, StageData, StageRow, TwoStageProblem,
};
use tpt_opt_robust::value::value_metrics;

/// News-vendor-style two-stage problem with known optimum:
/// min 2x + 8·E[(d − x)⁺] over demands {2, 4} w.p. 1/2 each → RP* = 8 at
/// x* = 4; E[WS] = 6; EEV = 10 (expected-demand solution x = 3).
fn news_vendor() -> TwoStageProblem {
    // Recourse row: x + y ≥ d encoded as −x − y ≤ −d.
    let mk = |d: f64| StageData {
        cost: vec![8.0],
        rows: vec![StageRow { w: vec![-1.0], t: vec![-1.0], h: -d, sense: RowSense::Le }],
    };
    TwoStageProblem {
        first_cost: vec![2.0],
        first_bounds: vec![(0.0, 10.0)],
        second_bounds: vec![(0.0, 10.0)],
        scenarios: vec![
            (Scenario { probability: 0.5, data: vec![2.0] }, mk(2.0)),
            (Scenario { probability: 0.5, data: vec![4.0] }, mk(4.0)),
        ],
    }
}

#[test]
fn two_stage_extensive_form_matches_analytic_optimum() {
    let sol = news_vendor().solve().unwrap();
    assert!((sol.objective - 8.0).abs() < 1e-6, "RP* = 8, got {}", sol.objective);
    assert!((sol.x[0] - 4.0).abs() < 1e-6, "x* = 4, got {}", sol.x[0]);
}

#[test]
fn vss_and_evpi_are_positive_and_correct() {
    let m = value_metrics(&news_vendor()).unwrap();
    assert!((m.rp - 8.0).abs() < 1e-6);
    assert!((m.ws - 6.0).abs() < 1e-6, "E[WS] = 6, got {}", m.ws);
    assert!((m.eev - 10.0).abs() < 1e-6, "EEV = 10, got {}", m.eev);
    assert!((m.vss - 2.0).abs() < 1e-6);
    assert!((m.evpi - 2.0).abs() < 1e-6);
}

/// The same news-vendor problem expressed as a 2-stage scenario *tree*
/// through [`multi_stage_model`] must reproduce the extensive-form answer
/// (and merge the shared root decision into one variable).
#[test]
fn multi_stage_tree_matches_two_stage() {
    let paths = vec![
        ScenarioPath {
            probability: 0.5,
            data: vec![vec![2.0], vec![2.0]], // stage-1 marker, stage-2 marker
        },
        ScenarioPath { probability: 0.5, data: vec![vec![2.0], vec![4.0]] },
    ];
    // Stage 1: one variable (the here-and-now order), stage 2: one variable
    // per leaf. Linking: y_t ≥ d_t − x_{t−1} chain collapsed to y₂ ≥ d₂ − x₁
    // plus y₁ free; we encode the recourse directly at the leaf:
    // row over concatenated block [x₁, y₂]: −x₁ − y₂ ≥ −d₂.
    let model = multi_stage_model(
        &paths,
        2,
        |_t, _data| vec![(0.0f64, 10.0f64)], // one var per node
        |t, _data| if t == 1 { vec![2.0] } else { vec![8.0] },
        |_t, prefix| {
            let d = prefix.last().unwrap()[0];
            vec![(vec![-1.0, -1.0], -d, RowSense::Le)]
        },
    )
    .unwrap();
    // Prefix merging: 1 shared stage-1 node + 2 leaves = 3 variables.
    assert_eq!(model.variables.len(), 3);
    let mut solver = MilpSolver::new();
    let sol = solver.solve(&model).unwrap();
    assert_eq!(sol.status, SolverStatus::Optimal);
    assert!(
        (sol.objective_value - 8.0).abs() < 1e-6,
        "tree optimum = 8, got {}",
        sol.objective_value
    );
    assert!((sol.primal[0] - 4.0).abs() < 1e-6);
}

/// SAA on a continuous-demand news vendor: d ~ U[2,4],
/// f(x) = 2x + 8·E[(d−x)⁺] = 2x + 2(4−x)² → x* = 3.5, f* = 7.5.
#[test]
fn saa_recovers_true_optimum_within_confidence() {
    let draw = |rng: &mut tpt_math_prob::Xoshiro256| vec![2.0 + 2.0 * rng.next_f64()];
    let evaluate = |x: &[f64], d: &Vec<f64>| 2.0 * x[0] + 8.0 * (d[0] - x[0]).max(0.0);
    let solve_sampled = |sample: &[Vec<f64>]| -> Result<(Vec<f64>, f64), String> {
        // Deterministic 1-D grid minimisation of the sample average.
        let mut best = (0.0f64, f64::INFINITY);
        let mut i = 0.0f64;
        while i <= 400.0 {
            let x = i * 10.0 / 400.0;
            let avg = sample.iter().map(|d| evaluate(&[x], d)).sum::<f64>() / sample.len() as f64;
            if avg < best.1 {
                best = (x, avg);
            }
            i += 1.0;
        }
        Ok((vec![best.0], best.1))
    };
    let config = SaaConfig {
        samples_per_replication: 150,
        replications: 20,
        validation_samples: 4000,
        confidence: 0.95,
        seed: 42,
    };
    let result = SaaSolver::new(config, solve_sampled, evaluate, draw).run().unwrap();
    assert!(
        result.lower_bound <= 7.5 + result.lower_bound_half_width + 1e-9,
        "LB must not exceed the true optimum beyond noise"
    );
    assert!(
        (result.upper_bound - 7.5).abs() < 0.35,
        "UB {} should estimate f* = 7.5",
        result.upper_bound
    );
    assert!(result.gap < 0.5, "gap {} too large", result.gap);
    assert!((result.x_best[0] - 3.5).abs() < 0.35, "x̂ = {}", result.x_best[0]);
}

#[test]
fn normal_quantile_matches_reference_values() {
    assert!((normal_quantile(0.975) - 1.959964).abs() < 1e-5);
    assert!((normal_quantile(0.5)).abs() < 1e-9);
    assert!((normal_quantile(0.9) - 1.281552).abs() < 1e-5);
}

/// Scenario chance constraint: x ≤ B where B ~ Uniform{1..10}, ε = 0.2 with
/// 10 samples → at most 2 violations allowed → x ≤ 3rd-smallest sample = 3.
#[test]
fn scenario_chance_enforces_var_budget() {
    let mut samples: Vec<(Vec<f64>, f64)> = Vec::new();
    for b in 1..=10 {
        samples.push((vec![1.0], b as f64));
    }
    // Maximise x (min −x): the VaR budget caps how small a bound x may face.
    let x = scenario_chance_model(vec![-1.0], vec![(0.0, 10.0)], vec![], samples, 0.2).unwrap();
    assert!((x[0] - 3.0).abs() < 1e-6, "x must equal 3, got {}", x[0]);
}

/// Gaussian deterministic equivalent: P(a·x ≤ 1) ≥ 0.95 with a ~ N(1, 0.2²)
/// gives the protected row (1 + 1.96·0.2)·x ≤ 1 → x ≈ 0.7184.
#[test]
fn gaussian_chance_row_protects_at_the_right_level() {
    let row = gaussian_chance_row(vec![1.0], vec![0.2], 1.0, 0.05);
    let mut model = tpt_opt_core::model::Model::new(1);
    model.variables[0].bound = tpt_opt_core::VarBound::continuous(0.0, 10.0);
    model.set_objective(tpt_opt_core::model::Objective {
        sense: tpt_opt_core::model::Sense::Minimize,
        indices: vec![0],
        coeffs: vec![-1.0],
        constant: 0.0,
    });
    model.add_constraint(tpt_opt_core::model::Constraint::le(
        vec![0],
        vec![row.mu[0] + row.protection[0]],
        row.rhs,
    ));
    let mut solver = MilpSolver::new();
    let sol = solver.solve(&model).unwrap();
    let expected = 1.0 / (1.0 + normal_quantile(0.95) * 0.2);
    assert!((sol.primal[0] - expected).abs() < 1e-6);
}

/// Bertsimas–Sim budgeted protection on `min −x s.t. (1±0.2)x ≤ 1`:
/// Γ=0 → x=1; Γ=1 → protection 0.2·max(x₁,x₂), optimised by the symmetric
/// split x₁=x₂ → x=1/1.1; Γ=2 (both deviate) → 1.2(x₁+x₂) ≤ 1 → x=1/1.2.
#[test]
fn budgeted_reformulation_interpolates_protection() {
    let row = UncertainRow { nominal: vec![1.0, 1.0], deviation: vec![0.2, 0.2], rhs: 1.0 };
    let cases = [(0.0, 1.0), (1.0, 1.0 / 1.1), (2.0, 1.0 / 1.2)];
    for &(gamma, expected) in &cases {
        let model = budgeted_reformulation(
            vec![-1.0, -1.0],
            vec![(0.0, 10.0), (0.0, 10.0)],
            vec![row.clone()],
            vec![gamma],
        );
        let mut solver = MilpSolver::new();
        let sol = solver.solve(&model).unwrap();
        assert_eq!(sol.status, SolverStatus::Optimal);
        let x = sol.primal[0] + sol.primal[1];
        assert!((x - expected).abs() < 1e-6, "Γ={gamma}: x={x}, expected {expected}");
    }
}

/// Ellipsoidal protection with diagonal Σ (σ = 0.2) at κ = 1.96:
/// (1 + κσ)x ≤ 1 → x = 1/(1 + 1.96·0.2).
#[test]
fn ellipsoid_reformulation_scales_with_kappa_sigma() {
    let row = UncertainRow { nominal: vec![1.0], deviation: vec![0.0], rhs: 1.0 };
    let model = ellipsoid_reformulation(
        vec![-1.0],
        vec![(0.0, 10.0)],
        vec![row],
        vec![vec![0.2]],
        normal_quantile(0.95),
    );
    let mut solver = MilpSolver::new();
    let sol = solver.solve(&model).unwrap();
    let expected = 1.0 / (1.0 + normal_quantile(0.95) * 0.2);
    assert!((sol.primal[0] - expected).abs() < 1e-6);
}

#[test]
fn box_worst_case_puts_mass_on_expensive_outcomes() {
    let wc = worst_case_box_expectation(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0], &[0.5, 0.5, 0.5]);
    assert!((wc - 2.5).abs() < 1e-9, "0.5·3 + 0.5·2 = 2.5, got {wc}");
}

/// DRO cutting plane on `min_x max_p E_p[a·x]` with a = ∓1 and box
/// p ∈ [0.25, 0.75]: worst case puts p₊ = 0.75 → value 0.5x → x* = 0.
#[test]
fn dro_cutting_plane_finds_robust_decision() {
    let solver =
        DroCuttingPlane::new(vec![(0.0, 5.0)], vec![0.25, 0.25], vec![0.75, 0.75], |s, x| {
            if s == 0 {
                -x[0]
            } else {
                x[0]
            }
        })
        .with_max_iterations(50);
    let (x, wc) = solver.solve().unwrap();
    assert!(x[0] < 1e-6, "robust decision is x = 0, got {}", x[0]);
    assert!(wc.abs() < 1e-6, "worst-case value 0, got {wc}");
}

#[test]
fn wasserstein_linear_worst_case_adds_lipschitz_margin() {
    let wc = worst_case_linear_wasserstein(&[vec![0.0], vec![2.0]], &[1.0], 0.0, 1.0);
    assert!((wc - 2.0).abs() < 1e-9, "mean 1 + θ·‖g‖ = 2, got {wc}");
}
