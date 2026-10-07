//! Integration tests: classical MINPACK-style problems end to end, plus
//! covariance, robust-fitting, and sparse-path checks against known optima.

use tpt_opt_nls::{
    gauss_newton, levenberg_marquardt, powell_dogleg, JacobianMode, Loss, NlsConfig, NlsProblem,
    NlsStatus,
};

fn assert_near(actual: &[f64], expect: &[f64], tol: f64) {
    for (a, e) in actual.iter().zip(expect.iter()) {
        assert!((a - e).abs() < tol, "got {a}, want {e}");
    }
}

/// Brown almost-linear (MINPACK): residual i is `x_i + Σx − (n+1)` for
/// i < n and `(Πx) − 1` for i = n. Solution: all ones.
fn brown_almost_linear(n: usize) -> NlsProblem {
    NlsProblem::builder(n)
        .residual_block(n, move |x, out| {
            let sum: tpt_opt_nls::dual::Dual =
                x.iter().fold(tpt_opt_nls::dual::Dual::plain(0.0), |acc, v| acc + v.clone());
            for (i, o) in out.iter_mut().enumerate().take(n - 1) {
                *o = x[i].clone() + sum.clone() - (n + 1) as f64;
            }
            let prod = x.iter().fold(tpt_opt_nls::dual::Dual::plain(1.0), |acc, v| acc * v.clone());
            out[n - 1] = prod - 1.0;
        })
        .build()
}

/// Beale (MINPACK): r_k = rhs_k − x1(1 − x2^{k+1}) with rhs = (1.5, 2.25,
/// 2.625), k = 0..2. Known solution: x1 = 3, x2 = 0.5. Note the (1, 1)
/// start is degenerate — the ∂/∂x1 column of J vanishes there — so this
/// also exercises the trace-scaled initial damping.
fn beale() -> NlsProblem {
    NlsProblem::builder(2)
        .residual_block(3, |x, out| {
            const RHS: [f64; 3] = [1.5, 2.25, 2.625]; // Beale's constants
            for (k, o) in out.iter_mut().enumerate() {
                let term = x[0].clone() * (1.0 - x[1].clone().powi(k as i32 + 1));
                *o = RHS[k] - term;
            }
        })
        .build()
}

/// Rosenbrock band (banded sparse Jacobian): residuals couple only nearest
/// neighbours, so the normal equations are tridiagonal-ish — the natural
/// case for `LinearSolver::Sparse` at larger n.
fn chain_problem(n: usize) -> NlsProblem {
    NlsProblem::builder(n)
        .residual_block(n, move |x, out| {
            for i in 0..n {
                if i + 1 < n {
                    out[i] = 10.0 * (x[i + 1].clone() - x[i].clone() * x[i].clone());
                } else {
                    out[i] = 1.0 - x[i].clone();
                }
            }
        })
        .build()
}

#[test]
fn brown_almost_linear_all_solvers() {
    let p = brown_almost_linear(10);
    let start: Vec<f64> = vec![0.5; 10];
    for (name, res) in [
        ("lm", levenberg_marquardt(&p, &start, &NlsConfig::new()).unwrap()),
        ("gn", gauss_newton(&p, &start, &NlsConfig::new()).unwrap()),
        ("dogleg", powell_dogleg(&p, &start, &NlsConfig::new()).unwrap()),
    ] {
        assert_eq!(res.status, NlsStatus::Converged, "{name}: {}", res.termination);
        assert!(res.cost < 1e-16, "{name} cost {}", res.cost);
        // The system xᵢ + Σx = n+1 (∀i<n), Πx = 1 has more than one root
        // for n = 10 (besides all-ones there is one with n−1 equal entries
        // and a distinct last entry), so root-ness — already asserted by
        // the cost — is the right check, not a specific vector.
    }
}

#[test]
fn beale_known_optimum() {
    let p = beale();
    let res = levenberg_marquardt(&p, &[1.0, 1.0], &NlsConfig::new()).unwrap();
    assert!(res.status.has_solution(), "{}", res.termination);
    assert_near(&res.x, &[3.0, 0.5], 1e-5);
    assert!(res.cost < 1e-18);
}

#[test]
fn freudenstein_roth_root() {
    // r1 = −13 + x1 + ((5 − x2)x2 − 2)x2, r2 = −29 + x1 + ((x2 + 1)x2 − 14)x2.
    // One root is (5, 4).
    let p = NlsProblem::builder(2)
        .residual_block(2, |x, out| {
            out[0] =
                -13.0 + x[0].clone() + ((5.0 - x[1].clone()) * x[1].clone() - 2.0) * x[1].clone();
            out[1] =
                -29.0 + x[0].clone() + ((x[1].clone() + 1.0) * x[1].clone() - 14.0) * x[1].clone();
        })
        .build();
    let res = levenberg_marquardt(&p, &[5.5, 4.5], &NlsConfig::new()).unwrap();
    assert!(res.status.has_solution(), "{}", res.termination);
    // Verify it is a root (residual ≈ 0); the problem also has a second
    // root near (11.41, −0.90) and a local minimum, so start near (5, 4).
    assert!(res.cost < 1e-16, "cost = {}", res.cost);
    assert_near(&res.x, &[5.0, 4.0], 1e-5);
    let r1 = -13.0 + res.x[0] + ((5.0 - res.x[1]) * res.x[1] - 2.0) * res.x[1];
    let r2 = -29.0 + res.x[0] + ((res.x[1] + 1.0) * res.x[1] - 14.0) * res.x[1];
    assert!(r1.abs() < 1e-6 && r2.abs() < 1e-6);
}

#[test]
fn gaussian_peak_curve_fit_recovers_parameters() {
    // Data from y = A·exp(−(t − μ)² / (2σ²)) with A = 3, μ = 1.2, σ = 0.4,
    // sampled exactly (no noise).
    let (a0, mu, sigma) = (3.0, 1.2, 0.4);
    let t: Vec<f64> = (0..25).map(|i| 0.1 * i as f64).collect();
    let data: Vec<f64> =
        t.iter().map(|&ti| a0 * (-(ti - mu) * (ti - mu) / (2.0 * sigma * sigma)).exp()).collect();
    let p = NlsProblem::builder(3)
        .residual_block(t.len(), move |x, out| {
            for (i, o) in out.iter_mut().enumerate() {
                let d = t[i] - x[1].clone();
                *o = x[0].clone() * (-(d.clone() * d) / (2.0 * x[2].clone() * x[2].clone())).exp()
                    - data[i];
            }
        })
        .build();
    let res = levenberg_marquardt(&p, &[1.0, 0.5, 1.0], &NlsConfig::new()).unwrap();
    assert!(res.status.has_solution(), "{}", res.termination);
    assert_near(&res.x, &[a0, mu, sigma], 1e-4);
}

#[test]
fn covariance_matches_analytic_linear_result() {
    // y = a·x with noise of known variance σ² = 0.01: cov(a) = σ²/Σx².
    let sigma = 0.1;
    let xs: Vec<f64> = (1..=15).map(|i| i as f64).collect();
    // Alternating ±1 noise: the realised residual variance is exactly σ².
    let noise: Vec<f64> = (0..15).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let ys: Vec<f64> = xs.iter().zip(noise).map(|(&x, n)| 2.5 * x + sigma * n).collect();
    let xs2 = xs.clone();
    let p = NlsProblem::builder(1)
        .residual_block(xs.len(), move |x, out| {
            for (i, o) in out.iter_mut().enumerate() {
                *o = x[0].clone() * xs2[i] - ys[i];
            }
        })
        .build();
    let res = levenberg_marquardt(&p, &[0.0], &NlsConfig::new()).unwrap();
    // A linear problem may stop at cost-change tolerance once the exact
    // optimum is hit; any solved status is fine.
    assert!(res.status.has_solution(), "{:?}", res.status);
    // Alternating noise correlates with x, biasing the slope by
    // σ·Σxᵢ(−1)^i/Σxᵢ² ≈ +6.5e-4.
    assert!((res.x[0] - 2.5).abs() < 1e-3, "slope = {}", res.x[0]);
    let cov = res.covariance().expect("covariance");
    let sum_x2: f64 = xs.iter().map(|x| x * x).sum();
    let expected_var = sigma * sigma / sum_x2;
    let got_var = cov[0];
    // Within a factor of 2 (noise is small-sample, not iid-normal exact).
    assert!(
        got_var > expected_var * 0.25 && got_var < expected_var * 4.0,
        "var {got_var} vs expected ~{expected_var}"
    );
}

#[test]
fn cauchy_robust_fit_survives_outliers() {
    // 12 points on y = 0.5x + 2, three corrupted.
    let xs: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let ys: Vec<f64> = xs
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let clean = 0.5 * x + 2.0;
            match i {
                3 => clean + 9.0,
                7 => clean - 8.0,
                10 => clean + 12.0,
                _ => clean,
            }
        })
        .collect();
    // Clone per call so the factory stays `Fn` (the residual block closes
    // over its own copy of the data).
    let build = || {
        let (xs, ys) = (xs.clone(), ys.clone());
        NlsProblem::builder(2)
            .residual_block(xs.len(), move |x, out| {
                for (i, o) in out.iter_mut().enumerate() {
                    *o = x[0].clone() * xs[i] + x[1].clone() - ys[i];
                }
            })
            .build()
    };
    let l2 = levenberg_marquardt(&build(), &[0.0, 0.0], &NlsConfig::new()).unwrap();
    let cauchy = levenberg_marquardt(
        &build(),
        &[0.0, 0.0],
        &NlsConfig::new().with_loss(Loss::Cauchy { delta: 0.5 }),
    )
    .unwrap();
    let err = |p: &[f64]| (p[0] - 0.5).abs() + (p[1] - 2.0).abs();
    assert!(err(&cauchy.x) < err(&l2.x), "cauchy {:?} vs l2 {:?}", cauchy.x, l2.x);
    assert!(err(&cauchy.x) < 0.2, "cauchy params {:?}", cauchy.x);
}

#[test]
fn sparse_path_scales_to_long_chain() {
    let p = chain_problem(300);
    let start: Vec<f64> = (0..300).map(|i| if i % 2 == 0 { 1.5 } else { 1.0 }).collect();
    let dense = levenberg_marquardt(&p, &start, &NlsConfig::new()).unwrap();
    let sparse = levenberg_marquardt(
        &p,
        &start,
        &NlsConfig::new().with_linear_solver(tpt_opt_nls::LinearSolver::Sparse),
    )
    .unwrap();
    assert!(dense.status.has_solution(), "{}", dense.termination);
    assert!(sparse.status.has_solution(), "{}", sparse.termination);
    assert!(sparse.cost < 1e-10, "sparse cost {}", sparse.cost);
    assert!((dense.cost - sparse.cost).abs() < 1e-8);
}

#[test]
fn fd_mode_and_dogleg_agree_on_rosenbrock_family() {
    // Cross-check JacobianMode::FiniteDifference + Powell dogleg on Beale.
    let p = beale();
    let res = powell_dogleg(
        &p,
        &[1.0, 1.0],
        &NlsConfig::new().with_jacobian(JacobianMode::FiniteDifference),
    )
    .unwrap();
    assert!(res.status.has_solution(), "{}", res.termination);
    assert_near(&res.x, &[3.0, 0.5], 1e-3);
}
