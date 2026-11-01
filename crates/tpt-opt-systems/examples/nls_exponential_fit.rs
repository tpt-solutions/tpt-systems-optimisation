//! Fit `y = a·exp(−b·t)` to samples with Levenberg–Marquardt.
//!
//! The samples are noise-free and generated with `a = 3`, `b = 0.7`, so the
//! solver should recover those parameters to near machine precision.
//!
//! Run with: `cargo run -p tpt-opt-systems --example nls_exponential_fit --features nls`

use tpt_opt_systems::nls::dual::Dual;
use tpt_opt_systems::nls::{levenberg_marquardt, NlsConfig, NlsProblem};

fn main() {
    let times: Vec<f64> = (0..=10).map(|i| f64::from(i) * 0.5).collect();
    let samples: Vec<f64> = times.iter().map(|&t| 3.0 * (-0.7 * t).exp()).collect();

    // One residual per sample: model(t_k; a, b) - y_k. Autodiff supplies the Jacobian.
    let problem = NlsProblem::builder(2)
        .residual_block(times.len(), move |p, out| {
            for (k, o) in out.iter_mut().enumerate() {
                let decay = (p[1].clone() * Dual::plain(-times[k])).exp();
                *o = p[0].clone() * decay - samples[k];
            }
        })
        .build();

    let fit = levenberg_marquardt(&problem, &[1.0, 0.1], &NlsConfig::new())
        .expect("the problem is well formed");
    println!("status: {:?}", fit.status);
    println!("a = {:.9}, b = {:.9}", fit.x[0], fit.x[1]);
    println!("cost = {:e}", fit.cost);

    assert!((fit.x[0] - 3.0).abs() < 1e-6, "a = {}", fit.x[0]);
    assert!((fit.x[1] - 0.7).abs() < 1e-6, "b = {}", fit.x[1]);
}
