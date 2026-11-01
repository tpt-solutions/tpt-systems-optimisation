//! Minimise a noisy-free two-dimensional bowl with Bayesian optimisation.
//!
//! The objective has its minimum `1.0` at `(0.3, -0.2)`. The surrogate-driven
//! search should get close with a budget of 40 evaluations.
//!
//! Run with: `cargo run -p tpt-opt-systems --example bayes_minimise --features bayes`

use tpt_opt_systems::bayes::{minimise, BoConfig};

fn main() {
    let bowl = |x: &[f64]| (x[0] - 0.3).powi(2) + 2.0 * (x[1] + 0.2).powi(2) + 1.0;
    let bounds = [(-1.0, 1.0), (-1.0, 1.0)];

    let result = minimise(bowl, &bounds, 40, &BoConfig::default()).expect("valid bounds");
    println!("evaluations: {}", result.evaluations);
    println!("best x = [{:.4}, {:.4}], f = {:.6}", result.x[0], result.x[1], result.f);

    assert!(result.f < 1.05, "f = {}", result.f);
}
