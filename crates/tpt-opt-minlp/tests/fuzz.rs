//! Fuzz testing for the MINLP outer-approximation solver: seeded random
//! convex programs with verified invariants.
//!
//! For each seed we build a convex quadratic objective over box bounds (one
//! variable forced integral) plus one linear constraint, then check:
//!
//! 1. *Bound consistency*: the returned point respects its bounds and any
//!    integer variables are integral.
//! 2. *Constraint satisfaction*: every nonlinear constraint holds within a
//!    loose multiple of the configured feasibility tolerance.
//! 3. *Objective consistency*: the reported objective equals the model's own
//!    evaluation at the returned point.
//! 4. *Certificate sanity*: the lower bound never exceeds the best primal
//!    value (for a minimisation problem).

use tpt_opt_minlp::model::{MinlpModel, VarKind};
use tpt_opt_minlp::oa::{outer_approximate, OaConfig};

/// Tiny deterministic xorshift RNG so failures are reproducible by seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[test]
fn fuzz_random_convex_minlps_invariants_hold() {
    // Ten seeds keeps the suite under ~2 minutes while still covering the
    // instance space (each OA solve runs several NLP subproblems).
    for seed in 1u64..=10 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_right(9) | 1);
        let n = 2 + rng.below(2) as usize; // 2..=3 variables

        // Random convex quadratic: sum w_i x_i^2 + c_i x_i, w_i > 0.
        let w: Vec<f64> = (0..n).map(|_| 0.5 + rng.below(20) as f64 * 0.25).collect();
        let c: Vec<f64> = (0..n).map(|_| -(rng.below(40) as f64)).collect();
        let obj = move |x: &[f64]| -> f64 {
            x.iter().zip(&w).zip(&c).map(|((&xi, wi), ci)| wi * xi * xi + ci * xi).sum()
        };

        let mut model = MinlpModel::new(n, obj);
        for i in 0..n {
            let ub = 2.0 + rng.below(4) as f64;
            let kind = if i == 0 { VarKind::Integer } else { VarKind::Continuous };
            model.set_var(i, kind, 0.0, ub);
        }

        // One linear knapsack-style constraint: sum a_i x_i <= b.
        let a: Vec<f64> = (0..n).map(|_| 0.5 + rng.below(15) as f64 * 0.5).collect();
        let b = 1.0 + rng.below(12) as f64;
        let a_c = a.clone();
        let a_g = a.clone();
        model.add_le(
            move |x: &[f64]| x.iter().zip(&a_c).map(|(xi, ai)| ai * xi).sum::<f64>() - b,
            move |_x: &[f64], out: &mut [f64]| {
                for (o, ai) in out.iter_mut().zip(&a_g) {
                    *o = *ai;
                }
            },
        );

        let cfg = OaConfig { max_iter: 40, ..OaConfig::default() };
        let res = outer_approximate(&model, &cfg);

        if let Some(x) = &res.x {
            // Bounds respected.
            for (i, &xi) in x.iter().enumerate() {
                assert!(
                    xi >= model.lbs[i] - 1e-4 && xi <= model.ubs[i] + 1e-4,
                    "seed {seed}: x[{i}]={xi} outside [{},{}]",
                    model.lbs[i],
                    model.ubs[i]
                );
                if model.vars[i] == VarKind::Integer {
                    assert!(
                        (xi - xi.round()).abs() < 1e-4,
                        "seed {seed}: integer var x[{i}]={xi} not integral"
                    );
                }
            }

            // Constraints satisfied (loose multiple of feas_tol).
            for ci in 0..model.constraints.len() {
                let v = model.violation(ci, x, 1e-4);
                assert!(v <= 1e-4, "seed {seed}: constraint {ci} violated by {v}");
            }

            // Objective consistency.
            let true_obj = model.eval_objective(x);
            let reported = res.objective.expect("x present implies objective present");
            assert!(
                (true_obj - reported).abs() < 1e-4,
                "seed {seed}: reported objective {reported} != evaluated {true_obj}"
            );
        }

        // Certificate sanity: lower bound cannot exceed best primal value.
        if let Some(obj) = res.objective {
            assert!(
                res.lower_bound <= obj + 1e-4,
                "seed {seed}: lower bound {} exceeds primal {obj}",
                res.lower_bound
            );
        }
    }
}
