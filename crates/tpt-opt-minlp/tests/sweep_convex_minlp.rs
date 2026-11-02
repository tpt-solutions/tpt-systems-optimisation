//! Randomised check of the MINLP solvers on small convex instances.
//!
//! Each instance has one integer variable `z ∈ {0, …, 3}` and one continuous
//! variable `x`. The objective is a convex quadratic, and the constraints are
//! linear in `x` for fixed `z`, so every inner problem is convex and unimodal.
//! The reference enumerates `z` and minimises over `x` with a ternary search,
//! which gives the global optimum. Outer approximation, generalised Benders and
//! SQP branch-and-bound must match it, and must not report a point where the
//! reference finds none.

use tpt_opt_minlp::gbd::{generalized_benders, GbdConfig};
use tpt_opt_minlp::model::{MinlpModel, VarKind};
use tpt_opt_minlp::oa::{outer_approximate, OaConfig, OaStatus};
use tpt_opt_minlp::sqp::{sqp_branch_and_bound, SqpConfig};

const Z_MAX: i64 = 3;

/// Deterministic generator, so any failure reproduces exactly.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn real(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.unit() * (hi - lo)
    }
}

#[derive(Clone, Copy)]
struct Instance {
    /// Objective `(x − a·z − b)² + k·(z − d)² + e·x`.
    a: f64,
    b: f64,
    k: f64,
    d: f64,
    e: f64,
    /// Linear constraints on `x`: `x ≤ p·z + q` and `x ≥ r·z + s`.
    p: f64,
    q: f64,
    r: f64,
    s: f64,
    /// Box for `x`.
    lo: f64,
    hi: f64,
}

fn random_instance(rng: &mut Lcg) -> Instance {
    Instance {
        a: rng.real(-1.0, 1.0),
        b: rng.real(-1.0, 2.0),
        k: rng.real(0.1, 2.0),
        d: rng.real(0.0, Z_MAX as f64),
        e: rng.real(-1.0, 1.0),
        p: rng.real(-0.5, 0.5),
        q: rng.real(-1.0, 3.0),
        r: rng.real(-0.5, 0.5),
        s: rng.real(-2.0, 1.0),
        lo: -2.0,
        hi: 3.0,
    }
}

fn objective(inst: Instance, z: f64, x: f64) -> f64 {
    (x - inst.a * z - inst.b).powi(2) + inst.k * (z - inst.d).powi(2) + inst.e * x
}

/// Reference optimum by enumerating `z` and ternary-searching `x`, or `None` when
/// no `z` admits a feasible `x`.
fn reference(inst: Instance) -> Option<f64> {
    let mut best: Option<f64> = None;
    for z in 0..=Z_MAX {
        let zf = z as f64;
        let lo = inst.lo.max(inst.r * zf + inst.s);
        let hi = inst.hi.min(inst.p * zf + inst.q);
        if lo > hi + 1e-12 {
            continue;
        }
        let (mut a, mut b) = (lo, hi.max(lo));
        for _ in 0..300 {
            let m1 = a + (b - a) / 3.0;
            let m2 = b - (b - a) / 3.0;
            if objective(inst, zf, m1) < objective(inst, zf, m2) {
                b = m2;
            } else {
                a = m1;
            }
        }
        let v = objective(inst, zf, 0.5 * (a + b));
        best = Some(best.map_or(v, |x| x.min(v)));
    }
    best
}

fn model(inst: Instance) -> MinlpModel {
    let mut m = MinlpModel::new(2, move |x: &[f64]| objective(inst, x[0], x[1]));
    m.set_var(0, VarKind::Integer, 0.0, Z_MAX as f64);
    m.set_var(1, VarKind::Continuous, inst.lo, inst.hi);
    // x − p·z − q ≤ 0
    m.add_le(
        move |x: &[f64]| x[1] - inst.p * x[0] - inst.q,
        move |_x: &[f64], g: &mut [f64]| {
            g[0] = -inst.p;
            g[1] = 1.0;
        },
    );
    // r·z + s − x ≤ 0
    m.add_le(
        move |x: &[f64]| inst.r * x[0] + inst.s - x[1],
        move |_x: &[f64], g: &mut [f64]| {
            g[0] = inst.r;
            g[1] = -1.0;
        },
    );
    m
}

#[test]
fn minlp_solvers_match_enumeration_on_convex_instances() {
    let mut rng = Lcg(91);
    let (mut feasible, mut infeasible) = (0, 0);
    for case in 0..30 {
        let inst = random_instance(&mut rng);
        let truth = reference(inst);
        let oa_cfg = OaConfig { max_iter: 300, abs_gap: 1e-6, ..OaConfig::default() };
        let oa = outer_approximate(&model(inst), &oa_cfg);
        let gbd = generalized_benders(&model(inst), &GbdConfig::default());
        let sqp = sqp_branch_and_bound(&model(inst), &SqpConfig::default());
        match truth {
            Some(best) => {
                feasible += 1;
                for (name, status, value) in
                    [("OA", oa.status, oa.objective), ("SQP", sqp.status, sqp.objective)]
                {
                    assert_eq!(status, OaStatus::Optimal, "case {case}: {name} status");
                    let v =
                        value.unwrap_or_else(|| panic!("case {case}: {name} returned no value"));
                    assert!(
                        (v - best).abs() < 1e-3,
                        "case {case}: {name} {v} vs enumeration {best}"
                    );
                }
            }
            None => {
                infeasible += 1;
                for (name, status, value) in [
                    ("OA", oa.status, oa.objective),
                    ("GBD", gbd.status, gbd.objective),
                    ("SQP", sqp.status, sqp.objective),
                ] {
                    assert!(
                        status != OaStatus::Optimal || value.is_none(),
                        "case {case}: {name} reports an optimum on an infeasible instance"
                    );
                }
            }
        }
    }
    assert!(feasible > 8, "only {feasible} feasible instances (infeasible {infeasible})");
}
