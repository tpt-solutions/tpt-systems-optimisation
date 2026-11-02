//! Strong-duality check of the LP engine's duals on random bounded LPs.
//!
//! At an optimum the dual objective, `bᵀy` plus each variable's reduced cost
//! times its active bound, must equal the primal objective. A wrong dual, sign
//! or magnitude, breaks that equality, so this catches dual errors that a
//! primal-only check would miss. Duals are what column generation prices with.

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::{Tolerances, VarBound};
use tpt_opt_milp::lp::{solve_lp, LpStatus};

/// Deterministic generator, so any failure reproduces exactly.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn int(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.unit() * (hi - lo + 1) as f64) as i64
    }
}

#[test]
fn lp_duals_satisfy_strong_duality() {
    let mut rng = Lcg(5);
    let mut checked = 0;
    for case in 0..3000 {
        let n = rng.int(1, 4) as usize;
        let mut m = Model::new(0);
        let mut lb = Vec::new();
        let mut ub = Vec::new();
        for _ in 0..n {
            let l = rng.int(-2, 1) as f64;
            let u = l + rng.int(1, 5) as f64;
            lb.push(l);
            ub.push(u);
            m.add_variable(VarBound::continuous(l, u));
        }
        for _ in 0..rng.int(1, 3) {
            let idx: Vec<usize> = (0..n).collect();
            let coeffs: Vec<f64> = (0..n).map(|_| rng.int(-3, 3) as f64).collect();
            let rhs = rng.int(-4, 8) as f64;
            let con = match rng.int(0, 2) {
                0 => Constraint::le(idx, coeffs, rhs),
                1 => Constraint::ge(idx, coeffs, rhs),
                _ => Constraint::equality(idx, coeffs, rhs),
            };
            m.add_constraint(con);
        }
        let c: Vec<f64> = (0..n).map(|_| rng.int(-3, 3) as f64).collect();
        m.set_objective(Objective {
            sense: Sense::Minimize,
            indices: (0..n).collect(),
            coeffs: c.clone(),
            constant: 0.0,
        });

        let s = solve_lp(&m, &lb, &ub, Tolerances::spec_default());
        if s.status != LpStatus::Optimal {
            continue;
        }
        checked += 1;

        // bᵀy, using the side of each row the dual points at (equality: its rhs).
        let mut dual_obj = 0.0;
        for (i, con) in m.constraints.iter().enumerate() {
            let y = s.dual[i];
            if y.abs() < 1e-12 {
                continue;
            }
            let b = if y > 0.0 { con.lower } else { con.upper };
            assert!(b.is_finite(), "case {case}: dual {y} on a row with no finite side");
            dual_obj += y * b;
        }
        // Reduced cost of each variable, taken at whichever bound it sits on.
        for j in 0..n {
            let mut d = c[j];
            for (i, con) in m.constraints.iter().enumerate() {
                for (&v, &a) in con.indices.iter().zip(con.coeffs.iter()) {
                    if v == j {
                        d -= s.dual[i] * a;
                    }
                }
            }
            dual_obj += if d >= 0.0 { d * lb[j] } else { d * ub[j] };
        }
        assert!(
            (dual_obj - s.objective).abs() < 1e-6,
            "case {case}: dual objective {dual_obj} vs primal {}",
            s.objective
        );
    }
    assert!(checked > 1000, "only {checked} optimal LPs were checked");
}
