//! Closed-form check of the semidefinite path.
//!
//! The matrix `X(x) = [[1 + x₀, x₁], [x₁, 1 − x₀]]` is positive semidefinite
//! exactly when `(1 + x₀)(1 − x₀) ≥ x₁²`, that is when `x` lies in the unit disc.
//! Maximising `c·x` over the disc gives `‖c‖₂`, attained at `c/‖c‖`. The SDP
//! solver must reach that value and return a point inside the disc.

use tpt_opt_conic::{solve_socp, ConeProgram, ConicStatus, SdpBlock};
use tpt_opt_core::model::Sense;

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

fn disc_program(c: Vec<f64>) -> ConeProgram {
    // X(x) = X₀ + x₀·X₁ + x₁·X₂ with X₀ = I, X₁ = diag(1, −1), X₂ = [[0,1],[1,0]].
    let block = SdpBlock {
        dim: 2,
        x0: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        xs: vec![vec![vec![1.0, 0.0], vec![0.0, -1.0]], vec![vec![0.0, 1.0], vec![1.0, 0.0]]],
    };
    ConeProgram {
        n: 2,
        c,
        sense: Sense::Maximize,
        bounds: vec![(-2.0, 2.0), (-2.0, 2.0)],
        eq_a: Vec::new(),
        eq_b: Vec::new(),
        soc_rows: Vec::new(),
        sdp_blocks: vec![block],
    }
}

#[test]
fn sdp_disc_matches_the_closed_form() {
    let mut rng = Lcg(101);
    for case in 0..12 {
        let c = vec![rng.real(-2.0, 2.0), rng.real(-2.0, 2.0)];
        let norm = (c[0] * c[0] + c[1] * c[1]).sqrt();
        if norm < 0.2 {
            continue;
        }
        let sol = solve_socp(&disc_program(c.clone()), 1e-6, 400);
        assert_eq!(sol.status, ConicStatus::Optimal, "case {case}: {sol:?}");
        let inside = sol.x[0] * sol.x[0] + sol.x[1] * sol.x[1];
        assert!(inside <= 1.0 + 1e-4, "case {case}: point {:?} is outside the disc", sol.x);
        assert!(
            (sol.objective - norm).abs() < 2e-3,
            "case {case}: objective {} vs closed form {norm}",
            sol.objective
        );
    }
}
