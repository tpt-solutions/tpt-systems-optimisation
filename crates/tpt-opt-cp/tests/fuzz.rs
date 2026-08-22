//! Fuzz testing for the CP engine: seeded random models with verified
//! invariants.
//!
//! For each seed we generate a small random CSP (integer domains, random
//! linear constraints, occasional AllDifferent) and check two invariants:
//!
//! 1. *Soundness*: any assignment the solver returns satisfies every
//!    constraint's own `check`.
//! 2. *Completeness*: on tiny instances the full solution set returned by
//!    [`tpt_opt_cp::solver::solutions`] equals brute-force enumeration over
//!    the domain product.

use std::collections::BTreeSet;

use tpt_opt_cp::constraints::{AllDifferent, Linear};
use tpt_opt_cp::model::{CpModel, Relation};
use tpt_opt_cp::solver::{solutions, solve};

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

fn build_model(rng: &mut Rng) -> (CpModel, Vec<(usize, usize)>) {
    let n = 2 + rng.below(3) as usize; // 2..=4 variables
    let hi = 1 + rng.below(3) as usize; // domain [0, hi], hi in 1..=3
    let mut model = CpModel::new();
    let mut doms = Vec::with_capacity(n);
    for _ in 0..n {
        let v = model.add_var(0, hi);
        doms.push((v, hi));
    }

    let num_constraints = 1 + rng.below(3) as usize;
    for _ in 0..num_constraints {
        let terms: Vec<(usize, i64)> = (0..n)
            .filter_map(|i| {
                let r = rng.below(10);
                if r < 7 {
                    let coef = rng.below(5) as i64 - 2; // -2..=2
                    let coef = if coef == 0 { 1 } else { coef };
                    Some((i, coef))
                } else {
                    None
                }
            })
            .collect();
        if terms.is_empty() {
            continue;
        }
        let rel = match rng.below(3) {
            0 => Relation::Le,
            1 => Relation::Ge,
            _ => Relation::Eq,
        };
        // rhs in a band around the achievable range so roughly half the
        // generated instances are satisfiable.
        let rhs = rng.below((2 * hi as i64 + 2) as u64) as i64;
        model.add_constraint(Box::new(Linear::new(terms, rel, rhs)));
    }

    // Occasionally require distinctness (only meaningful when the joint
    // domain is large enough to admit permutations).
    if n >= 2 && hi >= n - 1 && rng.below(2) == 0 {
        let vs: Vec<usize> = (0..n).collect();
        model.add_constraint(Box::new(AllDifferent::new(vs)));
    }

    (model, doms)
}

/// Brute-force all assignments and keep those satisfying every constraint.
fn brute_force(model: &CpModel, doms: &[(usize, usize)]) -> BTreeSet<Vec<usize>> {
    let constraints = model.constraints();
    let mut out = BTreeSet::new();
    let mut assign = vec![0usize; model.num_vars()];
    fn rec(
        idx: usize,
        assign: &mut Vec<usize>,
        doms: &[(usize, usize)],
        constraints: &[Box<dyn tpt_opt_cp::constraints::Constraint>],
        out: &mut BTreeSet<Vec<usize>>,
    ) {
        if idx == doms.len() {
            if constraints.iter().all(|c| c.check(assign)) {
                out.insert(assign.clone());
            }
            return;
        }
        let (_, hi) = doms[idx];
        for v in 0..=hi {
            assign[doms[idx].0] = v;
            rec(idx + 1, assign, doms, constraints, out);
        }
    }
    rec(0, &mut assign, doms, constraints, &mut out);
    out
}

#[test]
fn fuzz_random_cps_sound_and_complete() {
    for seed in 1u64..=200 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let (model, doms) = build_model(&mut rng);

        // Soundness: whatever the solver returns must satisfy every
        // constraint's own predicate.
        if let Some(sol) = solve(&model) {
            for c in model.constraints() {
                assert!(
                    c.check(&sol.assignment),
                    "seed {seed}: solver returned an assignment violating a constraint"
                );
            }
        }

        // Completeness: the enumerated solution set equals brute force.
        let expected = brute_force(&model, &doms);
        let got: BTreeSet<Vec<usize>> =
            solutions(&model, 10_000).into_iter().map(|s| s.assignment).collect();
        assert_eq!(got, expected, "seed {seed}: solution set disagrees with brute force");
    }
}
