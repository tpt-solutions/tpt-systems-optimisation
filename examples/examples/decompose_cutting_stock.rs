//! Large-scale decomposition through the umbrella crate: the classic cutting-
//! stock problem solved by **branch-and-price** — column generation over an
//! exponential pattern pool with an enumerative knapsack pricer, then an
//! integer restricted-master solve. Cross-checked against a monolithic MILP
//! over all patterns.
//!
//! Instance: rolls of width 10; orders for widths 5, 4 and 3, three of each.
//! Known optimum: 4 rolls.
//!
//! Run with: cargo run --manifest-path examples/Cargo.toml --example decompose_cutting_stock

use tpt_opt_systems::core::{Solver, SolverStatus};
use tpt_opt_systems::decompose::{
    BranchAndPrice, Column, DwBlock, DwProblem, Pricer, RowSense,
};
use tpt_opt_systems::{milp, Constraint, Model, Objective, VarBound};

/// All cutting patterns `(c5, c4, c3)` with `5·c5 + 4·c4 + 3·c3 <= capacity`.
fn enumerate_patterns(capacity: usize) -> Vec<[usize; 3]> {
    let mut patterns = Vec::new();
    for c5 in 0..=(capacity / 5) {
        for c4 in 0..=((capacity - 5 * c5) / 4) {
            for c3 in 0..=((capacity - 5 * c5 - 4 * c4) / 3) {
                patterns.push([c5, c4, c3]);
            }
        }
    }
    patterns
}

/// Pricer that scans every feasible pattern for the most negative reduced
/// cost, plus a dual-neutral cleanup pass (cheap patterns that do not help
/// the LP bound are often pivotal for the integer master).
struct KnapsackPricer {
    patterns: Vec<[usize; 3]>,
}

impl KnapsackPricer {
    fn best(&self, pi: &[f64], sigma_k: f64) -> Option<(f64, [usize; 3])> {
        self.patterns
            .iter()
            .map(|pat| {
                let rc = 1.0
                    - pi.iter().zip(pat.iter()).map(|(&p, &c)| p * c as f64).sum::<f64>()
                    - sigma_k;
                (rc, *pat)
            })
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
    }

    fn column_for(pat: &[usize; 3]) -> Column {
        Column {
            block: 0,
            cost: 1.0, // one roll per pattern
            coeffs: pat.iter().map(|&c| c as f64).collect(),
            point: pat.iter().map(|&c| c as f64).collect(),
        }
    }
}

impl Pricer for KnapsackPricer {
    fn price(
        &mut self,
        _block: usize,
        pi: &[f64],
        sigma_k: f64,
    ) -> Result<Option<Column>, tpt_opt_systems::core::OptError> {
        match self.best(pi, sigma_k) {
            Some((rc, pat)) if rc < -1e-7 => Ok(Some(Self::column_for(&pat))),
            _ => Ok(None),
        }
    }

    fn price_cleanup(
        &mut self,
        _block: usize,
        pi: &[f64],
        sigma_k: f64,
    ) -> Result<Vec<Column>, tpt_opt_systems::core::OptError> {
        Ok(self
            .patterns
            .iter()
            .filter(|pat| {
                let rc = 1.0
                    - pi.iter().zip(pat.iter()).map(|(&p, &c)| p * c as f64).sum::<f64>()
                    - sigma_k;
                rc <= 1e-6
            })
            .map(Self::column_for)
            .collect())
    }
}

fn main() {
    let demands = [3.0f64, 3.0, 3.0];
    let capacity = 10usize;
    let patterns = enumerate_patterns(capacity);
    println!("{} feasible patterns", patterns.len());

    // --- monolithic reference MILP over all patterns ---------------------------
    let mut mono = Model::new(patterns.len());
    for v in mono.variables.iter_mut() {
        v.bound = VarBound::integer(0.0, f64::INFINITY);
    }
    mono.set_objective(Objective::minimize((0..patterns.len()).collect(), vec![1.0; patterns.len()]));
    for (i, &demand) in demands.iter().enumerate() {
        let idx: Vec<usize> = (0..patterns.len()).filter(|&p| patterns[p][i] > 0).collect();
        let co: Vec<f64> = idx.iter().map(|&p| patterns[p][i] as f64).collect();
        mono.add_constraint(Constraint::ge(idx, co, demand));
    }
    let mono_sol = milp::MilpSolver::new().solve(&mono).expect("monolithic MILP solves");
    println!("monolithic MILP: {} rolls", mono_sol.objective_value);

    // --- branch-and-price -------------------------------------------------------
    // One block whose "local polyhedron" is trivial; all structure lives in
    // the coupling rows (demand coverage) and is priced by the knapsack scan.
    let problem = DwProblem {
        coupling_rhs: demands.to_vec(),
        coupling_sense: vec![RowSense::Ge; 3],
        blocks: vec![DwBlock { cost: vec![1.0], coupling: vec![vec![1.0]], local_rows: vec![] }],
    };
    let bp = BranchAndPrice::new(&problem, KnapsackPricer { patterns })
        .with_convexity(false) // set-covering style master: unbounded multiplicity
        .solve()
        .expect("branch-and-price solves");
    println!(
        "branch-and-price: {} rolls ({} pricing rounds, {} columns in pool)",
        bp.objective, bp.pricing_rounds, bp.columns
    );

    assert_eq!(bp.status, SolverStatus::Optimal);
    assert!((bp.objective - 4.0).abs() < 1e-6, "known optimum is 4 rolls");
    assert!((bp.objective - mono_sol.objective_value).abs() < 1e-6);
}