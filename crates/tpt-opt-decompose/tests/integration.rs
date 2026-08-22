//! Integration tests for `tpt-opt-decompose`.

use tpt_opt_decompose::lagrangian::{
    lagrangian_bundle_level, lagrangian_subgradient, surrogate_search,
};
use tpt_opt_decompose::{
    detect_structure, BendersProblem, BendersSolver, BlockRow, BranchAndPrice, Column,
    DantzigWolfe, DualConfig, DwBlock, DwLocalRow, DwProblem, RecourseBlock, RmpPool, RowSense,
    Stabilization, Strategy,
};

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

// ---------------------------------------------------------------------------
// Benders
// ---------------------------------------------------------------------------

/// Two-stage capacity problem: integer capacity x ∈ [0, 4] at cost 2/unit;
/// demand d = 3 served by recourse y at cost 5/unit with y ≥ d − x.
#[test]
fn benders_capacity_single_scenario() {
    let problem = BendersProblem {
        first_cost: vec![2.0],
        first_bounds: vec![(0.0, 4.0)],
        first_integer: vec![true],
        blocks: vec![RecourseBlock {
            cost: vec![5.0],
            rows: vec![BlockRow { y: vec![1.0], x: vec![1.0], sense: RowSense::Ge, rhs: 3.0 }],
            y_upper: vec![f64::INFINITY],
        }],
        weights: vec![1.0],
    };
    let res = BendersSolver::new(&problem).solve().unwrap();
    assert_eq!(res.status, tpt_opt_core::SolverStatus::Optimal);
    assert!(approx(res.objective, 6.0, 1e-5), "objective {}", res.objective);
    assert!(approx(res.x[0], 3.0, 1e-6));
}

/// Two scenarios with unequal weights; optimum balances both.
/// Q1(x): y ≥ 2 − x cost 4 ; Q2(x): z ≥ 3 − x cost 6 ; c = [1].
/// f(x) = x + 0.5·4·max(0,2−x) + 0.5·6·max(0,3−x), x ∈ [0, 4] continuous.
/// For x ≤ 2: x + 4 − 2x + 9 − 3x = 13 − 4x → decreasing.
/// For 2 ≤ x ≤ 3: x + 9 − 3x = 9 − 2x → decreasing.
/// For x ≥ 3: x → increasing. Optimum x = 3, value 3.
#[test]
fn benders_two_scenarios_weighted() {
    let problem = BendersProblem {
        first_cost: vec![1.0],
        first_bounds: vec![(0.0, 4.0)],
        first_integer: vec![false],
        blocks: vec![
            RecourseBlock {
                cost: vec![4.0],
                rows: vec![BlockRow { y: vec![1.0], x: vec![1.0], sense: RowSense::Ge, rhs: 2.0 }],
                y_upper: vec![f64::INFINITY],
            },
            RecourseBlock {
                cost: vec![6.0],
                rows: vec![BlockRow { y: vec![1.0], x: vec![1.0], sense: RowSense::Ge, rhs: 3.0 }],
                y_upper: vec![f64::INFINITY],
            },
        ],
        weights: vec![0.5, 0.5],
    };
    let res = BendersSolver::new(&problem).solve().unwrap();
    assert_eq!(res.status, tpt_opt_core::SolverStatus::Optimal);
    assert!(approx(res.objective, 3.0, 1e-5), "objective {}", res.objective);
    assert!(approx(res.x[0], 3.0, 1e-4));
}

/// Feasibility cuts: recourse infeasible unless x ≥ 2
/// (row: y ≥ 4 − 2x with y ≤ 0 upper bound ⇒ requires 4 − 2x ≤ 0).
#[test]
fn benders_feasibility_cuts() {
    let problem = BendersProblem {
        first_cost: vec![1.0],
        first_bounds: vec![(0.0, 5.0)],
        first_integer: vec![true],
        blocks: vec![RecourseBlock {
            cost: vec![1.0],
            rows: vec![BlockRow { y: vec![1.0], x: vec![2.0], sense: RowSense::Ge, rhs: 4.0 }],
            y_upper: vec![0.0],
        }],
        weights: vec![1.0],
    };
    let res = BendersSolver::new(&problem).solve().unwrap();
    assert_eq!(res.status, tpt_opt_core::SolverStatus::Optimal);
    // Cheapest feasible x is 2 (cost 2); y forced to 0.
    assert!(approx(res.objective, 2.0, 1e-5), "objective {}", res.objective);
    assert!(approx(res.x[0], 2.0, 1e-6));
}

/// Stabilisation must not change the answer (certified by unrestricted solve).
#[test]
fn benders_trust_region_matches_plain() {
    let problem = BendersProblem {
        first_cost: vec![2.0],
        first_bounds: vec![(0.0, 4.0)],
        first_integer: vec![true],
        blocks: vec![RecourseBlock {
            cost: vec![5.0],
            rows: vec![BlockRow { y: vec![1.0], x: vec![1.0], sense: RowSense::Ge, rhs: 3.0 }],
            y_upper: vec![f64::INFINITY],
        }],
        weights: vec![1.0],
    };
    let plain = BendersSolver::new(&problem).solve().unwrap();
    let tr = BendersSolver::new(&problem)
        .with_stabilization(Stabilization::TrustRegion { initial_delta: 0.0, max_delta: 10.0 })
        .solve()
        .unwrap();
    assert_eq!(tr.status, tpt_opt_core::SolverStatus::Optimal);
    assert!(approx(plain.objective, tr.objective, 1e-6));
}

// ---------------------------------------------------------------------------
// Dantzig–Wolfe
// ---------------------------------------------------------------------------

/// Two independent blocks coupled by one resource row:
/// block k has variables (a_k, b_k) with local row a_k + b_k ≥ 1 and costs
/// (1, 3); coupling: a_1 + a_2 ≥ 2. Optimum: use cheap a's: a_1 = a_2 = 1,
/// b_k = 0 ⇒ objective 2.
#[test]
fn dantzig_wolfe_two_blocks_coupling_row() {
    let problem = DwProblem {
        coupling_rhs: vec![2.0],
        coupling_sense: vec![RowSense::Ge],
        blocks: vec![
            DwBlock {
                cost: vec![1.0, 3.0],
                coupling: vec![vec![1.0, 0.0]],
                local_rows: vec![DwLocalRow {
                    coeffs: vec![1.0, 1.0],
                    sense: RowSense::Ge,
                    rhs: 1.0,
                }],
            },
            DwBlock {
                cost: vec![1.0, 3.0],
                coupling: vec![vec![1.0, 0.0]],
                local_rows: vec![DwLocalRow {
                    coeffs: vec![1.0, 1.0],
                    sense: RowSense::Ge,
                    rhs: 1.0,
                }],
            },
        ],
    };
    let res = DantzigWolfe::new(&problem).solve().unwrap();
    assert_eq!(res.status, tpt_opt_core::SolverStatus::Optimal);
    assert!(approx(res.objective, 2.0, 1e-5), "objective {}", res.objective);
    for p in &res.points {
        assert!(!p.is_empty());
        assert!(approx(p[0] + p[1], 1.0, 1e-5));
    }
    let total_a: f64 = res.points.iter().map(|p| p[0]).sum();
    assert!(total_a >= 2.0 - 1e-5);
}

/// Equality coupling: Σ a_k = 2 exactly.
#[test]
fn dantzig_wolfe_equality_coupling() {
    let problem = DwProblem {
        coupling_rhs: vec![2.0],
        coupling_sense: vec![RowSense::Eq],
        blocks: vec![
            DwBlock {
                cost: vec![1.0, 3.0],
                coupling: vec![vec![1.0, 0.0]],
                local_rows: vec![DwLocalRow {
                    coeffs: vec![1.0, 1.0],
                    sense: RowSense::Ge,
                    rhs: 1.0,
                }],
            },
            DwBlock {
                cost: vec![1.0, 3.0],
                coupling: vec![vec![1.0, 0.0]],
                local_rows: vec![DwLocalRow {
                    coeffs: vec![1.0, 1.0],
                    sense: RowSense::Ge,
                    rhs: 1.0,
                }],
            },
        ],
    };
    let res = DantzigWolfe::new(&problem).solve().unwrap();
    assert_eq!(res.status, tpt_opt_core::SolverStatus::Optimal);
    assert!(approx(res.objective, 2.0, 1e-5));
}

/// Infeasible coupling (demand exceeds what blocks can supply).
#[test]
fn dantzig_wolfe_detects_infeasibility() {
    let problem = DwProblem {
        coupling_rhs: vec![100.0],
        coupling_sense: vec![RowSense::Ge],
        blocks: vec![DwBlock {
            cost: vec![1.0],
            coupling: vec![vec![1.0]],
            local_rows: vec![DwLocalRow { coeffs: vec![1.0], sense: RowSense::Le, rhs: 1.0 }],
        }],
    };
    let res = DantzigWolfe::new(&problem).solve().unwrap();
    assert_eq!(res.status, tpt_opt_core::SolverStatus::Infeasible);
}

/// Pool deduplication and capacity cap behave as documented.
#[test]
fn rmp_pool_dedup_and_cap() {
    let mut pool = RmpPool::new().with_max_columns(2);
    let mk = |v: f64| Column { block: 0, cost: v, coeffs: vec![v], point: vec![v] };
    assert!(pool.try_insert(mk(1.0)));
    assert!(!pool.try_insert(mk(1.0))); // duplicate
    assert!(pool.try_insert(mk(2.0)));
    assert!(!pool.try_insert(mk(3.0))); // full
    assert_eq!(pool.columns().len(), 2);
}

// ---------------------------------------------------------------------------
// Branch-and-price (cutting stock)
// ---------------------------------------------------------------------------

/// Cutting stock: roll width 10, order widths {5, 4, 3} each with demand 3.
/// Enumerate all patterns of (c5, c4, c3) with 5c5+4c4+3c3 ≤ 10 and price by
/// enumeration; compare against the monolithic pattern-MILP optimum.
#[test]
fn branch_price_cutting_stock() {
    let demands = [3.0f64, 3.0, 3.0];
    let capacity = 10usize;

    // All feasible patterns.
    let mut patterns: Vec<[usize; 3]> = Vec::new();
    for c5 in 0..=2 {
        for c4 in 0..=2 {
            for c3 in 0..=3 {
                if 5 * c5 + 4 * c4 + 3 * c3 <= capacity {
                    patterns.push([c5, c4, c3]);
                }
            }
        }
    }

    // Monolithic reference MILP over all patterns.
    {
        use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
        use tpt_opt_core::solver::Solver;
        use tpt_opt_core::VarBound;
        use tpt_opt_milp::MilpSolver;
        let mut model = Model::new(patterns.len());
        for v in model.variables.iter_mut() {
            v.bound = VarBound::integer(0.0, f64::INFINITY);
        }
        model.set_objective(Objective {
            sense: Sense::Minimize,
            indices: (0..patterns.len()).collect(),
            coeffs: vec![1.0; patterns.len()],
            constant: 0.0,
        });
        for (i, &demand) in demands.iter().enumerate() {
            let idx: Vec<usize> = (0..patterns.len()).filter(|&p| patterns[p][i] > 0).collect();
            let co: Vec<f64> = idx.iter().map(|&p| patterns[p][i] as f64).collect();
            model.add_constraint(Constraint::ge(idx, co, demand));
        }
        let sol = MilpSolver::new().solve(&model).unwrap();
        assert_eq!(sol.status, tpt_opt_core::SolverStatus::Optimal);

        // Branch-and-price with an enumerative knapsack pricer.
        struct KnapsackPricer {
            patterns: Vec<[usize; 3]>,
        }
        impl tpt_opt_decompose::Pricer for KnapsackPricer {
            fn price(
                &mut self,
                _block: usize,
                pi: &[f64],
                sigma_k: f64,
            ) -> Result<Option<Column>, tpt_opt_core::OptError> {
                let mut best: Option<(f64, usize)> = None;
                for (pi_idx, pat) in self.patterns.iter().enumerate() {
                    let rc = 1.0
                        - pi.iter().zip(pat.iter()).map(|(&p, &c)| p * c as f64).sum::<f64>()
                        - sigma_k;
                    if best.map_or(true, |(b, _)| rc < b) {
                        best = Some((rc, pi_idx));
                    }
                }
                match best {
                    Some((rc, i)) if rc < -1e-7 => {
                        let pat = self.patterns[i];
                        Ok(Some(Column {
                            block: 0,
                            cost: 1.0,
                            coeffs: pat.iter().map(|&c| c as f64).collect(),
                            point: pat.iter().map(|&c| c as f64).collect(),
                        }))
                    }
                    _ => Ok(None),
                }
            }

            fn price_cleanup(
                &mut self,
                _block: usize,
                pi: &[f64],
                sigma_k: f64,
            ) -> Result<Vec<Column>, tpt_opt_core::OptError> {
                // Dual-neutral patterns (rc ≤ 0 up to tolerance) are often
                // pivotal for the integer master; hand them all over.
                Ok(self
                    .patterns
                    .iter()
                    .filter(|pat| {
                        let rc = 1.0
                            - pi.iter().zip(pat.iter()).map(|(&p, &c)| p * c as f64).sum::<f64>()
                            - sigma_k;
                        rc <= 1e-6
                    })
                    .map(|pat| Column {
                        block: 0,
                        cost: 1.0,
                        coeffs: pat.iter().map(|&c| c as f64).collect(),
                        point: pat.iter().map(|&c| c as f64).collect(),
                    })
                    .collect())
            }
        }

        let problem = DwProblem {
            coupling_rhs: demands.to_vec(),
            coupling_sense: vec![RowSense::Ge; 3],
            blocks: vec![DwBlock {
                cost: vec![1.0],
                coupling: vec![vec![1.0]],
                local_rows: vec![],
            }],
        };
        let bp = BranchAndPrice::new(&problem, KnapsackPricer { patterns })
            .with_convexity(false)
            .solve()
            .unwrap();
        assert_eq!(bp.status, tpt_opt_core::SolverStatus::Optimal);
        assert!(
            approx(bp.objective, sol.objective_value, 1e-5),
            "bp {} vs milp {}",
            bp.objective,
            sol.objective_value
        );
    }
}

// ---------------------------------------------------------------------------
// Lagrangian
// ---------------------------------------------------------------------------

/// Concave piecewise-linear dual of a tiny covering relaxation:
/// L(λ) = min_{x∈[0,1]²} 3x₁ + 2x₂ + λ(2 − 2x₁ − x₂), maximised over λ ≥ 0.
/// Piecewise: L = 2λ for λ ∈ [0, 3/2]; L = 3 for λ ∈ [3/2, 2];
/// L = 5 − λ for λ ≥ 2 ⇒ analytic peak 3 at λ ∈ [1.5, 2]
/// (= the constrained optimum min 3x₁+2x₂ s.t. 2x₁+x₂ ≥ 2, which is 3).
#[test]
fn lagrangian_subgradient_reaches_optimum() {
    // Oracle: minimise over x∈[0,1]^2 the Lagrangian; returns value/subgrad.
    let eval = |lam: &[f64]| -> f64 {
        // min 3x1+2x2+λ(2−2x1−x2) = 2λ + min(0, 3−2λ) + min(0, 2−λ)
        let mut v = 2.0 * lam[0];
        v += (3.0 - 2.0 * lam[0]).min(0.0);
        v += (2.0 - lam[0]).min(0.0);
        v
    };
    let grad = |lam: &[f64]| -> Vec<f64> {
        // ∂L/∂λ = 2 − 2x1 − x2 at any inner minimiser.
        let x1 = if 3.0 - 2.0 * lam[0] < 0.0 { 1.0 } else { 0.0 };
        let x2 = if 2.0 - lam[0] < 0.0 { 1.0 } else { 0.0 };
        vec![2.0 - 2.0 * x1 - x2]
    };
    let cfg = DualConfig { max_iterations: 200, initial_step: 0.5, target: None, tolerance: 1e-8 };
    let res = lagrangian_subgradient(vec![0.0], &cfg, eval, grad);
    assert!(res.value >= 3.0 - 1e-2 && res.value <= 3.0 + 1e-6, "dual value {}", res.value);
}

/// Bundle method on the same dual must reach the same peak.
#[test]
fn lagrangian_bundle_level_matches_subgradient() {
    let eval = |lam: &[f64]| -> f64 {
        let mut v = 2.0 * lam[0];
        v += (3.0 - 2.0 * lam[0]).min(0.0);
        v += (2.0 - lam[0]).min(0.0);
        v
    };
    let grad = |lam: &[f64]| -> Vec<f64> {
        let x1 = if 3.0 - 2.0 * lam[0] < 0.0 { 1.0 } else { 0.0 };
        let x2 = if 2.0 - lam[0] < 0.0 { 1.0 } else { 0.0 };
        vec![2.0 - 2.0 * x1 - x2]
    };
    let cfg = DualConfig { max_iterations: 60, initial_step: 1.0, target: None, tolerance: 1e-8 };
    let res = lagrangian_bundle_level(vec![0.0], &cfg, eval, grad).unwrap();
    assert!(res.value >= 3.0 - 1e-3, "bundle dual value {}", res.value);
}

/// Surrogate search improves over μ = 1 on a simple concave response.
#[test]
fn surrogate_search_improves() {
    // S(μ) = −(μ − 2)² + 4, concave peak at μ = 2 with value 4.
    let cfg = DualConfig::default();
    let (mu, val) = surrogate_search(1, &cfg, |m: &[f64]| -(m[0] - 2.0) * (m[0] - 2.0) + 4.0);
    assert!(val > 3.5, "surrogate value {}", val);
    assert!(approx(mu[0], 2.0, 0.75), "mu {}", mu[0]);
}

// ---------------------------------------------------------------------------
// Structure detection
// ---------------------------------------------------------------------------

#[test]
fn structure_detects_independent_blocks() {
    use tpt_opt_core::model::{Constraint, Model};
    let mut model = Model::new(4);
    model.add_constraint(Constraint::ge(vec![0, 1], vec![1.0, 1.0], 1.0));
    model.add_constraint(Constraint::ge(vec![2, 3], vec![1.0, 1.0], 1.0));
    let report = detect_structure(&model);
    assert_eq!(report.num_components, 2);
    assert!(report.linking_rows.is_empty());
    assert!(report.linking_cols.is_empty());
    assert_eq!(report.strategy, Strategy::IndependentBlocks);
}

#[test]
fn structure_detects_linking_row_as_dantzig_wolfe() {
    use tpt_opt_core::model::{Constraint, Model};
    let mut model = Model::new(4);
    model.add_constraint(Constraint::ge(vec![0, 1], vec![1.0, 1.0], 1.0));
    model.add_constraint(Constraint::ge(vec![2, 3], vec![1.0, 1.0], 1.0));
    model.add_constraint(Constraint::le(vec![0, 2], vec![1.0, 1.0], 3.0)); // links blocks
    let report = detect_structure(&model);
    assert_eq!(report.num_components, 2);
    assert_eq!(report.linking_rows, vec![2]);
    assert_eq!(report.strategy, Strategy::DantzigWolfe);
}

#[test]
fn structure_dense_model_is_direct() {
    use tpt_opt_core::model::{Constraint, Model};
    let mut model = Model::new(3);
    model.add_constraint(Constraint::ge(vec![0, 1, 2], vec![1.0, 1.0, 1.0], 1.0));
    let report = detect_structure(&model);
    assert_eq!(report.num_components, 1);
    assert_eq!(report.strategy, Strategy::Direct);
}
