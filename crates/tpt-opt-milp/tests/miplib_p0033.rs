//! Integration test: MIPLIB 3.0 instance `p0033` solved to proven optimality.
//!
//! Data transcribed from the OSiL encoding distributed with the SCIP check
//! suite (scipopt/scip, `check/instances/MIP/p0033.osil`), which mirrors the
//! classic MIPLIB 3.0 instance: 33 binary variables, 15 knapsack-style rows,
//! known optimal objective value **3089**.
//!
//! The instance is embedded directly (a few KB of integers) so the test needs
//! no network access and adds no large fixtures to the published crate.

use tpt_opt_core::{
    bounds::VarBound,
    model::{Constraint, Model, Objective},
    solver::{Solver, SolverStatus},
};
use tpt_opt_milp::MilpSolver;

/// Objective coefficients (minimisation).
const OBJ: [f64; 33] = [
    171.0, 171.0, 171.0, 171.0, 163.0, 162.0, 163.0, 69.0, 69.0, 183.0, 183.0, 183.0, 183.0, 49.0,
    183.0, 258.0, 517.0, 250.0, 500.0, 250.0, 500.0, 159.0, 318.0, 159.0, 318.0, 159.0, 318.0,
    159.0, 318.0, 114.0, 228.0, 159.0, 318.0,
];

/// Rows as `(variable indices, coefficients, upper bound)`; every row is
/// `sum coef * x <= ub` (the OSiL encoding gives only upper bounds).
const ROWS: [(&[usize], &[f64], f64); 15] = [
    (
        // R114 <= 1
        &[0, 1, 2, 3],
        &[1.0, 1.0, 1.0, 1.0],
        1.0,
    ),
    (
        // R115 <= 1
        &[4, 5, 6],
        &[1.0, 1.0, 1.0],
        1.0,
    ),
    (
        // R116 <= 1
        &[7, 8],
        &[1.0, 1.0],
        1.0,
    ),
    (
        // R117 <= 1
        &[9, 10, 11, 12, 14],
        &[1.0, 1.0, 1.0, 1.0, 1.0],
        1.0,
    ),
    (
        // R118 <= -5
        &[9, 15, 16],
        &[-230.0, -200.0, -400.0],
        -5.0,
    ),
    (
        // R119 <= 2700
        &[2, 3, 4, 5, 7, 8, 11, 12, 13, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        &[
            300.0, 300.0, 285.0, 285.0, 265.0, 265.0, 230.0, 230.0, 190.0, 200.0, 400.0, 200.0,
            400.0, 200.0, 400.0, 200.0, 400.0, 200.0, 400.0,
        ],
        2700.0,
    ),
    (
        // R120 <= -2600
        &[2, 3, 4, 5, 7, 8, 11, 12, 13, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        &[
            -300.0, -300.0, -285.0, -285.0, -265.0, -265.0, -230.0, -230.0, -190.0, -200.0, -400.0,
            -200.0, -400.0, -200.0, -400.0, -200.0, -400.0, -200.0, -400.0,
        ],
        -2600.0,
    ),
    (
        // R121 <= -100
        &[3, 29, 30],
        &[-300.0, -200.0, -400.0],
        -100.0,
    ),
    (
        // R122 <= -900
        &[0, 5, 8, 13, 25, 26],
        &[-300.0, -285.0, -265.0, -190.0, -200.0, -400.0],
        -900.0,
    ),
    (
        // R123 <= -1656
        &[0, 2, 5, 8, 12, 13, 25, 26, 27, 28],
        &[-300.0, -300.0, -285.0, -265.0, -230.0, -190.0, -200.0, -400.0, -200.0, -400.0],
        -1656.0,
    ),
    (
        // R124 <= -335
        &[4, 7, 10, 21, 22],
        &[-285.0, -265.0, -230.0, -200.0, -400.0],
        -335.0,
    ),
    (
        // R125 <= -1026
        &[4, 7, 10, 11, 21, 22, 23, 24],
        &[-285.0, -265.0, -230.0, -230.0, -200.0, -400.0, -200.0, -400.0],
        -1026.0,
    ),
    (
        // R126 <= -5
        &[1, 17, 18],
        &[-300.0, -200.0, -400.0],
        -5.0,
    ),
    (
        // R127 <= -500
        &[1, 17, 18, 19, 20],
        &[-300.0, -200.0, -400.0, -200.0, -400.0],
        -500.0,
    ),
    (
        // R128 <= -270
        &[6, 31, 32],
        &[-285.0, -200.0, -400.0],
        -270.0,
    ),
];

fn build_model() -> Model {
    let mut m = Model::new(33);
    let idx: Vec<usize> = (0..33).collect();
    let coefs: Vec<f64> = OBJ.to_vec();
    m.set_objective(Objective::minimize(idx, coefs));
    for (vars, coefs, ub) in ROWS.iter() {
        m.add_constraint(Constraint::le(vars.to_vec(), coefs.to_vec(), *ub));
    }
    for v in m.variables.iter_mut() {
        v.bound = VarBound::binary();
    }
    m
}

#[test]
fn miplib_p0033_optimal_3089() {
    let m = build_model();
    let mut solver = MilpSolver::new().with_seed(42).with_cuts(true).with_parallel_cuts(3);
    let sol = solver.solve(&m).unwrap();
    assert_eq!(sol.status, SolverStatus::Optimal, "p0033 must be solved to proven optimality");
    assert!(
        (sol.objective_value - 3089.0).abs() < 1e-4,
        "p0033 optimum {} != known 3089",
        sol.objective_value
    );
    // The reported primal vector must be integral and feasible.
    let x: Vec<f64> = (0..33).map(|i| sol.primal_value(i).unwrap()).collect();
    for xi in &x {
        assert!((xi - xi.round()).abs() < 1e-6, "non-integral value {xi}");
    }
    for (vars, coefs, ub) in ROWS.iter() {
        let act: f64 = vars.iter().zip(coefs.iter()).map(|(&v, &c)| c * x[v]).sum();
        assert!(act <= ub + 1e-6, "row violated: {act} > {ub}");
    }
}
