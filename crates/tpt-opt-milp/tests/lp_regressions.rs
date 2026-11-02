//! Regression tests for the MILP engine's LP relaxation.
//!
//! Each case once produced a wrong answer. A basic artificial could stay
//! positive while phase I reported feasibility, an artificial could grow in
//! phase II, and so could an equality row (the cases below are the traced
//! instances; a random cross-check against the network LP caught the rest).

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::{Solver, SolverStatus, Tolerances, VarBound};
use tpt_opt_milp::{solve_lp, LpStatus, MilpSolver};

fn lp(model: &Model) -> (LpStatus, f64, Vec<f64>) {
    let lb: Vec<f64> = model.variables.iter().map(|v| v.bound.bound.lower).collect();
    let ub: Vec<f64> = model.variables.iter().map(|v| v.bound.bound.upper).collect();
    let s = solve_lp(model, &lb, &ub, Tolerances::spec_default());
    (s.status, s.objective, s.x)
}

/// Largest violation of any constraint at `x`.
fn violation(model: &Model, x: &[f64]) -> f64 {
    model.constraints.iter().map(|c| -c.slack(x)).fold(0.0, f64::max)
}

#[test]
fn equality_pair_with_a_conflicting_inequality_is_infeasible() {
    // x1 = 1/3 and x0 = 5/6 are forced, and then -3·x0 + 2·x1 ≈ -1.83 cannot be ≤ -4.
    let mut m = Model::new(0);
    m.add_variable(VarBound::continuous(-1.0, 4.0));
    m.add_variable(VarBound::continuous(-3.0, 4.0));
    m.add_constraint(Constraint::equality(vec![0, 1], vec![2.0, -2.0], 1.0));
    m.add_constraint(Constraint::equality(vec![0, 1], vec![0.0, 3.0], 1.0));
    m.add_constraint(Constraint::le(vec![0, 1], vec![-3.0, 2.0], -4.0));
    m.set_objective(Objective {
        sense: Sense::Minimize,
        indices: vec![0, 1],
        coeffs: vec![-3.0, 2.0],
        constant: 0.0,
    });
    let (status, _, _) = lp(&m);
    assert_eq!(status, LpStatus::Infeasible);
}

#[test]
fn optimal_point_satisfies_every_row_on_a_fixed_variable_model() {
    // A fixed variable (lower = upper) with three rows, one of them an equality.
    let mut m = Model::new(0);
    m.add_variable(VarBound::continuous(0.0, 6.0));
    m.add_variable(VarBound::continuous(-1.0, -1.0));
    m.add_variable(VarBound::continuous(-3.0, 5.0));
    m.add_variable(VarBound::continuous(-1.0, 3.0));
    m.add_variable(VarBound::continuous(2.0, 4.0));
    let all: Vec<usize> = (0..5).collect();
    m.add_constraint(Constraint::ge(all.clone(), vec![2.0, 1.0, 1.0, -1.0, 1.0], 9.0));
    m.add_constraint(Constraint::equality(all.clone(), vec![-2.0, -3.0, -3.0, 2.0, 3.0], -3.0));
    m.add_constraint(Constraint::equality(all.clone(), vec![0.0, 1.0, 2.0, -3.0, 3.0], 0.0));
    m.set_objective(Objective {
        sense: Sense::Minimize,
        indices: all,
        coeffs: vec![0.0, 3.0, -1.0, 0.0, 0.0],
        constant: 0.0,
    });
    let (status, objective, x) = lp(&m);
    assert_eq!(status, LpStatus::Optimal);
    assert!(violation(&m, &x) < 1e-6, "violation {}", violation(&m, &x));
    assert!((objective + 5.0).abs() < 1e-6, "objective {objective}");
}

#[test]
fn milp_solver_reports_infeasible_for_the_conflicting_pair() {
    let mut m = Model::new(0);
    m.add_variable(VarBound::continuous(-1.0, 4.0));
    m.add_variable(VarBound::continuous(-3.0, 4.0));
    m.add_constraint(Constraint::equality(vec![0, 1], vec![2.0, -2.0], 1.0));
    m.add_constraint(Constraint::equality(vec![0, 1], vec![0.0, 3.0], 1.0));
    m.add_constraint(Constraint::le(vec![0, 1], vec![-3.0, 2.0], -4.0));
    m.set_objective(Objective {
        sense: Sense::Minimize,
        indices: vec![0, 1],
        coeffs: vec![-3.0, 2.0],
        constant: 0.0,
    });
    assert_eq!(MilpSolver::new().solve(&m).unwrap().status, SolverStatus::Infeasible);
}

#[test]
fn root_cuts_keep_the_optimum_when_bounds_are_negative() {
    // Integer variables with negative lower bounds. Gomory cuts were built from
    // columns with the shift sign flipped, so they removed the feasible point
    // x = (-2, 0, 0, -1), and `with_cuts(true)` reported infeasible. Enumeration
    // gives the optimum 0.
    let mut m = Model::new(0);
    m.add_variable(VarBound::integer(-2.0, -1.0));
    m.add_variable(VarBound::integer(-1.0, 2.0));
    m.add_variable(VarBound::integer(0.0, 1.0));
    m.add_variable(VarBound::integer(-1.0, 0.0));
    let all: Vec<usize> = (0..4).collect();
    m.add_constraint(Constraint::equality(all.clone(), vec![0.0, -3.0, -1.0, -1.0], 1.0));
    m.add_constraint(Constraint::ge(all.clone(), vec![-3.0, -2.0, -1.0, 0.0], 5.0));
    m.set_objective(Objective {
        sense: Sense::Maximize,
        indices: all,
        coeffs: vec![1.0, -3.0, 1.0, -3.0],
        constant: -1.0,
    });
    let plain = MilpSolver::new().solve(&m).unwrap();
    let cut = MilpSolver::new().with_cuts(true).solve(&m).unwrap();
    assert_eq!(plain.status, SolverStatus::Optimal);
    assert_eq!(cut.status, SolverStatus::Optimal);
    assert!((plain.objective_value - 0.0).abs() < 1e-9);
    assert!((cut.objective_value - 0.0).abs() < 1e-9, "with cuts: {}", cut.objective_value);
}

/// Duals are the sensitivity of the objective to each right-hand side, in the
/// objective's own sense. Each row kind, an orientation-flipped row, a maximised
/// objective, and a basis with an interior variable are covered.
#[test]
fn duals_are_rhs_sensitivities_for_every_row_kind() {
    let solve = |m: &Model| {
        let lb: Vec<f64> = m.variables.iter().map(|v| v.bound.bound.lower).collect();
        let ub: Vec<f64> = m.variables.iter().map(|v| v.bound.bound.upper).collect();
        solve_lp(m, &lb, &ub, Tolerances::spec_default())
    };
    let one_var = |sense: Sense, coeff: f64, row: Constraint| {
        let mut m = Model::new(0);
        m.add_variable(VarBound::continuous(0.0, 10.0));
        m.add_constraint(row);
        m.set_objective(Objective { sense, indices: vec![0], coeffs: vec![coeff], constant: 0.0 });
        m
    };

    // min x, x >= 1: d obj / d rhs = +1.
    let s = solve(&one_var(Sense::Minimize, 1.0, Constraint::ge(vec![0], vec![1.0], 1.0)));
    assert!((s.dual[0] - 1.0).abs() < 1e-9, "ge: {:?}", s.dual);

    // min -x, x <= 1: d obj / d rhs = -1.
    let s = solve(&one_var(Sense::Minimize, -1.0, Constraint::le(vec![0], vec![1.0], 1.0)));
    assert!((s.dual[0] + 1.0).abs() < 1e-9, "le: {:?}", s.dual);

    // min x, x = 2: d obj / d rhs = +1.
    let s = solve(&one_var(Sense::Minimize, 1.0, Constraint::equality(vec![0], vec![1.0], 2.0)));
    assert!((s.dual[0] - 1.0).abs() < 1e-9, "eq: {:?}", s.dual);

    // Flipped row: min x, -x <= -1 (so x >= 1); obj = -rhs, so the dual is -1.
    let s = solve(&one_var(Sense::Minimize, 1.0, Constraint::le(vec![0], vec![-1.0], -1.0)));
    assert!((s.dual[0] + 1.0).abs() < 1e-9, "flipped: {:?}", s.dual);

    // max x, x <= 3: d obj / d rhs = +1 in the maximisation's own sense.
    let s = solve(&one_var(Sense::Maximize, 1.0, Constraint::le(vec![0], vec![1.0], 3.0)));
    assert!((s.dual[0] - 1.0).abs() < 1e-9, "max: {:?}", s.dual);

    // Interior basis: min -3x0 + 3x1 - 2x2 - x3 with one equality. The only
    // variable strictly inside its box is x0, so its reduced cost is zero and the
    // dual is c0 / a0 = 1.5.
    let mut m = Model::new(0);
    for (lo, hi) in [(-2.0, 3.0), (-1.0, 2.0), (-1.0, 2.0), (1.0, 3.0)] {
        m.add_variable(VarBound::continuous(lo, hi));
    }
    m.add_constraint(Constraint::equality(vec![0, 1, 2, 3], vec![-2.0, -2.0, 2.0, -2.0], 3.0));
    m.set_objective(Objective {
        sense: Sense::Minimize,
        indices: vec![0, 1, 2, 3],
        coeffs: vec![-3.0, 3.0, -2.0, -1.0],
        constant: 0.0,
    });
    let s = solve(&m);
    assert!((s.dual[0] - 1.5).abs() < 1e-9, "interior: {:?}", s.dual);
}
