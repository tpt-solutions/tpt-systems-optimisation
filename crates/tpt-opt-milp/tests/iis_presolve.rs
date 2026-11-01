//! IIS minimality and presolve soundness, checked against the MILP solver.

use tpt_opt_core::expr::ModelBuilder;
use tpt_opt_core::model::Model;
use tpt_opt_core::{Solver, SolverStatus};
use tpt_opt_milp::{find_iis, presolve_model, MilpSolver};

/// Copy of `model` keeping only the constraints at `keep`.
fn only(model: &Model, keep: &[usize]) -> Model {
    let mut m = model.clone();
    m.constraints = keep.iter().map(|&i| model.constraints[i].clone()).collect();
    m
}

fn status(model: &Model) -> SolverStatus {
    MilpSolver::new().solve(model).unwrap().status
}

#[test]
fn iis_is_infeasible_and_irreducible() {
    let mut b = ModelBuilder::new();
    let x = b.continuous("x", 0.0, 10.0);
    let y = b.continuous("y", 0.0, 10.0);
    let z = b.continuous("z", 0.0, 10.0);
    b.constrain((x + y).ge(8.0)); // 0: needs x + y ≥ 8
    b.constrain(x.le(2.0)); // 1
    b.constrain(y.le(3.0)); // 2  — with 1, x + y ≤ 5 < 8
    b.constrain(z.ge(0.0)); // 3: harmless
    b.constrain((x + z).le(20.0)); // 4: harmless
    b.minimize(x + y + z);
    let model = b.build().unwrap();

    let iis = find_iis(&model).unwrap();
    assert_eq!(iis.constraints, vec![0, 1, 2]);

    let all = iis.constraints.clone();
    assert_eq!(status(&only(&model, &all)), SolverStatus::Infeasible);
    for drop in 0..all.len() {
        let mut rest = all.clone();
        rest.remove(drop);
        assert_ne!(
            status(&only(&model, &rest)),
            SolverStatus::Infeasible,
            "removing {drop} should restore feasibility"
        );
    }
}

#[test]
fn iis_of_a_feasible_model_is_an_error() {
    let mut b = ModelBuilder::new();
    let x = b.continuous("x", 0.0, 10.0);
    b.constrain(x.ge(1.0));
    b.minimize(x);
    assert!(find_iis(&b.build().unwrap()).is_err());
}

#[test]
fn presolve_preserves_the_optimum() {
    let mut b = ModelBuilder::new();
    let x = b.integer("x", 0.0, 10.0);
    let y = b.integer("y", 0.0, 10.0);
    let z = b.continuous("z", 0.0, 10.0);
    b.constrain((2.0 * x).le(9.0)); // singleton: x ≤ 4
    b.constrain((x + y).le(7.5));
    b.constrain((x - z).ge(-1.0));
    b.constrain((0.0 * z).le(5.0)); // empty row
    b.maximize(3.0 * x + 2.0 * y + z);
    let model = b.build().unwrap();

    let (reduced, report) = presolve_model(&model);
    assert!(!report.infeasible);
    assert_eq!(report.removed_rows, 2);
    assert_eq!(reduced.num_constraints(), 2);

    let before = MilpSolver::new().solve(&model).unwrap();
    let after = MilpSolver::new().solve(&reduced).unwrap();
    assert!((before.objective_value - after.objective_value).abs() < 1e-6);
}

#[test]
fn presolve_flags_infeasible_singleton_pair() {
    let mut b = ModelBuilder::new();
    let x = b.integer("x", 0.0, 10.0);
    b.constrain((2.0 * x).ge(3.0)); // x ≥ 1.5 → x ≥ 2
    b.constrain((2.0 * x).le(3.0)); // x ≤ 1.5 → x ≤ 1
    b.minimize(x);
    let (_, report) = presolve_model(&b.build().unwrap());
    assert!(report.infeasible);
}
