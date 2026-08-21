//! Integration tests for the public `tpt-opt-core` API.
//!
//! These exercise the canonical problem representation, the solver contract
//! shims, tolerances, bounds, and sparse-matrix assembly. The `Rng`-free tests
//! here run under the default `std` feature; the crate stays `no_std` otherwise.

use tpt_opt_core::{
    bounds::{Bound, VarBound, VarType},
    custom::CustomConstraint,
    error::{InfeasibilityReport, OptError},
    matrix::{model_to_csr, ConstraintMatrix},
    model::{Constraint, Model, Objective},
    solver::{Solution, SolveParameters, Solver, SolverStatus, WarmStart},
    tolerance::Tolerances,
};

#[test]
fn model_build_and_validate() {
    let mut m = Model::new(3);
    // Minimise x0 + 2*x1 subject to x0 + x1 >= 1, x0 - x2 <= 3.
    m.set_objective(Objective::minimize(vec![0, 1], vec![1.0, 2.0]));
    let c0 = Constraint::ge(vec![0, 1], vec![1.0, 1.0], 1.0);
    let c1 = Constraint::le(vec![0, 2], vec![1.0, -1.0], 3.0);
    assert_eq!(m.add_constraint(c0), 0);
    assert!(m.add_constraint(c1) > 0);
    m.variables[0].bound = VarBound::continuous(0.0, 10.0);
    assert!(m.validate().is_ok());
}

#[test]
fn constraint_eval_and_slack() {
    let c = Constraint::equality(vec![0, 1], vec![2.0, -1.0], 4.0);
    let x = [2.0, 0.0]; // 2*2 - 0 = 4 -> satisfied
    assert!(c.is_satisfied(&x, 1e-9));
    assert!((c.slack(&x)).abs() < 1e-9);
    let x2 = [0.0, 0.0];
    assert!(!c.is_satisfied(&x2, 1e-9));
    assert!(c.slack(&x2) < 0.0);
}

#[test]
fn bound_feasibility() {
    let b = VarBound::binary();
    assert!(b.is_integral());
    assert!(b.feasible(0.0, 1e-9));
    assert!(b.feasible(1.0, 1e-9));
    assert!(!b.feasible(0.5, 1e-9));
    assert!(!b.feasible(2.0, 1e-9));

    let semi = VarBound::semi_continuous(2.0, 5.0);
    assert!(semi.feasible(0.0, 1e-9));
    assert!(semi.feasible(3.0, 1e-9));
    assert!(!semi.feasible(1.0, 1e-9));

    assert_eq!(VarBound::continuous(-1.0, 1.0).bound, Bound::boxed(-1.0, 1.0));
    assert_eq!(VarType::Integer as u8, VarType::Integer as u8);
    assert_eq!(Bound::free().lower, f64::NEG_INFINITY);
}

#[test]
fn tolerances_defaults_and_override() {
    let t = Tolerances::spec_default();
    assert_eq!(t.integrality, 1e-6);
    assert_eq!(t.feasibility, 1e-6);
    assert_eq!(t.optimality_gap, 1e-4);
    assert_eq!(t.pivoting, 1e-9);
    let t2 = t.with_integrality(1e-4).with_feasibility(1e-5);
    assert_eq!(t2.integrality, 1e-4);
    assert_eq!(t2.feasibility, 1e-5);
    assert_eq!(Tolerances::default(), Tolerances::spec_default());
}

#[test]
fn solve_parameters_builder() {
    let p = SolveParameters::defaults()
        .with_time_limit(60.0)
        .with_threads(4)
        .with_seed(42)
        .with_gap(1e-3, 1e-3);
    assert_eq!(p.time_limit, Some(60.0));
    assert_eq!(p.threads, 4);
    assert_eq!(p.seed, Some(42));
    assert_eq!(p.absolute_gap, 1e-3);
}

#[test]
fn solver_status_has_solution() {
    assert!(SolverStatus::Optimal.has_solution());
    assert!(SolverStatus::TimeLimit.has_solution());
    assert!(!SolverStatus::Infeasible.has_solution());
    assert!(!SolverStatus::NumericalIssue.has_solution());
}

#[test]
fn solution_builder_attaches_metadata() {
    let s = Solution::new(vec![1.0, 2.0], 5.0, SolverStatus::Optimal)
        .with_dual(vec![0.5])
        .with_reduced_costs(vec![-1.0, 0.0])
        .with_slacks(vec![0.0])
        .with_iterations(12)
        .with_solve_time(0.01);
    assert_eq!(s.primal_value(1), Some(2.0));
    assert_eq!(s.dual, vec![0.5]);
    assert_eq!(s.iterations, Some(12));
    assert_eq!(s.solve_time, Some(0.01));
}

#[test]
fn warm_start_helpers() {
    assert!(WarmStart::empty().is_empty());
    assert!(!WarmStart::primal(vec![1.0, 2.0]).is_empty());
    assert!(!WarmStart::dual(vec![0.0]).is_empty());
}

#[test]
fn error_display_and_diagnostics() {
    let e = OptError::infeasible(
        InfeasibilityReport::new("conflict").with_violated(3).with_conflict(1),
    );
    assert!(e.is_infeasible());
    let msg = format!("{e}");
    assert!(msg.contains("conflict"));
    assert!(msg.contains("3"));

    let u = OptError::Unbounded;
    assert!(u.is_unbounded());
    assert_eq!(format!("{u}"), "model is unbounded");
}

// A trivial recorder solver used to validate the `Solver` trait contract.
struct RecorderSolver {
    last: Option<Solution>,
    status: SolverStatus,
}

impl RecorderSolver {
    fn new() -> Self {
        Self { last: None, status: SolverStatus::Optimal }
    }
}

impl Solver<Model> for RecorderSolver {
    fn solve(&mut self, model: &Model) -> Result<Solution, OptError> {
        let primal = vec![0.0; model.num_vars];
        let obj = model.objective.eval(&primal);
        let s = Solution::new(primal, obj, self.status);
        self.last = Some(s.clone());
        Ok(s)
    }
    fn set_parameter(&mut self, _p: &SolveParameters) -> Result<(), OptError> {
        Ok(())
    }
    fn warm_start(&mut self, _w: WarmStart) -> Result<(), OptError> {
        Ok(())
    }
    fn status(&self) -> SolverStatus {
        self.status
    }
    fn solution(&self) -> Option<Solution> {
        self.last.clone()
    }
}

#[test]
fn solver_trait_contract() {
    let mut solver = RecorderSolver::new();
    let m = Model::with_name(2, "test");
    let sol = solver.solve(&m).unwrap();
    assert_eq!(solver.status(), SolverStatus::Optimal);
    assert_eq!(solver.solution(), Some(sol.clone()));
    assert_eq!(sol.primal_value(0), Some(0.0));
}

/// A custom constraint x0*x1 - 1 = 0 (satisfied at (1,1)).
struct ProductOne;
impl CustomConstraint for ProductOne {
    fn arity(&self) -> usize {
        2
    }
    fn evaluate(&self, x: &[f64]) -> f64 {
        x[0] * x[1] - 1.0
    }
    fn gradient(&self, x: &[f64], grad: &mut [f64]) {
        grad[0] = x[1];
        grad[1] = x[0];
    }
}

#[test]
fn custom_constraint_eval() {
    let c = ProductOne;
    assert!((c.evaluate(&[1.0, 1.0])).abs() < 1e-12);
    assert!(c.is_violated(&[2.0, 2.0], 1e-9));
    let mut g = [0.0; 2];
    c.gradient(&[2.0, 3.0], &mut g);
    assert_eq!(g, [3.0, 2.0]);
}

#[test]
fn model_to_csr_assembly() {
    let mut m = Model::new(3);
    m.add_constraint(Constraint::equality(vec![0, 2], vec![1.0, 4.0], 2.0));
    m.add_constraint(Constraint::le(vec![1], vec![2.0], 1.0));
    let csr = model_to_csr(&m);
    assert_eq!(csr.nrows(), 2);
    assert_eq!(csr.ncols(), 3);
    // Row 0: (0,1.0), (2,4.0). Value at (0,2) should be 4.0.
    let cm = ConstraintMatrix::from_model(&m);
    assert_eq!(cm.csr.nrows(), 2);
    assert_eq!(cm.csc.ncols(), 3);
}
