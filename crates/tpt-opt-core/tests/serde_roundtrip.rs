//! Round-trip tests for the optional `serde` feature.
//!
//! JSON is used as the representative self-describing format. Infinite
//! variable bounds and one-sided row bounds are encoded as `null` on the wire
//! (see the `bounds`/`model` module docs), so models containing them still
//! round-trip losslessly.

#![cfg(feature = "serde")]

use tpt_opt_core::bounds::{Bound, VarBound, VarType};
use tpt_opt_core::error::{InfeasibilityReport, OptError};
use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::solver::{Solution, SolveParameters, SolverStatus, Verbosity, WarmStart};
use tpt_opt_core::tolerance::Tolerances;

fn sample_model() -> Model {
    // max 3x + 2y + 5  s.t. x + y <= 4; x,y integer in [0, 4]
    let mut m = Model::with_name(2, "serde-sample");
    m.variables[0].bound = VarBound::integer(0.0, 4.0);
    m.variables[1].bound = VarBound::integer(0.0, 4.0);
    m.add_constraint(Constraint::le(vec![0, 1], vec![1.0, 1.0], 4.0));
    m.set_objective(Objective {
        sense: Sense::Maximize,
        indices: vec![0, 1],
        coeffs: vec![3.0, 2.0],
        constant: 5.0,
    });
    m
}

#[test]
fn model_round_trips_through_json() {
    let m = sample_model();
    let json = serde_json::to_string(&m).expect("serialize model");
    let back: Model = serde_json::from_str(&json).expect("deserialize model");
    assert_eq!(m, back);
}

#[test]
fn solution_round_trips_through_json() {
    let sol = Solution::new(vec![1.0, 2.5, -0.25], 12.75, SolverStatus::Optimal)
        .with_dual(vec![1.0, -2.0])
        .with_reduced_costs(vec![0.0, 0.5, 1.5])
        .with_slacks(vec![0.5])
        .with_iterations(42)
        .with_solve_time(0.125);
    let json = serde_json::to_string(&sol).expect("serialize solution");
    let back: Solution = serde_json::from_str(&json).expect("deserialize solution");
    assert_eq!(sol, back);
}

#[test]
fn parameters_tolerances_and_warm_start_round_trip() {
    let params = SolveParameters::defaults()
        .with_time_limit(30.0)
        .with_threads(4)
        .with_seed(7)
        .with_gap(1e-6, 1e-5)
        .with_verbosity(Verbosity::Verbose)
        .with_tolerances(Tolerances::spec_default().with_pivoting(1e-10));
    let json = serde_json::to_string(&params).unwrap();
    assert_eq!(params, serde_json::from_str(&json).unwrap());

    let warm =
        WarmStart { primal: Some(vec![1.0, 2.0]), dual: None, status: Some(SolverStatus::Optimal) };
    let json = serde_json::to_string(&warm).unwrap();
    assert_eq!(warm, serde_json::from_str::<WarmStart>(&json).unwrap());
}

#[test]
fn errors_and_reports_round_trip() {
    let err = OptError::Infeasible(
        InfeasibilityReport::new("rows 0 and 1 conflict")
            .with_violated(0)
            .with_violated(1)
            .with_conflict(3),
    );
    let json = serde_json::to_string(&err).unwrap();
    assert_eq!(err, serde_json::from_str::<OptError>(&json).unwrap());

    let err = OptError::NumericalIssue("pivot below threshold".into());
    let json = serde_json::to_string(&err).unwrap();
    assert_eq!(err, serde_json::from_str::<OptError>(&json).unwrap());
}

#[test]
fn bound_kinds_survive_a_round_trip() {
    for vb in [
        VarBound::continuous(-5.0, 5.0),
        VarBound::integer(0.0, 10.0),
        VarBound::binary(),
        VarBound::semi_continuous(2.0, 8.0),
    ] {
        let json = serde_json::to_string(&vb).unwrap();
        let back: VarBound = serde_json::from_str(&json).unwrap();
        assert_eq!(vb, back);
    }
    assert_eq!(serde_json::from_str::<VarType>("\"Binary\"").unwrap(), VarType::Binary);
    let b = Bound::boxed(-1.5, 2.5);
    let json = serde_json::to_string(&b).unwrap();
    assert_eq!(b, serde_json::from_str::<Bound>(&json).unwrap());
}

#[test]
fn infinite_bounds_round_trip_through_json() {
    // Free continuous variable + a <= row (lower = -inf) + a >= row
    // (upper = +inf): all three exercise the null encoding.
    let mut m = Model::new(1);
    m.variables[0].bound = VarBound::continuous(Bound::UNBOUNDED_LOWER, Bound::UNBOUNDED_UPPER);
    m.add_constraint(Constraint::le(vec![0], vec![1.0], 10.0));
    m.add_constraint(Constraint::ge(vec![0], vec![1.0], -10.0));
    m.set_objective(Objective::minimize(vec![0], vec![1.0]));

    let json = serde_json::to_string(&m).expect("serialize model with infinities");
    assert!(json.contains("null"), "open ends must be encoded as null");
    let back: Model = serde_json::from_str(&json).expect("deserialize model");
    assert_eq!(m, back);
    assert_eq!(back.variables[0].bound.bound.lower, f64::NEG_INFINITY);
    assert_eq!(back.constraints[0].lower, f64::NEG_INFINITY);
    assert_eq!(back.constraints[1].upper, f64::INFINITY);
}

#[test]
fn deserialised_model_still_solves_consistently() {
    // The point of serialisation: a model that round-trips must behave
    // identically when handed back to a solver.
    let m = sample_model();
    let json = serde_json::to_string(&m).unwrap();
    let back: Model = serde_json::from_str(&json).unwrap();
    assert_eq!(back.validate(), Ok(()));
    assert_eq!(back.objective.eval(&[2.0, 2.0]), 15.0);
    assert!(back.constraints[0].is_satisfied(&[2.0, 2.0], 1e-9));
}
