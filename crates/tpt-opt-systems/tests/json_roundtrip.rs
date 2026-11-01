//! JSON import/export through the umbrella crate (feature `serde`).

#![cfg(feature = "serde")]

use tpt_opt_systems::expr::ModelBuilder;
use tpt_opt_systems::json::{model_from_json, model_to_json};

#[test]
fn model_with_infinite_bounds_round_trips() {
    let mut b = ModelBuilder::new();
    let free = b.variable(
        "free",
        tpt_opt_systems::core::VarBound::continuous(f64::NEG_INFINITY, f64::INFINITY),
    );
    let n = b.integer("n", 0.0, 7.0);
    b.constrain(free.le(3.0));
    b.constrain((free + n).ge(-1.0));
    b.minimize(free + 2.0 * n + 1.5);
    let model = b.build().unwrap();

    let text = model_to_json(&model).unwrap();
    let back = model_from_json(&text).unwrap();

    assert_eq!(back.variables[0].bound.bound.lower, f64::NEG_INFINITY);
    assert_eq!(back.variables[0].bound.bound.upper, f64::INFINITY);
    assert_eq!(back.constraints[0].lower, f64::NEG_INFINITY);
    assert_eq!(back.constraints[1].upper, f64::INFINITY);
    assert_eq!(back.objective.constant, 1.5);
    assert_eq!(back.objective.coeffs, model.objective.coeffs);
}

#[test]
fn malformed_json_is_reported_not_panicked() {
    assert!(model_from_json("{ not json").is_err());
}
