//! JSON import and export for canonical [`Model`]s (feature `serde`).
//!
//! Infinite bounds are written as `null` and read back as infinities, so
//! unbounded variables and one-sided rows round-trip exactly. Parsed models
//! are validated before they are returned.

use tpt_opt_core::{Model, OptError};

/// Serialise `model` as pretty-printed JSON.
///
/// ```
/// use tpt_opt_systems::expr::ModelBuilder;
/// use tpt_opt_systems::json::{model_from_json, model_to_json};
///
/// let mut b = ModelBuilder::new();
/// let x = b.continuous("x", 0.0, 4.0);
/// b.maximize(x);
/// let model = b.build().unwrap();
///
/// let text = model_to_json(&model).unwrap();
/// let back = model_from_json(&text).unwrap();
/// assert_eq!(back.objective.coeffs, model.objective.coeffs);
/// ```
pub fn model_to_json(model: &Model) -> Result<String, OptError> {
    serde_json::to_string_pretty(model)
        .map_err(|e| OptError::invalid_model(format!("cannot serialise model: {e}")))
}

/// Parse a model from JSON and validate it.
pub fn model_from_json(text: &str) -> Result<Model, OptError> {
    let model: Model = serde_json::from_str(text)
        .map_err(|e| OptError::invalid_model(format!("invalid model JSON: {e}")))?;
    model.validate()?;
    Ok(model)
}
