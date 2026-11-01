//! Load a model from MPS text and solve it with the MILP backend.
//!
//! With no argument the embedded 0/1 knapsack is used. Pass a path to solve a
//! free-format MPS file instead:
//!
//! `cargo run -p tpt-opt-systems --example mps_loading --features milp -- path/to/model.mps`

use tpt_opt_systems::milp::format::{read_mps, solve_parsed};

/// Maximise 5·x1 + 4·x2 + 3·x3 subject to 4·x1 + 3·x2 + 2·x3 ≤ 7 with binary
/// variables (MPS minimises, so the costs are negated). Optimum: x1 = x2 = 1,
/// x3 = 0, value 9 (objective −9).
const KNAPSACK_MPS: &str = "\
NAME          KNAP
ROWS
 N  COST
 L  CAP
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    X1        COST         -5.0            CAP       4.0
    X2        COST         -4.0            CAP       3.0
    X3        COST         -3.0            CAP       2.0
    MARKER                 'MARKER'                 'INTEND'
RHS
    RHS       CAP          7.0
BOUNDS
 UP BND       X1           1
 UP BND       X2           1
 UP BND       X3           1
ENDATA
";

fn main() {
    let text = match std::env::args().nth(1) {
        Some(path) => {
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read `{path}`: {e}"))
        }
        None => KNAPSACK_MPS.to_string(),
    };

    let model = read_mps(&text).expect("valid free-format MPS");
    println!("variables: {}, constraints: {}", model.num_vars, model.num_constraints());

    let sol = solve_parsed(&model).expect("the model solves");
    println!("status: {:?}", sol.status);
    println!("objective: {}", sol.objective_value);
    println!("primal: {:?}", sol.primal);

    if std::env::args().nth(1).is_none() {
        assert!((sol.objective_value + 9.0).abs() < 1e-6);
    }
}
