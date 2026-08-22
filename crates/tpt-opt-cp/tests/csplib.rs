//! CSPLib-style integration benchmark for the CP engine.
//!
//! Solves **CSPLib prob019 — Magic Square** at order 3: place the numbers
//! 1..9 on a 3×3 grid so every row, column and main diagonal sums to the
//! magic constant 15. The complete solution set is exactly the 8
//! rotations/reflections of the Lo Shu square, so the test verifies both
//! feasibility and the exact solution count — a strong end-to-end check of
//! propagation (`AllDifferent` + linear constraints) and CBJ search.

use tpt_opt_cp::constraints::{AllDifferent, Linear};
use tpt_opt_cp::model::{CpModel, Relation};
use tpt_opt_cp::solver::{solutions, solve};

/// Build the order-3 magic-square model; returns `(model, vars)` where
/// `vars[r * 3 + c]` is the cell at row `r`, column `c`.
fn build_magic_square() -> (CpModel, Vec<usize>) {
    let mut m = CpModel::new();
    let cells: Vec<usize> = (0..9).map(|_| m.add_var(1, 9)).collect();

    // All nine numbers used exactly once.
    m.add_constraint(Box::new(AllDifferent::new(cells.clone())));

    // Every row, column and main diagonal sums to 15.
    let rows: [Vec<usize>; 3] =
        [(0, 1, 2), (3, 4, 5), (6, 7, 8)].map(|(a, b, c)| vec![cells[a], cells[b], cells[c]]);
    let cols: [Vec<usize>; 3] =
        [(0, 3, 6), (1, 4, 7), (2, 5, 8)].map(|(a, b, c)| vec![cells[a], cells[b], cells[c]]);
    let diags = vec![vec![cells[0], cells[4], cells[8]], vec![cells[2], cells[4], cells[6]]];
    for line in rows.into_iter().chain(cols).chain(diags) {
        let terms: Vec<(usize, i64)> = line.iter().map(|&v| (v, 1i64)).collect();
        m.add_constraint(Box::new(Linear::new(terms, Relation::Eq, 15)));
    }
    (m, cells)
}

fn is_valid(a: &[usize]) -> bool {
    let lines = [
        [0usize, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ];
    lines.iter().all(|l| l.iter().map(|&i| a[i]).sum::<usize>() == 15)
}

#[test]
fn magic_square_finds_a_valid_solution() {
    let (m, _) = build_magic_square();
    let sol = solve(&m).expect("order-3 magic square is satisfiable");
    assert!(is_valid(&sol.assignment), "solver returned an invalid grid: {:?}", sol.assignment);
}

#[test]
fn magic_square_has_exactly_eight_solutions() {
    let (m, _) = build_magic_square();
    // The full solution set is the 8 symmetries of the Lo Shu square.
    let sols = solutions(&m, 100);
    assert_eq!(sols.len(), 8, "expected exactly 8 magic squares, got {}", sols.len());
    for s in &sols {
        assert!(is_valid(&s.assignment));
    }
    // All solutions distinct.
    for i in 0..sols.len() {
        for j in (i + 1)..sols.len() {
            assert_ne!(sols[i].assignment, sols[j].assignment);
        }
    }
}
