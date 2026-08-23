//! Constraint programming through the umbrella crate: the classic n-queens
//! puzzle modelled with `AllDifferent` plus linear diagonal rows, solved by
//! propagation + conflict-directed backjumping — then exhaustive solution
//! counting on a smaller board to show enumeration agrees with the known
//! count.
//!
//! Run with: cargo run --manifest-path examples/Cargo.toml --example cp_nqueens

use tpt_opt_systems::cp::{
    constraints::{AllDifferent, Linear},
    model::{CpModel, Relation},
    solver::{solutions, solve},
};

/// Build the n-queens model: one variable per row holding its column.
fn queens(n: usize) -> (CpModel, Vec<usize>) {
    let mut m = CpModel::new();
    let cols: Vec<usize> = (0..n).map(|_| m.add_var(0, n - 1)).collect();
    // No two queens share a column…
    m.add_constraint(Box::new(AllDifferent::new(cols.clone())));
    // …nor a diagonal. d1 = col − row (offset by n to stay non-negative),
    // d2 = col + row; both sets must be distinct as well.
    let mut diag1 = Vec::new();
    let mut diag2 = Vec::new();
    for (row, &c) in cols.iter().enumerate() {
        let d1 = m.add_var(0, 2 * n);
        m.add_constraint(Box::new(Linear::new(
            vec![(c, 1), (d1, -1)],
            Relation::Eq,
            (row as i64) - (n as i64),
        )));
        let d2 = m.add_var(0, 2 * n);
        m.add_constraint(Box::new(Linear::new(
            vec![(c, 1), (d2, -1)],
            Relation::Eq,
            -(row as i64),
        )));
        diag1.push(d1);
        diag2.push(d2);
    }
    m.add_constraint(Box::new(AllDifferent::new(diag1)));
    m.add_constraint(Box::new(AllDifferent::new(diag2)));
    (m, cols)
}

fn main() {
    // --- solve 8-queens and print the board ----------------------------------
    let n = 8usize;
    let (m, cols) = queens(n);
    let sol = solve(&m).expect("8-queens has solutions");
    let placed: Vec<usize> = cols.iter().map(|&c| sol.assignment[c]).collect();
    println!("8-queens solution (column per row): {placed:?}");
    for r in 0..n {
        let mut line = String::new();
        for c in 0..n {
            line.push(if placed[r] == c { 'Q' } else { '.' });
        }
        println!("  {line}");
    }
    // Verify every constraint directly on the printed answer.
    for i in 0..n {
        for j in (i + 1)..n {
            assert_ne!(placed[i], placed[j], "same column");
            assert_ne!(
                (placed[i] as i64 - placed[j] as i64).abs(),
                (j - i) as i64,
                "same diagonal"
            );
        }
    }

    // --- exhaustive counting --------------------------------------------------
    // The 6-queens board has exactly 4 distinct solutions; enumerating all of
    // them exercises the search's completeness (CBJ must not skip branches).
    let (m6, cols6) = queens(6);
    let all = solutions(&m6, 100);
    println!("6-queens solution count: {} (known: 4)", all.len());
    assert_eq!(all.len(), 4);
    // Every enumerated solution must itself be valid.
    for s in &all {
        let p: Vec<usize> = cols6.iter().map(|&c| s.assignment[c]).collect();
        for i in 0..6 {
            for j in (i + 1)..6 {
                assert_ne!(p[i], p[j]);
                assert_ne!((p[i] as i64 - p[j] as i64).abs(), (j - i) as i64);
            }
        }
    }
}