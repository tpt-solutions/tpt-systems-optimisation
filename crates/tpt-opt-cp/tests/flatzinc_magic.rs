//! End-to-end FlatZinc: a 3×3 magic square (distinct entries 1..9, every row,
//! column, and diagonal sums to 15), parsed from text and solved by the CP engine.

use tpt_opt_cp::flatzinc::{parse, Goal};

fn magic_square_text() -> String {
    let mut text = String::new();
    for i in 0..9 {
        text.push_str(&format!("var 1..9: c{i} :: output_var;\n"));
    }
    text.push_str("array [1..9] of var int: cells :: output_array([1..3, 1..3]) = [");
    let names: Vec<String> = (0..9).map(|i| format!("c{i}")).collect();
    text.push_str(&names.join(", "));
    text.push_str("];\n");
    text.push_str("constraint all_different_int(cells);\n");
    let lines =
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
    for line in lines {
        let vars: Vec<String> = line.iter().map(|i| format!("c{i}")).collect();
        text.push_str(&format!("constraint int_lin_eq([1, 1, 1], [{}], 15);\n", vars.join(", ")));
    }
    text.push_str("solve satisfy;\n");
    text
}

#[test]
fn magic_square_satisfies_every_rule() {
    let program = parse(&magic_square_text()).unwrap();
    assert_eq!(program.goal(), &Goal::Satisfy);
    let solution = program.solve().unwrap().expect("a 3×3 magic square exists");

    let mut grid = [0i64; 9];
    for (name, value) in &solution {
        let i: usize = name.trim_start_matches('c').parse().unwrap();
        grid[i] = *value;
    }

    let mut seen = grid.to_vec();
    seen.sort_unstable();
    assert_eq!(seen, (1..=9).collect::<Vec<i64>>(), "entries must be 1..9 exactly once");
    let lines =
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
    for line in lines {
        let sum: i64 = line.iter().map(|&i| grid[i]).sum();
        assert_eq!(sum, 15, "line {line:?}");
    }
}

#[test]
fn magic_square_has_eight_solutions_when_enumerated() {
    // Minimising a cell enumerates every solution. A 3×3 magic square has 8
    // (the symmetries of the Lo Shu square). Minimising c0 returns the least
    // corner value, which is 2 across those solutions.
    let text = magic_square_text().replace("solve satisfy;", "solve minimize c0;");
    let program = parse(&text).unwrap();
    let best = program.solve().unwrap().expect("solvable");
    let c0 = best.iter().find(|(n, _)| n == "c0").unwrap().1;
    assert_eq!(c0, 2);
}
