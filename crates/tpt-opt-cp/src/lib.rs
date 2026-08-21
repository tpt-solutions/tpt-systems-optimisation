//! Constraint-programming engine: arc-consistency-style propagation, global
//! constraints and backtracking search.

pub mod constraints;
pub mod domain;
pub mod model;
pub mod solver;

#[cfg(test)]
mod tests {
    use crate::constraints::{AllDifferent, Cumulative, Element, Linear, Task};
    use crate::model::{CpModel, Relation};
    use crate::solver::{solve, solutions};

    #[test]
    fn n_queens_4() {
        // Place 4 queens so no two share a row, column, or diagonal.
        let n = 4usize;
        let mut m = CpModel::new();
        let cols: Vec<usize> = (0..n).map(|_| m.add_var(0, n - 1)).collect();
        m.add_constraint(Box::new(AllDifferent::new(cols.clone())));
        // Diagonals: col + row distinct, col - row distinct.
        let mut diag1 = Vec::new();
        let mut diag2 = Vec::new();
        for (row, &c) in cols.iter().enumerate() {
            let d1 = m.add_var(0, 2 * n);
            m.add_constraint(Box::new(Linear::new(
                vec![(c, 1), (d1, -1)],
                Relation::Eq,
                row as i64,
            )));
            let d2 = m.add_var(0, 2 * n);
            m.add_constraint(Box::new(Linear::new(
                vec![(c, 1), (d2, -1)],
                Relation::Eq,
                (row as i64) - (n as i64),
            )));
            diag1.push(d1);
            diag2.push(d2);
        }
        m.add_constraint(Box::new(AllDifferent::new(diag1)));
        m.add_constraint(Box::new(AllDifferent::new(diag2)));

        let sol = solve(&m).expect("4-queens has a solution");
        // Verify no two queens share a diagonal.
        for i in 0..n {
            for j in (i + 1)..n {
                let ci = sol.assignment[cols[i]] as i64;
                let cj = sol.assignment[cols[j]] as i64;
                assert_ne!((ci - cj).abs(), (j as i64 - i as i64).abs());
            }
        }
    }

    #[test]
    fn cumulative_two_tasks() {
        // Two tasks of demand 3 each, capacity 3: they cannot overlap.
        let mut m = CpModel::new();
        let s0 = m.add_var(0, 5);
        let s1 = m.add_var(0, 5);
        let tasks = vec![
            Task {
                start: s0,
                duration: 2,
                demand: 3,
            },
            Task {
                start: s1,
                duration: 2,
                demand: 3,
            },
        ];
        m.add_constraint(Box::new(Cumulative::new(tasks, 3)));
        let sol = solve(&m).expect("feasible");
        let a = sol.assignment[s0];
        let b = sol.assignment[s1];
        let overlap = (a.max(b)) < (a + 2).min(b + 2);
        assert!(!overlap, "tasks must not overlap at capacity 3");
    }

    #[test]
    fn element_lookup() {
        let arr = vec![10, 20, 30];
        let mut m = CpModel::new();
        let idx = m.add_var(0, 2);
        let val = m.add_var(0, 30);
        m.add_constraint(Box::new(Element::new(arr.clone(), idx, val)));
        let sol = solve(&m).expect("feasible");
        assert_eq!(arr[sol.assignment[idx]], sol.assignment[val]);
    }

    #[test]
    fn counts_solutions() {
        // x,y in {0,1}, x + y = 1 -> exactly two solutions.
        let mut m = CpModel::new();
        let x = m.add_var(0, 1);
        let y = m.add_var(0, 1);
        m.add_constraint(Box::new(Linear::new(vec![(x, 1), (y, 1)], Relation::Eq, 1)));
        let sols = solutions(&m, 10);
        assert_eq!(sols.len(), 2);
    }
}
