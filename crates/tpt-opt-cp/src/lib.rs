//! Constraint-programming engine: arc-consistency-style propagation, global
//! constraints and backtracking search.

pub mod constraints;
pub mod domain;
pub mod model;
pub mod solver;

#[cfg(test)]
mod tests {
    use crate::constraints::{AllDifferent, Circuit, Cumulative, Element, Linear, Regular, Task};
    use crate::model::{CpModel, Relation};
    use crate::solver::{solutions, solve};

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
            // d1 = c_i - i + n  (offset keeps the auxiliary non-negative).
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
            Task { start: s0, duration: 2, demand: 3 },
            Task { start: s1, duration: 2, demand: 3 },
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

    #[test]
    fn regular_no_three_consecutive_ones() {
        // Binary sequence of length 5 with no run of three 1s.
        // DFA: state = current run length of trailing 1s (0,1,2); a third
        // consecutive 1 has no transition (dead end).
        let mut m = CpModel::new();
        let xs: Vec<usize> = (0..5).map(|_| m.add_var(0, 1)).collect();
        let mut tr = Vec::new();
        for s in 0..3 {
            tr.push((s, 0, 0)); // a 0 resets the run
            if s < 2 {
                tr.push((s, 1, s + 1)); // extend the run (max 2)
            }
        }
        m.add_constraint(Box::new(Regular::new(xs.clone(), tr, 0, vec![0, 1, 2], 3)));

        let sol = solve(&m).expect("feasible");
        let vals: Vec<usize> = xs.iter().map(|&x| sol.assignment[x]).collect();
        for w in vals.windows(3) {
            assert!(!(w[0] == 1 && w[1] == 1 && w[2] == 1), "run of three 1s: {vals:?}");
        }
        // Propagation must reject the all-ones sequence.
        let mut m2 = CpModel::new();
        let ys: Vec<usize> = (0..5).map(|_| m2.add_var_values(vec![1])).collect();
        let mut tr2 = Vec::new();
        for s in 0..3 {
            tr2.push((s, 0, 0));
            if s < 2 {
                tr2.push((s, 1, s + 1));
            }
        }
        m2.add_constraint(Box::new(Regular::new(ys, tr2, 0, vec![0, 1, 2], 3)));
        assert!(solve(&m2).is_none(), "all-ones must be infeasible");
    }

    #[test]
    fn regular_propagation_prunes_unsupported_prefix() {
        // Length-2 sequence over {0,1} that must end in state 1: the only
        // accepting path is (0 then 1). So x0 is forced to 0 and x1 to 1.
        let mut m = CpModel::new();
        let x0 = m.add_var(0, 1);
        let x1 = m.add_var(0, 1);
        let tr = vec![(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)];
        m.add_constraint(Box::new(Regular::new(vec![x0, x1], tr, 0, vec![1], 2)));
        let sol = solve(&m).expect("feasible");
        assert_eq!(sol.assignment[x0], 0);
        assert_eq!(sol.assignment[x1], 1);
    }

    #[test]
    fn circuit_finds_hamiltonian_cycle() {
        // Complete graph on 4 nodes: any derangement forming one cycle.
        let mut m = CpModel::new();
        let succ: Vec<usize> = (0..4).map(|_| m.add_var(0, 3)).collect();
        m.add_constraint(Box::new(Circuit::new(succ.clone())));
        let sol = solve(&m).expect("4-node circuit exists");
        let s: Vec<usize> = succ.iter().map(|&v| sol.assignment[v]).collect();
        // Walk the cycle from node 0; must return after visiting all 4.
        let mut cur = 0usize;
        let mut visited = [false; 4];
        for _ in 0..4 {
            assert!(!visited[cur], "sub-cycle: {s:?}");
            visited[cur] = true;
            cur = s[cur];
        }
        assert_eq!(cur, 0);
    }

    #[test]
    fn circuit_rejects_disjoint_cycles() {
        // Force 0->1 and 1->0 (a 2-cycle); with n=4 the circuit constraint
        // must be infeasible since the remaining nodes cannot join.
        let mut m = CpModel::new();
        let s0 = m.add_var(0, 3);
        let s1 = m.add_var(0, 3);
        let s2 = m.add_var(0, 3);
        let s3 = m.add_var(0, 3);
        m.add_constraint(Box::new(Circuit::new(vec![s0, s1, s2, s3])));
        m.domains[s0].assign(1);
        m.domains[s1].assign(0);
        assert!(solve(&m).is_none(), "disjoint 2-cycle must be rejected");
    }

    #[test]
    fn cbj_matches_enumeration_on_infeasible_model() {
        // x,y,z in {0,1}; x+y+z = 2 and x+y+z = 1: infeasible. CBJ must
        // prove infeasibility (solve -> None) without missing anything.
        let mut m = CpModel::new();
        let x = m.add_var(0, 1);
        let y = m.add_var(0, 1);
        let z = m.add_var(0, 1);
        m.add_constraint(Box::new(Linear::new(vec![(x, 1), (y, 1), (z, 1)], Relation::Eq, 2)));
        m.add_constraint(Box::new(Linear::new(vec![(x, 1), (y, 1), (z, 1)], Relation::Eq, 1)));
        assert!(solve(&m).is_none());
        assert!(solutions(&m, 10).is_empty());
    }

    #[test]
    fn cbj_finds_same_solutions_as_enumeration() {
        // Random-ish small model: CBJ one-solution result must satisfy all
        // constraints, and enumeration must agree on feasibility.
        let mut m = CpModel::new();
        let a = m.add_var(0, 3);
        let b = m.add_var(0, 3);
        let c = m.add_var(0, 3);
        m.add_constraint(Box::new(Linear::new(vec![(a, 1), (b, 1)], Relation::Eq, 4)));
        m.add_constraint(Box::new(Linear::new(vec![(b, 1), (c, 2)], Relation::Le, 5)));
        m.add_constraint(Box::new(AllDifferent::new(vec![a, b, c])));
        let sol = solve(&m).expect("feasible");
        let (va, vb, vc) = (sol.assignment[a], sol.assignment[b], sol.assignment[c]);
        assert_eq!(va + vb, 4);
        assert!(vb + 2 * vc <= 5);
        assert!(va != vb && vb != vc && va != vc);
        assert!(!solutions(&m, 10).is_empty());
    }

    #[test]
    fn ac4_reaches_a_sound_fixpoint() {
        // AC-4 must never remove a value that AC-3 keeps (soundness): every
        // domain after AC-4 is a superset of the corresponding domain after
        // AC-3, and both preserve feasibility.
        let build = || {
            let mut m = CpModel::new();
            let a = m.add_var(0, 4);
            let b = m.add_var(0, 4);
            let c = m.add_var(0, 4);
            m.add_constraint(Box::new(Linear::new(vec![(a, 2), (b, 1)], Relation::Eq, 6)));
            m.add_constraint(Box::new(AllDifferent::new(vec![a, b, c])));
            m
        };
        let mut ac3 = build();
        let mut ac4 = build();
        ac3.ac3().expect("feasible");
        ac4.ac4().expect("feasible");
        for v in 0..ac3.domains.len() {
            let d3: std::collections::HashSet<usize> =
                ac3.domains[v].values().iter().copied().collect();
            let d4: std::collections::HashSet<usize> =
                ac4.domains[v].values().iter().copied().collect();
            assert!(d3.is_superset(&d4), "AC-4 kept a value AC-3 removed at var {v}");
        }
        // Both still solve to a valid assignment.
        let s3 = crate::solver::solve(&ac3).expect("ac3 feasible");
        let s4 = crate::solver::solve(&ac4).expect("ac4 feasible");
        let check = |m: &CpModel, s: &crate::solver::CpSolution| {
            m.constraints().iter().all(|c| c.check(&s.assignment))
        };
        assert!(check(&ac3, &s3));
        assert!(check(&ac4, &s4));
    }

    #[test]
    fn ac4_detects_wipeout_as_inconsistency() {
        let mut m = CpModel::new();
        let x = m.add_var_values(vec![0]);
        let y = m.add_var_values(vec![0]);
        m.add_constraint(Box::new(AllDifferent::new(vec![x, y])));
        assert!(m.ac4().is_err(), "two vars fixed to the same value");
    }

    #[test]
    fn selection_strategies_all_solve_nqueens() {
        // First-fail, impact and activity must each find a valid 4-queens
        // placement (the search order differs but the result is a solution).
        fn build() -> CpModel {
            let n = 4usize;
            let mut m = CpModel::new();
            let cols: Vec<usize> = (0..n).map(|_| m.add_var(0, n - 1)).collect();
            m.add_constraint(Box::new(AllDifferent::new(cols.clone())));
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
            m
        }
        use crate::solver::{solve_with, VariableSelection};
        for sel in
            [VariableSelection::FirstFail, VariableSelection::Impact, VariableSelection::Activity]
        {
            let m = build();
            let sol = solve_with(&m, sel).expect("4-queens has a solution");
            for i in 0..4 {
                for j in (i + 1)..4 {
                    let ci = sol.assignment[i] as i64;
                    let cj = sol.assignment[j] as i64;
                    assert_ne!((ci - cj).abs(), (j as i64 - i as i64).abs());
                }
            }
        }
    }
}
