# tpt-opt-cp

Constraint-programming (CP) engine for the `tpt-systems-optimisation`
workspace: integer domains, propagation to a fixpoint, global constraints,
and first-fail backtracking search.

## Features

- **Domains** — bounded integer domains with value-set operations
  (`domain::Domain`).
- **Propagation** — a fixpoint loop applies each constraint's domain filter
  until no constraint can prune further (`model::fixpoint`).
- **Constraints** — linear/equality relations, plus the globals
  `AllDifferent`, `Cumulative` (renewable-resource scheduling), `Element`
  (array indexing), and `Table` (explicit tuple enumeration).
- **Reification** — wrap any constraint in a boolean variable
  (`constraints::Reified`).
- **Search** — first-fail (smallest-domain) backtracking; find one solution
  (`solver::solve`) or enumerate up to a limit (`solver::solutions`).

## Quick start

```rust
use tpt_opt_cp::{
    constraints::{AllDifferent, Linear},
    model::{CpModel, Relation},
    solver::solve,
};

// 4-queens: one queen per row, distinct columns and diagonals.
let n = 4;
let mut m = CpModel::new();
let cols: Vec<usize> = (0..n).map(|_| m.add_var(0, n - 1)).collect();
m.add_constraint(Box::new(AllDifferent::new(cols.clone())));

for (row, &c) in cols.iter().enumerate() {
    // col - row distinct via an auxiliary variable per diagonal family.
    let d1 = m.add_var(0, 2 * n);
    m.add_constraint(Box::new(Linear::new(
        vec![(c, 1), (d1, -1)],
        Relation::Eq,
        row as i64 - n as i64,
    )));
}

let sol = solve(&m).expect("4-queens is satisfiable");
```

## Status

Part of the [tpt-systems-optimisation](https://github.com/tpt-solutions/tpt-systems-optimisation)
workspace. See the workspace `spec.txt` for the overall design and
`todo.md` for build status.

## License

Licensed under either of MIT or Apache-2.0 at your option.