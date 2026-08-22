# tpt-opt-multi

Multi-objective / Pareto optimisation for the `tpt-systems-optimisation`
workspace.

## Features

- **Pareto analysis** — dominance checking, non-dominated front extraction,
  and the additive epsilon indicator (`dominance`).
- **Hypervolume** — exact computation in 2-D and the WFG algorithm for N-D
  (`hypervolume`).
- **Normalisation** — scale objectives to comparable ranges from sample data
  (`ObjectiveNormalizer`).
- **NSGA-II** — a self-contained, seedable evolutionary multi-objective
  solver with fast non-dominated sorting and crowding distance (`Nsga2`).
- **Scalarisation** — weighted-sum and ε-constraint methods compiled to
  single-objective MILPs solved with `tpt-opt-milp` (`scalarize`).

## Quick start

```rust
use tpt_opt_multi::{dominates, hypervolume, pareto_front};

let pts = vec![vec![1.0, 5.0], vec![2.0, 3.0], vec![5.0, 1.0], vec![4.0, 4.0]];
// Minimisation semantics: (2,3) beats (4,4).
assert!(dominates(&[2.0, 3.0], &[4.0, 4.0]));
let front = pareto_front(&pts); // indices of non-dominated points
let front_pts: Vec<Vec<f64>> = front.iter().map(|&i| pts[i].clone()).collect();
let hv = hypervolume(&front_pts, &[6.0, 6.0]);
```

## Status

Part of the [tpt-systems-optimisation](https://github.com/tpt-solutions/tpt-systems-optimisation)
workspace. See the workspace `spec.txt` for the overall design and
`todo.md` for build status.

## License

Licensed under either of MIT or Apache-2.0 at your option.