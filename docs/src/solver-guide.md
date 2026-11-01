# Solver guide

The table in the project README (section "Which solver should I use?") maps
each problem shape to a starting solver. This page explains the trade-offs.

## Mixed-integer linear programs

`MilpSolver` is a branch-and-bound and branch-and-cut solver. Useful knobs:

- `with_cuts(true)` enables the root cut suite (Gomory, lift-and-project,
  clique, cover, and MIR cuts).
- `with_threads(n)` runs a deterministic parallel tree search. The result is
  identical for any thread count.
- `with_seed(s)` fixes the randomised heuristics and branching.
- `with_time_limit(d)` returns the best incumbent when time runs out.
- `with_progress_callback(cb)` reports incumbent and bound updates, and can
  abort the search.

With the `highs` feature, `HighsSolver` offers the same contract using the
HiGHS MIT-licensed binding. It needs a C++ toolchain at build time.

## Nonlinear and global problems

- `minlp::oa` (outer approximation) and `minlp::gbd` (generalised Benders) are
  for convex mixed-integer nonlinear problems.
- `minlp::spatial` is a global solver for bilinear programs over finite boxes.
  It is not a general global NLP solver.
- `nls` solves least-squares problems with autodiff Jacobians and robust
  losses. Use `factorgraph` for pose graphs and bundle adjustment.

## Uncertainty

- `robust` provides two-stage and multi-stage stochastic programs, SAA
  confidence intervals, VSS and EVPI, and robust counterparts. Gaussian chance
  constraints are available both as an LP approximation and as an exact SOCP.

## Metaheuristics and black boxes

- `heuristic` covers annealing, genetic algorithms, tabu search, and particle
  swarm. Every solver is seedable and deterministic per seed.
- `bayes` suits expensive black-box functions evaluated a few dozen times.

## Choosing between the options

| Situation | Choice |
|-----------|--------|
| Need a proof of optimality | MILP, MINLP, or CP |
| Need a good answer fast on a large instance | heuristic, or MILP with a time limit |
| Want several trade-off solutions | `multi` |
| Structure you can exploit (blocks, many columns) | `decompose` |

## Limitations to know about

- The bundled LP engine is dense. Large sparse LPs are better handed to HiGHS.
- Benchmark corpora such as MIPLIB are fetched on demand with
  `cargo xtask fetch-fixtures` and are not bundled.
