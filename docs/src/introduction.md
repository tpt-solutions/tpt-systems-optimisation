# Introduction

`tpt-systems-optimisation` is a pure-Rust optimisation suite. It covers linear
and mixed-integer programming, network flows, constraint programming,
metaheuristics, multi-objective optimisation, robust and stochastic models,
decomposition, conic programming, nonlinear least squares, factor-graph SLAM,
and Bayesian optimisation.

Everything is available through one umbrella crate, `tpt-opt-systems`, with a
flat feature per solver family. Each solver family is also published as its own
crate (`tpt-opt-milp`, `tpt-opt-network`, and so on) for users who want a
smaller dependency tree.

## Where to go next

- New to the crate? Start with [Getting started](./getting-started.md).
- Building a model? The [Modelling cookbook](./modelling-cookbook.md) has one
  worked example per problem shape.
- Choosing a solver? The [Solver guide](./solver-guide.md) explains the options
  and their trade-offs.
- Working from files? The [Command-line tool](./cli.md) reads, converts,
  presolves, diagnoses, and solves MPS and CPLEX-LP models.

The API reference is generated from rustdoc and is linked from the project
README.
