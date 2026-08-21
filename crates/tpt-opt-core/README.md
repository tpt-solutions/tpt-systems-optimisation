# tpt-opt-core

Foundation layer of the `tpt-systems-optimisation` workspace: the canonical
optimisation problem representation and the solver-agnosticism contract every
solver crate implements.

`no_std` compatible (optional `alloc`, enabled by the default `std` feature).

## What it provides

- **Model** — [`Model`], [`Variable`], [`Constraint`], [`Objective`],
  [`Sense`]: a linear canonical form every solver crate understands.
- **Bounds** — [`VarBound`], [`VarType`], [`Bound`]: continuous, integer,
  binary, and semi-continuous variables over (possibly infinite) intervals.
- **Tolerances** — [`Tolerances`]: spec §4 defaults (integrality `1e-6`,
  feasibility `1e-6`, optimality gap `1e-4`, pivoting `1e-9`), all overridable.
- **Solver contract** — [`Solver`], [`SolveParameters`], [`Solution`],
  [`SolverStatus`], [`WarmStart`], [`Verbosity`]: the consistent
  `solve` / `set_parameter` / `warm_start` / `status` / `solution` signature.
- **Errors** — [`OptError`], [`InfeasibilityReport`]: structured failures with
  infeasibility diagnostics.
- **Sparse matrices** — [`model_to_csr`], [`model_to_csc`], [`ConstraintMatrix`]:
  CSR/CSC assembly compatible with `tpt-math-linalg`.
- **Extensibility** — [`CustomConstraint`]: plug user-defined constraints into
  any solver.

## Example

```rust
use tpt_opt_core::{
    model::{Constraint, Model, Objective},
    solver::{SolveParameters, Solver, SolverStatus},
    bounds::VarBound,
};

// A trivial model; real solvers live in the sibling tpt-opt-* crates.
let mut m = Model::new(2);
m.set_objective(Objective::minimize(vec![0, 1], vec![1.0, 1.0]));
m.add_constraint(Constraint::ge(vec![0, 1], vec![1.0, 1.0], 1.0));
m.variables[0].bound = VarBound::continuous(0.0, f64::INFINITY);
assert!(m.validate().is_ok());
let _ = SolveParameters::defaults().with_seed(42);
```

## License

Licensed under either of MIT or Apache-2.0 at your option.
