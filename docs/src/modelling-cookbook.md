# Modelling cookbook

Each recipe is a complete program fragment. Recipes that use a solver assume
the `milp` feature.

## Knapsack (binary variables)

```rust
use tpt_opt_systems::prelude::*;

let mut b = ModelBuilder::new();
let items = [(5.0, 4.0), (4.0, 3.0), (3.0, 2.0)]; // (value, weight)
let x: Vec<Var> = (0..items.len()).map(|i| b.binary(format!("x{i}"))).collect();

let weight = items.iter().zip(&x).fold(LinExpr::zero(), |acc, (&(_, w), &v)| acc + w * v);
b.constrain(weight.le(7.0));
let value = items.iter().zip(&x).fold(LinExpr::zero(), |acc, (&(v, _), &xi)| acc + v * xi);
b.maximize(value);

let sol = solve_auto(&b.build()?)?;
```

## Relations and bounds

- `expr.le(r)`, `expr.ge(r)`, and `expr.eq(r)` build a relation. A constant on
  the expression moves onto the right-hand side automatically.
- `b.continuous`, `b.integer`, and `b.binary` create variables with bounds.
- `b.variable(name, VarBound)` accepts any bound kind, including semi-continuous.

## Fixed-charge and big-M logic

A fixed-charge cost `0.5·y + 40·z`, where any positive `y` requires `z = 1`.
Demand of 10 units forces the charge to be paid, so the optimum is `45`:

```rust
let mut b = ModelBuilder::new();
let y = b.continuous("y", 0.0, 100.0);
let z = b.binary("z");
b.constrain((y - 100.0 * z).le(0.0)); // y ≤ 100·z forces z = 1 when y > 0
b.constrain(y.ge(10.0)); // demand
b.minimize(0.5 * y + 40.0 * z);
```

## Reading and writing files

```rust
use tpt_opt_systems::milp::format::{read_mps, write_lp};

let model = read_mps(&std::fs::read_to_string("model.mps")?)?;
std::fs::write("model.lp", write_lp(&model))?;
```

Objective constants survive both formats. Variable and row names are not kept,
because the canonical model does not store them.

## Diagnosing infeasibility

When a model is infeasible, find the constraints that conflict:

```rust
use tpt_opt_systems::milp::find_iis;

let iis = find_iis(&model)?;
println!("conflicting rows: {:?}", iis.constraints);
```

The result is irreducible: removing any one of those rows makes the rest
feasible. Variable bounds are held fixed during the search.

## Presolve

`presolve_model` removes empty and singleton rows, tightens bounds, and keeps
variable indices unchanged, so a solution of the reduced model is a solution of
the original:

```rust
use tpt_opt_systems::milp::presolve_model;

let (reduced, report) = presolve_model(&model);
if report.infeasible {
    // no solution exists
}
```

## Saving and loading JSON

With the `serde` feature:

```rust
use tpt_opt_systems::json::{model_from_json, model_to_json};

let text = model_to_json(&model)?;
let back = model_from_json(&text)?;
```
