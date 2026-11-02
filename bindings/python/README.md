# tpt-opt (Python)

Python bindings for the tpt-systems-optimisation suite:

- `tpt_opt.Problem`: a linear or mixed-integer program, solved with the bundled
  branch-and-cut MILP solver (`tpt-opt-milp`). It can be built in code or read
  from MPS or CPLEX-LP text.
- `tpt_opt.simulated_annealing`: minimises a Python function over a box with
  the simulated-annealing heuristic (`tpt-opt-heuristic`).

The package is not on PyPI yet. Build it from this directory.

## Build and install

Requires Rust 1.85 and Python 3.9 or later.

```text
pip install maturin
maturin develop --release          # install into the current virtualenv
# or
maturin build --release            # wheel in target/wheels/
```

## Use

```python
import tpt_opt

p = tpt_opt.Problem()
x = p.add_variable(kind="binary")
y = p.add_variable(kind="binary")
z = p.add_variable(kind="binary")
p.add_constraint([(x, 4.0), (y, 3.0), (z, 2.0)], upper=7.0)
p.set_objective([(x, 5.0), (y, 4.0), (z, 3.0)], sense="maximize")

sol = p.solve(time_limit=5.0, threads=2, seed=1)
print(sol.status, sol.objective, sol.primal)   # optimal 9.0 [1.0, 1.0, 0.0]

lp_text = p.to_lp()                            # also to_mps()
again = tpt_opt.Problem.from_lp(lp_text)       # also from_mps()

best_x, best_value = tpt_opt.simulated_annealing(
    lambda x: (x[0] - 0.3) ** 2 + (x[1] + 0.2) ** 2,
    [(-1.0, 1.0), (-1.0, 1.0)],
    seed=7,
    iterations=5000,
)
```

Errors in the model (bad indices, unknown kinds, unparsable text) raise
`ValueError`. An exception raised by the function given to
`simulated_annealing` is re-raised after the search.

## Tests

```text
python -m unittest discover -s tests
```

## Scope

`tpt_opt.LinearMultiObjective` binds the linear multi-objective tools: weighted
sums, the ε-constraint method, the payoff table, and NSGA-II. Objectives are
linear over one box-bounded decision vector. Nonlinear objectives and the
other `multi` methods are not exposed yet.

## License

MIT OR Apache-2.0.
