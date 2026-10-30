# tpt-opt-bayes

Bayesian optimisation for expensive black-box objectives over a box. A
Gaussian-process surrogate with a Matérn 5/2 kernel models the objective, its
hyperparameters are fitted by marginal likelihood, and expected improvement
chooses each next evaluation.

Part of the `tpt-systems-optimisation` workspace. It builds on `tpt-opt-core`
for the L-BFGS hyperparameter fit.

## Quick start

```rust
use tpt_opt_bayes::{minimise, BoConfig};

let result = minimise(
    |x| (x[0] - 0.3).powi(2) + (x[1] + 0.2).powi(2),
    &[(0.0, 1.0), (-1.0, 1.0)],
    40,
    &BoConfig::default(),
)?;
println!("best {:?} = {}", result.x, result.f);
```

For ask/tell control (for example, evaluating points in a separate process),
use `BayesOpt::new`, `ask`, `tell` and `best`.

## Determinism

Every random draw comes from a generator seeded by `BoConfig::seed`. The same
seed and objective give the same evaluation sequence.

## Limitations

- Acquisition is maximised over random candidates and perturbations of the
  incumbent, not by a gradient method. This suits low to moderate dimension.
- The surrogate is an exact GP, so cost grows cubically with the number of
  observations.

## License

MIT OR Apache-2.0 (see LICENSE-MIT / LICENSE-APACHE).
