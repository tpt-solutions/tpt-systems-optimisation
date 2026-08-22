# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Scenario-based stochastic programming: two-stage extensive form
  (`TwoStageProblem`) and multi-stage scenario trees with prefix-merged
  non-anticipativity (`multi_stage_model`).
- Sample average approximation (`SaaSolver`) with statistical lower/upper
  bounds and optimality-gap confidence intervals.
- Value-of-stochastic-solution metrics: VSS and EVPI (`value_metrics`,
  `vss`, `evpi`).
- Chance constraints: scenario/binary-indicator VaR approximation
  (`scenario_chance_model`) and Gaussian deterministic equivalents
  (`gaussian_chance_row`, Acklam inverse-normal CDF).
- Adjustable robust optimisation: Bertsimas–Sim budgeted reformulation
  (`budgeted_reformulation`) and conservative ellipsoidal reformulation
  (`ellipsoid_reformulation`).
- Distributionally robust optimisation: box ambiguity worst case
  (`worst_case_box_expectation`), cutting-plane decision solver
  (`DroCuttingPlane`), and Wasserstein-ball worst case for linear losses
  (`worst_case_linear_wasserstein`).
- Integration test suite validating every framework against hand-computed
  optima (news-vendor RP*/WS/EEV/VSS/EVPI, VaR budgets, Gaussian protection
  levels, Bertsimas–Sim interpolation, Wasserstein Lipschitz margin).