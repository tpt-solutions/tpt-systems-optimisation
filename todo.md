# tpt-systems-optimisation — Build Todo

> Tracks bootstrap + full 10-crate build-out for the tpt-systems-optimisation
> workspace, per `spec.txt`. License for every crate: `MIT OR Apache-2.0`.
> Author: TPT Solutions. Unlike `tpt-math` (which stops at `status = "git"`),
> **this workspace targets crates.io release** — every phase includes
> publish-metadata steps, and the final phase gets every crate to
> `cargo publish --dry-run`-clean, in dependency order. Live `cargo publish`
> itself is intentionally **out of scope** for this checklist — it's a
> separate, later, human-triggered action once the dry-run is clean and
> crate names are confirmed reserved.
>
> External solver bindings: only **HiGHS** (MIT-licensed) gets a checklist
> phase, as an optional non-default feature. SCIP/Gurobi/CPLEX are
> proprietary/restrictively licensed and out of step with the permissive-
> license-only policy this workspace inherits from `tpt-math`'s `deny.toml` —
> deferred to a future-work note, not built here.

## Phase 0 — Repo Bootstrap

(one-time, mirrors `tpt-math`'s bootstrap)

- [ ] Create root `Cargo.toml` (`[workspace]`, `resolver = "2"`,
      `[workspace.package]`: `edition = "2021"`, `rust-version = "1.84"`,
      `license = "MIT OR Apache-2.0"`, `authors = ["TPT Solutions"]`,
      `homepage`/`repository = "https://github.com/tpt-solutions/tpt-systems-optimisation"`)
- [ ] Add `rust-toolchain.toml`
- [ ] Add `rustfmt.toml`
- [ ] Add `deny.toml` (mirror `tpt-math`'s: `advisories.yanked = "deny"`,
      `sources.unknown-registry = "deny"`, `sources.unknown-git = "deny"`,
      permissive-license allowlist)
- [ ] Add `.github/workflows/ci.yml` (fmt, clippy, test via cargo-nextest +
      doctests, no_std build via xtask, cargo-deny, feature-powerset via
      cargo-hack for the umbrella crate, bench-smoke compile-only)
- [ ] Add `LICENSE-MIT` and `LICENSE-APACHE`
- [ ] Create empty `crates/` directory
- [ ] Add a Rust `.gitignore` (`/target`, etc.)
- [ ] Write root `README.md` stub — workspace's role bridging `tpt-math`
      (pure math) and Tier 2 domain repos (energy, transportation, process,
      construction, earth, materials, medical, electronics); link to
      `spec.txt`
- [ ] `git init` (local only — no GitHub remote/push, matching `tpt-math`'s
      current stage)
- [ ] Initial commit
- [ ] Sanity check: `cargo build` succeeds on the empty workspace
- [ ] Note: the six upstream deps (`tpt-math-linalg`, `tpt-math-linalg-complex`,
      `tpt-math-optimize-convex`, `tpt-math-optimize-general`, `tpt-math-graph`,
      `tpt-math-numeric`, `tpt-math-prob`) already exist as workspace members
      in the sibling `tpt-math` repo (`d:\Programming\1PRODUCTION\Open
      Source\tpt-math\`) — wire them as `path` deps for local dev; switch to
      version deps once `tpt-math` itself publishes

## Per-Crate Checklist Template

Every phase below repeats this shape. `tpt-opt-systems` (Phase 10) uses the
umbrella variant instead of steps 2-4.

**Standard crate:**
1. Scaffold `crates/<name>/` (Cargo.toml inheriting workspace fields, `lib.rs` stub)
2. Wire dependencies (internal `tpt-opt-*`/`tpt-math-*` + external), `default = ["std"]` with additive `alloc` feature where applicable
3. Implement scope
4. Unit tests + doctests
5. Rustdoc (crate-level + public API)
6. `cargo fmt --check` / `cargo clippy --all-targets --all-features -- -D warnings` clean
7. `cargo deny check` clean
8. no_std target verification (`thumbv6m-none-eabi`) — only for crates the spec marks `no_std` compatible (core; others as feasible)
9. Add `README.md` + `CHANGELOG.md` (Keep-a-Changelog format)
10. Set crates.io metadata: `description`, `keywords`, `categories`, `readme = "README.md"`, `documentation = "https://docs.rs/<name>"`
11. Reserve the crate name on crates.io (confirm availability; placeholder publish or `cargo owner` check if claiming early)
12. `cargo package -p <name> --list` — confirm README/CHANGELOG/LICENSE are included and nothing unwanted leaks in
13. `cargo publish --dry-run -p <name>` clean

**Umbrella crate:**
1. Scaffold `crates/<name>/` (Cargo.toml with Cargo features gating each constituent re-export)
2. Wire optional deps + matching feature flags per constituent crate
3. Re-export each constituent's public API behind its feature
4. Rustdoc documenting the feature matrix
5. `cargo fmt` / `clippy` / `deny` clean across feature combinations
6. `README.md` + `CHANGELOG.md`, crates.io metadata, name reservation, `cargo package --list`, `cargo publish --dry-run` (same as steps 9-13 above)

---

## Phase 1 — tpt-opt-core

*Foundation layer: canonical problem representation + solver interface
contract. no_std with optional alloc. No internal `tpt-opt-*` deps —
depends on `tpt-math-linalg` (CSR/CSC compatibility).*

- [ ] Scaffold `crates/tpt-opt-core/`
- [ ] Wire deps: `tpt-math-linalg`; `default = ["std"]` + `alloc` feature
- [ ] Implement `Model`, `Variable`, `Constraint`, `Objective`, `Solver` traits
- [ ] Implement sparse constraint matrix representations (CSR/CSC) compatible with `tpt-math-linalg`
- [ ] Implement variable bound types (continuous, integer, binary, semi-continuous)
- [ ] Implement solver status enum (`Optimal`, `Infeasible`, `Unbounded`, `TimeLimit`, `NumericalIssue`, `Error`)
- [ ] Implement warm-start interfaces for reusing previous solutions
- [ ] Implement parameter tuning API (time limit, gap tolerance, thread count, verbosity)
- [ ] Implement solution extraction utilities (primal values, dual values, reduced costs, slack variables)
- [ ] Implement structured error types with infeasibility diagnostics
- [ ] Implement numerical tolerance defaults per spec §4 (integrality ε = 1e-6, feasibility δ = 1e-6, optimality gap = 1e-4, pivoting tolerance), all configurable
- [ ] Implement `CustomConstraint` extensibility trait (`evaluate`/`gradient`/`is_violated`) per spec §4
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] no_std+alloc verify (`thumbv6m-none-eabi`)
- [ ] README.md + CHANGELOG.md
- [ ] Crates.io metadata (description/keywords/categories/documentation)
- [ ] Reserve `tpt-opt-core` name on crates.io
- [ ] `cargo package --list` clean
- [ ] `cargo publish --dry-run` clean

## Phase 2 — tpt-opt-milp

*Branch-and-bound/branch-and-cut MILP solver. Depends on: tpt-opt-core,
tpt-math-linalg.*

- [ ] Scaffold `crates/tpt-opt-milp/`
- [ ] Wire deps: `tpt-opt-core`, `tpt-math-linalg`
- [ ] Implement branch-and-bound core with branch-and-cut enhancements
- [ ] Implement cutting-plane generation: Gomory mixed-integer cuts, clique cuts, cover inequalities, MIR cuts, lift-and-project cuts (root node + optional tree nodes)
- [ ] Implement primal heuristics: feasibility pump, rounding heuristics, RINS, local branching
- [ ] Implement node selection strategies: best-bound, best-estimate, depth-first
- [ ] Implement variable branching rules: most fractional, strong branching, pseudo-cost branching
- [ ] Implement special ordered sets (SOS1, SOS2)
- [ ] Implement indicator constraints ("if binary y=1 then linear constraint holds")
- [ ] Implement piecewise linear objectives
- [ ] Implement parallel tree search: work-stealing thread pool, concurrent cut generation, background primal heuristics
- [ ] Implement `.with_seed(...)` deterministic branching/heuristics + `.with_threads(...)`/`.with_parallel_cuts(...)` per spec §4 examples
- [ ] Unit tests + doctests (small hand-crafted MILP examples)
- [ ] Integration test: at least one MIPLIB 2017 benchmark instance solved to optimality
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] README.md + CHANGELOG.md
- [ ] Crates.io metadata
- [ ] Reserve `tpt-opt-milp` name on crates.io
- [ ] `cargo package --list` clean
- [ ] `cargo publish --dry-run` clean

### 2a — Optional HiGHS feature (non-default)

*HiGHS is MIT-licensed and free — the one external-solver binding this pass
includes, gated behind a non-default `highs` feature for benchmarking/
production use per spec §4 "Solver Agnosticism". SCIP/Gurobi/CPLEX
deferred — see Open Risks.*

- [ ] Add optional `highs` feature wiring a HiGHS Rust binding (e.g. `highs`/`highs-sys` crate) as an alternate `Solver` impl
- [ ] Document the added C++ build-toolchain requirement when `highs` is enabled
- [ ] Cross-solver validation test (in-house vs. HiGHS) on shared small MILP instances, feature-gated
- [ ] Confirm `highs`/`highs-sys` license (MIT) passes `cargo deny check` when the feature is enabled
- [ ] `cargo publish --dry-run -p tpt-opt-milp --no-default-features` and `--all-features` both clean

## Phase 3 — tpt-opt-network

*Network flow + graph-based optimization. Depends on: tpt-opt-core,
tpt-math-graph.*

- [ ] Scaffold `crates/tpt-opt-network/`
- [ ] Wire deps: `tpt-opt-core`, `tpt-math-graph`
- [ ] Implement network simplex algorithm for min-cost flow (capacity constraints, multi-commodity)
- [ ] Implement successive shortest path algorithm
- [ ] Implement Hungarian algorithm for assignment/matching problems
- [ ] Implement AC-OPF (polar + rectangular coordinate variants, interior-point methods)
- [ ] Implement DC-OPF (linearized power flow, fast approximation)
- [ ] Implement security-constrained OPF (SC-OPF) with contingency constraints
- [ ] Implement graph preprocessing utilities: cycle detection, bridge identification, biconnected component decomposition, series-parallel reduction
- [ ] Implement dynamic networks (time-varying capacities/costs) with warm-starting between time periods
- [ ] Unit tests + doctests
- [ ] Integration test: at least one Netlib-style or hand-crafted min-cost-flow/OPF benchmark
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] README.md + CHANGELOG.md
- [ ] Crates.io metadata
- [ ] Reserve `tpt-opt-network` name on crates.io
- [ ] `cargo package --list` clean
- [ ] `cargo publish --dry-run` clean

## Phase 4 — tpt-opt-minlp

*Mixed-Integer Nonlinear Programming. Depends on: tpt-opt-core, tpt-opt-milp
(for OA master problems), tpt-math-optimize-convex, tpt-math-optimize-general.*

- [ ] Scaffold `crates/tpt-opt-minlp/`
- [ ] Wire deps: `tpt-opt-core`, `tpt-opt-milp`, `tpt-math-optimize-convex`, `tpt-math-optimize-general`
- [ ] Implement outer-approximation (OA) for convex MINLP (MILP master + NLP subproblems)
- [ ] Implement generalized Benders decomposition (GBD) for complicating variables
- [ ] Implement sequential quadratic programming (SQP) branch-and-bound for non-convex MINLP
- [ ] Implement convex relaxations: McCormick envelopes, alpha-BB techniques
- [ ] Implement indicator constraints with nonlinear consequents
- [ ] Implement logical constraints (AND/OR/XOR on binary variables)
- [ ] Implement complementarity constraints
- [ ] Implement convergence certificates + duality gap tracking
- [ ] Unit tests + doctests
- [ ] Integration test: at least one MINLPLib benchmark instance
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] README.md + CHANGELOG.md
- [ ] Crates.io metadata
- [ ] Reserve `tpt-opt-minlp` name on crates.io
- [ ] `cargo package --list` clean
- [ ] `cargo publish --dry-run` clean

## Phase 5 — tpt-opt-cp

*Constraint programming engine. Depends on: tpt-opt-core.*

- [ ] Scaffold `crates/tpt-opt-cp/`
- [ ] Wire deps: `tpt-opt-core`
- [ ] Implement constraint propagation: AC-3 (binary constraints), AC-4 (fine-grained domain reduction), maintained arc consistency (incremental)
- [ ] Implement global constraint: `alldifferent` (Hall's theorem filtering)
- [ ] Implement global constraint: `cumulative` (resource-constrained scheduling, time-table + energetic reasoning)
- [ ] Implement global constraint: `element` (array indexing)
- [ ] Implement global constraint: `table` (explicit tuple enumeration, GAC support)
- [ ] Implement global constraint: `regular` (automaton-based sequence constraints)
- [ ] Implement global constraint: `circuit` (Hamiltonian cycle)
- [ ] Implement search strategies: first-fail, domain splitting, impact-based, activity-based
- [ ] Implement reification (constraints → boolean variables)
- [ ] Implement conflict-directed backjumping + no-good recording
- [ ] Unit tests + doctests
- [ ] Integration test: at least one CSPLib benchmark instance
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] README.md + CHANGELOG.md
- [ ] Crates.io metadata
- [ ] Reserve `tpt-opt-cp` name on crates.io
- [ ] `cargo package --list` clean
- [ ] `cargo publish --dry-run` clean

## Phase 6 — tpt-opt-heuristic

*Metaheuristic optimization algorithms. Depends on: tpt-opt-core,
tpt-math-prob (for `Rng`/sampling).*

- [ ] Scaffold `crates/tpt-opt-heuristic/`
- [ ] Wire deps: `tpt-opt-core`, `tpt-math-prob`
- [ ] Implement simulated annealing (geometric/adaptive/reheating cooling schedules, configurable neighborhoods)
- [ ] Implement genetic algorithms: crossover (single-point, two-point, uniform, order-based), mutation (bit-flip, swap, inversion, scramble), selection (tournament, roulette, rank)
- [ ] Implement tabu search: adaptive tenure, aspiration criteria, diversification/intensification
- [ ] Implement particle swarm optimization (PSO): inertia weight adaptation, topologies (global, local, Von Neumann)
- [ ] Ensure every heuristic is seedable via an `Rng` parameter for deterministic reproducibility (spec §4)
- [ ] Support custom neighborhood structures via trait objects
- [ ] Implement convergence history tracking
- [ ] Unit tests + doctests (incl. determinism test: same seed → same result)
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] README.md + CHANGELOG.md
- [ ] Crates.io metadata
- [ ] Reserve `tpt-opt-heuristic` name on crates.io
- [ ] `cargo package --list` clean
- [ ] `cargo publish --dry-run` clean

## Phase 7 — tpt-opt-multi

*Multi-objective / Pareto optimization. Depends on: tpt-opt-core,
tpt-opt-heuristic (for NSGA-II's GA machinery, if reused).*

- [ ] Scaffold `crates/tpt-opt-multi/`
- [ ] Wire deps: `tpt-opt-core`, `tpt-opt-heuristic`
- [ ] Implement ε-constraint method (iterative single-objective subproblems)
- [ ] Implement weighted sum + weighted Tchebycheff scalarization with adaptive weight generation
- [ ] Implement NSGA-II (fast non-dominated sorting, crowding distance)
- [ ] Implement NSGA-III style reference-point preference articulation
- [ ] Implement hypervolume calculation (WFG algorithm)
- [ ] Implement Pareto dominance checking + epsilon-indicator
- [ ] Implement objective normalization for disparate scales
- [ ] Implement decision-making utilities: knee point detection, trade-off analysis, solution clustering
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] README.md + CHANGELOG.md
- [ ] Crates.io metadata
- [ ] Reserve `tpt-opt-multi` name on crates.io
- [ ] `cargo package --list` clean
- [ ] `cargo publish --dry-run` clean

## Phase 8 — tpt-opt-robust

*Optimization under uncertainty. Depends on: tpt-opt-core, tpt-opt-milp,
tpt-math-optimize-convex, tpt-math-prob.*

- [ ] Scaffold `crates/tpt-opt-robust/`
- [ ] Wire deps: `tpt-opt-core`, `tpt-opt-milp`, `tpt-math-optimize-convex`, `tpt-math-prob`
- [ ] Implement scenario-based stochastic programming (two-stage and multi-stage, recourse decisions)
- [ ] Implement sample average approximation (SAA) with statistical confidence intervals
- [ ] Implement adjustable robust optimization (ARO): budgeted uncertainty sets (Γ-robustness), ellipsoidal uncertainty sets, tractable LP/SOCP/SDP reformulations
- [ ] Implement chance constraints (scenario approximation + conservative deterministic equivalents)
- [ ] Implement distributionally robust optimization: moment-based + Wasserstein-ball ambiguity sets
- [ ] Implement value of stochastic solution (VSS) and expected value of perfect information (EVPI) calculations
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] README.md + CHANGELOG.md
- [ ] Crates.io metadata
- [ ] Reserve `tpt-opt-robust` name on crates.io
- [ ] `cargo package --list` clean
- [ ] `cargo publish --dry-run` clean

## Phase 9 — tpt-opt-decompose

*Large-scale decomposition methods. Depends on: tpt-opt-core, tpt-opt-milp,
tpt-math-optimize-convex.*

- [ ] Scaffold `crates/tpt-opt-decompose/`
- [ ] Wire deps: `tpt-opt-core`, `tpt-opt-milp`, `tpt-math-optimize-convex`
- [ ] Implement Benders decomposition (master + subproblems on complicating variables)
- [ ] Implement Pareto-optimal cut generation for Benders
- [ ] Implement stabilization techniques (level set, trust region) for oscillation prevention
- [ ] Implement Dantzig-Wolfe decomposition (set partitioning/covering master, independent block subproblems)
- [ ] Implement column generation with pricing subproblems
- [ ] Implement branch-and-price for integer solutions
- [ ] Implement restricted master problem management
- [ ] Implement Lagrangian relaxation: subgradient optimization, bundle methods, surrogate constraints
- [ ] Implement automatic decomposable-structure detection + strategy recommendation
- [ ] Unit tests + doctests
- [ ] Rustdoc
- [ ] `cargo fmt` / `clippy` clean
- [ ] `cargo deny check` clean
- [ ] README.md + CHANGELOG.md
- [ ] Crates.io metadata
- [ ] Reserve `tpt-opt-decompose` name on crates.io
- [ ] `cargo package --list` clean
- [ ] `cargo publish --dry-run` clean

## Phase 10 — tpt-opt-systems (umbrella)

*Feature-gated umbrella re-exporting all solver crates. Flat feature tree
(no nesting) per spec §3.*

- [ ] Scaffold `crates/tpt-opt-systems/`
- [ ] Wire optional deps + one flat feature per solver crate: `milp`, `minlp`, `network`, `cp`, `heuristic`, `multi`, `robust`, `decompose`
- [ ] Confirm no-features build re-exports only `tpt-opt-core`
- [ ] Re-export each constituent crate's public API behind its feature
- [ ] Implement `MilpBuilder`, `NetworkFlowBuilder` convenience constructors
- [ ] Implement unified `OptimizationError` wrapping solver-specific errors with algorithm context
- [ ] Implement format conversion utilities (e.g. network-flow-to-MILP for solvers lacking specialized network algorithms)
- [ ] Rustdoc documenting the full feature matrix
- [ ] `cargo fmt` / `clippy` / `deny` clean across feature combinations
- [ ] README.md + CHANGELOG.md
- [ ] Crates.io metadata
- [ ] Reserve `tpt-opt-systems` name on crates.io
- [ ] `cargo package --list` clean
- [ ] `cargo publish --dry-run` clean (no-default-features and all-features)

---

## Design-Principle Cross-Cutting Checklist

*Spec §4 principles that span every crate — verify once per principle across
the whole workspace rather than duplicating per-phase.*

- [ ] **Solver Agnosticism**: confirm every solver crate's primary solve type implements `tpt-opt-core`'s `Solver<M>` trait (`solve`/`set_parameter`/`warm_start`/`status`/`solution`) with a consistent signature
- [ ] **Reproducibility**: confirm every heuristic/branching/parallel component across `tpt-opt-milp`, `tpt-opt-heuristic`, `tpt-opt-multi` accepts a seed and produces deterministic output for a fixed seed, including under parallel execution (deterministic work distribution)
- [ ] **Numerical Stability**: confirm tolerance defaults (integrality 1e-6, feasibility 1e-6, optimality gap 1e-4, pivoting threshold) are configurable and documented consistently across `tpt-opt-milp`/`tpt-opt-minlp`/`tpt-opt-network`; confirm numerical-issue detection (cycling, stalling, ill-conditioning) surfaces via `SolverStatus::NumericalIssue` rather than silently wrong results
- [ ] **Parallelism**: confirm the work-stealing thread pool pattern is shared/consistent between `tpt-opt-milp`'s tree search and `tpt-opt-network`'s multi-threaded linear algebra; confirm thread-safety for concurrent solves on different models
- [ ] **Extensibility**: confirm `CustomConstraint` (from `tpt-opt-core`) is actually usable end-to-end in at least `tpt-opt-milp` and `tpt-opt-cp` via a test that defines and plugs in a custom constraint

## Testing Strategy (spec §6)

- [ ] Unit tests: confirm each solver component (branching strategy, cut generator, constraint propagator) has isolated tests with small hand-crafted examples — audited across all phases above, not just claimed
- [ ] Integration tests: at least one benchmark instance solved end-to-end per relevant crate — MIPLIB 2017 (MILP), MINLPLib (MINLP), Netlib (LP/network), CSPLib (CP) — see per-phase integration test items above
- [ ] Performance regression tests: track solve time, memory usage, node count on a small fixed benchmark set; wire into CI as a non-blocking report (not a hard gate, to avoid CI flakiness from timing variance)
- [ ] Fuzz testing: random optimization problem generator (varying size/density/constraint types); verify solver invariants (constraints satisfied, objective matches reported value, integrality satisfied)
- [ ] Cross-solver validation: compare in-house MILP results against HiGHS (feature-gated, see Phase 2a) — SCIP/Gurobi comparison explicitly deferred
- [ ] Numerical stability tests: solve problems with varying condition numbers/scaling, confirm robustness or correct `NumericalIssue` reporting
- [ ] Parallel correctness tests: parallel solver output matches sequential solver output (within numerical tolerance) for the same seed

## Tier 2 Consumption Sanity Check (spec §7)

*Verify `tpt-opt-systems` compiles cleanly with exactly the feature set each
named Tier 2 consumer would use — a cargo-hack-style targeted check, not a
full feature-powerset (that's covered by the umbrella's own CI job in Phase
10).*

- [ ] `tpt-energy`: `features = ["milp", "network", "robust"]` builds clean
- [ ] `tpt-transportation`: `features = ["milp", "cp", "multi", "heuristic"]` builds clean
- [ ] `tpt-process`: `features = ["minlp", "decompose", "multi"]` builds clean
- [ ] `tpt-construction`: `features = ["cp", "milp", "multi"]` builds clean
- [ ] `tpt-earth`: `features = ["network", "robust"]` builds clean
- [ ] `tpt-materials`: `features = ["multi", "heuristic"]` builds clean
- [ ] `tpt-medical`: `features = ["milp", "cp"]` builds clean
- [ ] `tpt-electronics`: `features = ["milp", "network"]` builds clean
- [ ] Add a CI job iterating all 8 combinations above (list-driven, not hand-duplicated per-combination CI steps)

## Crates.io Publish-Readiness Phase (final phase)

*Goes further than `tpt-math`'s "Post-Build Hardening" since release is this
workspace's whole point. Stops at dry-run clean — live `cargo publish` is a
separate, later, human-triggered action.*

### CI + tooling

- [ ] Add `xtask` crate (`fmt`/`clippy`/`test`/`deny`/`no-std`/`check` subcommands, mirroring `tpt-math`'s xtask) + `.cargo/config.toml` alias
- [ ] Add root `justfile` with recipes shelling out to `cargo xtask *`
- [ ] Add `examples/` workspace member (unpublished) with a few runnable cross-crate programs (e.g. milp+core, network+milp-via-conversion, heuristic+multi)
- [ ] Add `cargo-hack` feature-powerset CI job for `tpt-opt-systems`
- [ ] Add MSRV policy: pin `rust-version` in `[workspace.package]`, add a CI job building against that exact toolchain
- [ ] Wire `cargo semver-checks` into CI (informational for the 0.1.0 baseline; becomes a real gate starting the first post-0.1.0 change)
- [ ] Add `bench-smoke` CI job (compile-only `cargo bench --no-run`) for crates with `criterion` benches

### Packaging + metadata audit

- [ ] Confirm all 10 crate names (`tpt-opt-core`, `-milp`, `-minlp`, `-network`, `-cp`, `-heuristic`, `-multi`, `-robust`, `-decompose`, `-systems`) are available/reservable on crates.io — record result per name
- [ ] Confirm every crate's `Cargo.toml` has `description`, `keywords` (≤5), `categories` (valid crates.io category slugs), `readme`, `documentation`, `license`, `repository`
- [ ] Add `[package.metadata.docs.rs]` to `tpt-opt-systems` (and any crate with non-default features) so docs.rs builds with `all-features = true`
- [ ] `cargo package -p <crate> --list` audited for every crate — confirm README/CHANGELOG/LICENSE-MIT/LICENSE-APACHE included, no stray files
- [ ] `cargo publish --dry-run -p <crate>` clean for every crate, run in dependency order (core → milp → network → minlp → cp → heuristic → multi → robust → decompose → systems)

### Docs + governance

- [ ] Root `README.md`: full crate map, build order, Tier 2 consumption examples (mirroring spec §7's table), quick-start snippet, link to `spec.txt`
- [ ] Root `SECURITY.md` (no-`unsafe` policy if adopted, `deny.toml` posture, panic/`Result` convention, disclosure contact)
- [ ] Root `CONTRIBUTING.md` (per-crate checklist reference, `deny.toml` license policy, issues-only vs. external-PR workflow — decide and state)
- [ ] Root `CHANGELOG.md` or per-crate-only convention — decide and state explicitly (tpt-math uses per-crate only)

### Verification

- [ ] `cargo test --workspace --all-features` passes
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean
- [ ] `cargo deny check` clean workspace-wide
- [ ] `cargo doc --workspace --no-deps` succeeds with no broken intra-doc links
- [ ] `cargo publish --dry-run` clean for all 10 crates, confirmed in one final pass after any last-minute doc/metadata edits

## Open Risks / Assumptions

- [ ] **HiGHS build dependency**: the optional `highs` feature (Phase 2a) pulls in a C++ build via `highs-sys`-style bindings — document the added build-toolchain requirement in CI and in `tpt-opt-milp`'s README; confirm it doesn't break the no-default-features build for `no_std`/minimal consumers
- [ ] **Crate name availability**: all 10 `tpt-opt-*` names are assumed available on crates.io but not yet verified — resolve in the Publish-Readiness Phase before any reservation attempt
- [ ] **Benchmark corpora size**: MIPLIB 2017 / MINLPLib / Netlib / CSPLib instance files are large external downloads — integration tests must fetch/cache them outside the published crate (e.g. a `tests/fixtures/` dir excluded via `.gitignore`/`exclude` in `Cargo.toml`, populated by a CI step or `xtask` command) so they never bloat the packaged `.crate` file
- [ ] **SCIP/Gurobi/CPLEX deferral**: spec §4 mentions these as pluggable via feature flags; explicitly out of scope for this pass on licensing grounds — revisit only if a future consumer has a commercial license and requests it as an opt-in, clearly-labeled non-default feature
