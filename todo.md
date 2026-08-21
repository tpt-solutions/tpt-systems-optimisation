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

- [x] Create root `Cargo.toml` (`[workspace]`, `resolver = "2"`,
      `[workspace.package]`: `edition = "2021"`, `rust-version = "1.84"`,
      `license = "MIT OR Apache-2.0"`, `authors = ["TPT Solutions"]`,
      `homepage`/`repository = "https://github.com/tpt-solutions/tpt-systems-optimisation"`)
- [x] Add `rust-toolchain.toml`
- [x] Add `rustfmt.toml`
- [x] Add `deny.toml` (mirror `tpt-math`'s: `advisories.yanked = "deny"`,
      `sources.unknown-registry = "deny"`, `sources.unknown-git = "deny"`,
      permissive-license allowlist) — **FIXED**: removed the unrecognized
      `[advisories.osv]` table (it is not valid for the installed cargo-deny
      0.20.2); `cargo deny check` now passes clean (advisories/bans/licenses/
      sources ok). See Open Risks.
- [x] Add `.github/workflows/ci.yml` (fmt, clippy, test via cargo-nextest +
      doctests, no_std build via xtask, cargo-deny, feature-powerset via
      cargo-hack for the umbrella crate, bench-smoke compile-only)
- [x] Add `LICENSE-MIT` and `LICENSE-APACHE`
- [x] Create empty `crates/` directory
- [x] Add a Rust `.gitignore` (`/target`, etc.)
- [x] Write root `README.md` stub — workspace's role bridging `tpt-math`
      (pure math) and Tier 2 domain repos (energy, transportation, process,
      construction, earth, materials, medical, electronics); link to
      `spec.txt`
- [x] `git init` (local only — no GitHub remote/push, matching `tpt-math`'s
      current stage)
- [x] Initial commit
- [x] Sanity check: `cargo build` succeeded on the empty workspace at
      bootstrap time — note the workspace no longer builds clean now that
      Phase 3 (`tpt-opt-network`) has been scaffolded; see Phase 3 status
- [x] Note: the six upstream deps (`tpt-math-linalg`, `tpt-math-linalg-complex`,
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

**Status: implementation essentially complete and clippy-clean; publish-readiness steps not started.**

- [x] Scaffold `crates/tpt-opt-core/`
- [x] Wire deps: `tpt-math-linalg` (+ `tpt-math-numeric`); `default = ["std"]` + `alloc` feature
- [x] Implement `Model`, `Variable`, `Constraint`, `Objective`, `Solver` traits (`model.rs`, `solver.rs`)
- [x] Implement sparse constraint matrix representations (CSR/CSC) compatible with `tpt-math-linalg` (`matrix.rs`)
- [x] Implement variable bound types (continuous, integer, binary, semi-continuous) (`bounds.rs`)
- [x] Implement solver status enum (`Optimal`, `Infeasible`, `Unbounded`, `TimeLimit`, `NumericalIssue`, `Error`)
- [x] Implement warm-start interfaces for reusing previous solutions (`WarmStart` in `solver.rs`)
- [x] Implement parameter tuning API (time limit, gap tolerance, thread count, verbosity) (`SolveParameters`)
- [x] Implement solution extraction utilities (primal values, dual values, reduced costs, slack variables) (`Solution`)
- [x] Implement structured error types with infeasibility diagnostics (`error.rs`)
- [x] Implement numerical tolerance defaults per spec §4 (integrality ε = 1e-6, feasibility δ = 1e-6, optimality gap = 1e-4, pivoting tolerance), all configurable (`tolerance.rs`)
- [x] Implement `CustomConstraint` extensibility trait (`evaluate`/`gradient`/`is_violated`) per spec §4 (`custom.rs`)
- [x] Unit tests + doctests (`tests/core_api.rs`; doctests embedded in doc comments)
- [x] Rustdoc (crate-level + public API doc comments present throughout)
- [x] `cargo fmt` / `clippy` clean — verified: `cargo clippy -p tpt-opt-core --all-targets --all-features -- -D warnings` passes with zero warnings
- [ ] `cargo deny check` clean — blocked workspace-wide by the broken `deny.toml` (see Phase 0 note / Open Risks)
- [ ] no_std+alloc verify (`thumbv6m-none-eabi`) — not verified (target not installed locally; needs `rustup target add thumbv6m-none-eabi` + CI wiring)
- [x] README.md + CHANGELOG.md
- [x] Crates.io metadata (description/keywords/categories/documentation) — present in `Cargo.toml`
- [ ] Reserve `tpt-opt-core` name on crates.io
- [ ] `cargo package --list` clean
- [ ] `cargo publish --dry-run` clean

## Phase 2 — tpt-opt-milp

*Branch-and-bound/branch-and-cut MILP solver. Depends on: tpt-opt-core,
tpt-math-linalg.*

**Status: core branch-and-bound solver works and lib builds, but several
checklist sub-items are not yet implemented, clippy is not clean, and one
test file fails to compile. `README.md`/`CHANGELOG.md` are missing.**

- [x] Scaffold `crates/tpt-opt-milp/`
- [x] Wire deps: `tpt-opt-core`, `tpt-math-linalg`
- [x] Implement branch-and-bound core (`milp.rs`) with a root-node Gomory cut pass (`cuts.rs`) — "branch-and-cut" only in the limited sense of root cuts, not general tree-wide cut management
- [ ] Implement cutting-plane generation: Gomory mixed-integer cuts, clique cuts, cover inequalities, MIR cuts, lift-and-project cuts (root node + optional tree nodes) — **only Gomory cuts implemented**; clique/cover/MIR/lift-and-project not started
- [ ] Implement primal heuristics: feasibility pump, rounding heuristics, RINS, local branching — feasibility pump and rounding implemented (`try_rounding`/`try_feasibility_pump` in `milp.rs`); RINS and local branching not started
- [ ] Implement node selection strategies: best-bound, best-estimate, depth-first — only depth-first (LIFO stack) diving implemented; best-bound and best-estimate not started
- [ ] Implement variable branching rules: most fractional, strong branching, pseudo-cost branching — most-fractional and a best-effort pseudo-cost estimate implemented; strong branching not started
- [ ] Implement special ordered sets (SOS1, SOS2) — not started
- [ ] Implement indicator constraints ("if binary y=1 then linear constraint holds") — not started
- [ ] Implement piecewise linear objectives — not started
- [ ] Implement parallel tree search: work-stealing thread pool, concurrent cut generation, background primal heuristics — not started (`with_threads` exists but is a best-effort no-op stub per its own doc comment; no thread pool)
- [x] Implement `.with_seed(...)` deterministic branching/heuristics + `.with_threads(...)` — `with_parallel_cuts(...)` not implemented (no matches found in source)
- [x] Unit tests + doctests (small hand-crafted MILP examples) (`tests/milp_api.rs`) — note `tests/debug_lp.rs` currently fails to compile (`solve_lp` signature mismatch: passes `&Tolerances` where `Tolerances` is expected)
- [ ] Integration test: at least one MIPLIB 2017 benchmark instance solved to optimality — not present
- [x] Rustdoc — doc comments present throughout
- [ ] `cargo fmt` / `clippy` clean — **not clean**: `cargo clippy -p tpt-opt-milp --all-targets` currently reports 16 warnings (incl. a `clippy::question_mark` lint) and the `debug_lp.rs` test fails to build
- [ ] `cargo deny check` clean — blocked workspace-wide by the broken `deny.toml` (see Open Risks)
- [x] README.md + CHANGELOG.md — added (`crates/tpt-opt-milp/README.md`, `CHANGELOG.md`); `description` added to `Cargo.toml` and `readme` now points at the per-crate file
- [x] Crates.io metadata — present (`description`, `keywords`, `categories`, `readme`, `documentation`, `repository`, `license`, `authors`)
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

**Status: COMPILES and clippy/fmt/test-clean.** Previously BROKEN: `lib.rs`
declared `pub mod opf;` / `pub mod dynamic;` whose source files were missing,
and `lp.rs`/`min_cost_flow.rs` had borrow errors. Both missing modules were
implemented (`src/opf.rs`, `src/dynamic.rs`) and the borrow errors were fixed;
the crate now builds and its 7 tests pass. OPF coverage: DC-OPF (LP via the
in-crate two-phase simplex), AC-OPF (polar NLP via `tpt_math_optimize_general`
augmented-Lagrangian solver), and SC-OPF (base-case DC-OPF + N-1 contingency
re-solves). Dynamic networks solve each period's min-cost flow with a
warm-start hint. The crate was also added as a dependency on
`tpt-math-optimize-general` (required by `ac_opf`).

- [x] Scaffold `crates/tpt-opt-network/`
- [x] Wire deps: `tpt-opt-core`, `tpt-math-graph`, `tpt-math-optimize-general` (added for AC-OPF)
- [x] Implement network simplex algorithm for min-cost flow (capacity constraints, multi-commodity) (`min_cost_flow.rs`, `network_simplex`)
- [x] Implement successive shortest path algorithm (`min_cost_flow.rs`, primary method per its own doc comment)
- [x] Implement Hungarian algorithm for assignment/matching problems (`assignment.rs`, `hungarian`)
- [x] Implement AC-OPF (polar coordinates, augmented-Lagrangian NLP) — `src/opf.rs` (`ac_opf`)
- [x] Implement DC-OPF (linearised power flow, solved as an LP) — `src/opf.rs` (`dc_opf`)
- [x] Implement security-constrained OPF (SC-OPF) with N-1 contingency constraints — `src/opf.rs` (`sc_opf`)
- [x] Implement graph preprocessing utilities: cycle detection, bridge identification, biconnected component decomposition (`graph_preprocess.rs`) — series-parallel reduction not confirmed present, needs a closer look
- [x] Implement dynamic networks (time-varying supplies) with warm-starting between periods — `src/dynamic.rs` (`DynamicNetwork`)
- [ ] Unit tests + doctests — `cargo test -p tpt-opt-network` passes (7 tests); doctests not yet added
- [ ] Integration test: at least one Netlib-style or hand-crafted min-cost-flow/OPF benchmark — not present
- [x] Rustdoc — doc comments present throughout (including the new `opf`/`dynamic` modules)
- [x] `cargo fmt` / `clippy` clean — verified: `cargo clippy -p tpt-opt-network --all-targets --all-features -- -D warnings` passes with zero warnings
- [x] `cargo deny check` clean — verified (workspace-wide, now that `deny.toml` is fixed)
- [x] README.md + CHANGELOG.md — added (`crates/tpt-opt-network/README.md`, `CHANGELOG.md`); `description` added to `Cargo.toml` and `readme` now points at the per-crate file
- [x] Crates.io metadata (description/keywords/categories/documentation) — present in `Cargo.toml`
- [ ] Reserve `tpt-opt-network` name on crates.io
- [ ] `cargo package --list` clean
- [ ] `cargo publish --dry-run` clean

## Phase 4 — tpt-opt-minlp

*Mixed-Integer Nonlinear Programming. Depends on: tpt-opt-core, tpt-opt-milp
(for OA master problems), tpt-math-optimize-convex, tpt-math-optimize-general.*

**Status: scaffolded only.** `src/lib.rs` is a 4-line doc-comment stub with
no implementation; `Cargo.toml` dependencies are already wired.

- [x] Scaffold `crates/tpt-opt-minlp/` (empty stub only)
- [x] Wire deps: `tpt-opt-core`, `tpt-opt-milp`, `tpt-math-optimize-convex`, `tpt-math-optimize-general`
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

**Status: scaffolded only.** `src/lib.rs` is a 4-line doc-comment stub with
no implementation; `Cargo.toml` dependency is already wired.

- [x] Scaffold `crates/tpt-opt-cp/` (empty stub only)
- [x] Wire deps: `tpt-opt-core`
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

**Status: implementation complete and clippy-clean, the most finished crate
alongside `tpt-opt-core`; publish-readiness steps not started.**

- [x] Scaffold `crates/tpt-opt-heuristic/`
- [x] Wire deps: `tpt-opt-core`, `tpt-math-prob`
- [x] Implement simulated annealing (geometric/adaptive/reheating cooling schedules, configurable neighborhoods) (`annealing.rs`)
- [x] Implement genetic algorithms: crossover (single-point, two-point, uniform, order-based), mutation (bit-flip, swap, inversion, scramble), selection (tournament, roulette, rank) (`ga.rs`) — all variants confirmed present
- [x] Implement tabu search: adaptive tenure, aspiration criteria, diversification/intensification (`tabu.rs`)
- [x] Implement particle swarm optimization (PSO): inertia weight adaptation, topologies (global, local, Von Neumann) (`pso.rs`) — confirmed `Global`/`Ring`/`VonNeumann` topologies
- [x] Ensure every heuristic is seedable via an `Rng` parameter for deterministic reproducibility (spec §4) (`rng.rs`, `.with_seed(...)` on all four solvers)
- [x] Support custom neighborhood structures via trait objects (`neighborhood.rs`)
- [x] Implement convergence history tracking (`history.rs`)
- [x] Unit tests + doctests (incl. determinism test: same seed → same result) (`tests/heuristics.rs::determinism_same_seed_all_heuristics`)
- [x] Rustdoc
- [x] `cargo fmt` / `clippy` clean — verified: `cargo clippy -p tpt-opt-heuristic --all-targets --all-features -- -D warnings` passes with zero warnings
- [ ] `cargo deny check` clean — blocked workspace-wide by the broken `deny.toml` (see Open Risks)
- [x] README.md + CHANGELOG.md
- [ ] Crates.io metadata — not yet spot-checked against `Cargo.toml`
- [ ] Reserve `tpt-opt-heuristic` name on crates.io
- [ ] `cargo package --list` clean
- [ ] `cargo publish --dry-run` clean

## Phase 7 — tpt-opt-multi

*Multi-objective / Pareto optimization. Depends on: tpt-opt-core,
tpt-opt-heuristic (for NSGA-II's GA machinery, if reused).*

**Status: scaffolded only.** `src/lib.rs` is a 4-line doc-comment stub with
no implementation; `Cargo.toml` dependencies are already wired.

- [x] Scaffold `crates/tpt-opt-multi/` (empty stub only)
- [x] Wire deps: `tpt-opt-core`, `tpt-opt-heuristic`
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

**Status: scaffolded only.** `src/lib.rs` is a 4-line doc-comment stub with
no implementation; `Cargo.toml` dependencies are already wired.

- [x] Scaffold `crates/tpt-opt-robust/` (empty stub only)
- [x] Wire deps: `tpt-opt-core`, `tpt-opt-milp`, `tpt-math-optimize-convex`, `tpt-math-prob`
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

**Status: scaffolded only.** `src/lib.rs` is a 4-line doc-comment stub with
no implementation; `Cargo.toml` dependencies are already wired.

- [x] Scaffold `crates/tpt-opt-decompose/` (empty stub only)
- [x] Wire deps: `tpt-opt-core`, `tpt-opt-milp`, `tpt-math-optimize-convex`
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

**Status: scaffolded only.** `src/lib.rs` is a 4-line doc-comment stub;
`Cargo.toml` has no dependencies and only the default `std` feature — the
per-solver optional deps/features (`milp`, `minlp`, `network`, `cp`,
`heuristic`, `multi`, `robust`, `decompose`) are not wired yet. Also blocked
in practice until Phase 4/5/7/8/9 crates have real implementations to
re-export.

- [x] Scaffold `crates/tpt-opt-systems/` (empty stub only)
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

- [x] **`deny.toml` schema mismatch**: `cargo deny check` previously failed
      immediately with a config-deserialization error — `[advisories.osv]`
      is not a recognized key for the installed `cargo-deny` 0.20.2. **FIXED**:
      removed the `[advisories.osv]` table; `cargo deny check` now passes clean.
- [x] **`tpt-opt-network` does not compile**: `lib.rs` declared and
      re-exported from `mod opf` and `mod dynamic`, but `src/opf.rs` and
      `src/dynamic.rs` did not exist on disk, and `src/lp.rs` had 3
      immutable/mutable-borrow conflicts plus 5 assignments through `&self`
      that needed `&mut self`. **FIXED**: implemented both missing modules
      (`opf.rs` with DC/AC/SC-OPF, `dynamic.rs` with `DynamicNetwork`) and
      corrected the borrow errors; the crate now compiles, is clippy/fmt-clean,
      and its 7 tests pass.
- [ ] **HiGHS build dependency**: the optional `highs` feature (Phase 2a) pulls in a C++ build via `highs-sys`-style bindings — document the added build-toolchain requirement in CI and in `tpt-opt-milp`'s README; confirm it doesn't break the no-default-features build for `no_std`/minimal consumers
- [ ] **Crate name availability**: all 10 `tpt-opt-*` names are assumed available on crates.io but not yet verified — resolve in the Publish-Readiness Phase before any reservation attempt
- [ ] **Benchmark corpora size**: MIPLIB 2017 / MINLPLib / Netlib / CSPLib instance files are large external downloads — integration tests must fetch/cache them outside the published crate (e.g. a `tests/fixtures/` dir excluded via `.gitignore`/`exclude` in `Cargo.toml`, populated by a CI step or `xtask` command) so they never bloat the packaged `.crate` file
- [ ] **SCIP/Gurobi/CPLEX deferral**: spec §4 mentions these as pluggable via feature flags; explicitly out of scope for this pass on licensing grounds — revisit only if a future consumer has a commercial license and requests it as an opt-in, clearly-labeled non-default feature
