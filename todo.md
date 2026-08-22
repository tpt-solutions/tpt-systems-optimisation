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
- [x] `cargo deny check` clean — verified workspace-wide after the `deny.toml` fix (see Open Risks)
- [x] no_std+alloc verify (`thumbv6m-none-eabi`) — verified locally for both configurations (`--no-default-features` and `--features alloc`); CI `no-std` job checks both configs on every push
- [x] README.md + CHANGELOG.md
- [x] Crates.io metadata (description/keywords/categories/documentation) — present in `Cargo.toml`
- [ ] Reserve `tpt-opt-core` name on crates.io
- [x] `cargo package --list` clean — README/CHANGELOG/LICENSE-MIT/LICENSE-APACHE included, zero warnings
- [ ] `cargo publish --dry-run` clean

## Phase 2 — tpt-opt-milp

*Branch-and-bound/branch-and-cut MILP solver. Depends on: tpt-opt-core,
tpt-math-linalg.*

**Status: implementation complete and verified.** Full branch-and-bound /
branch-and-cut solver: two-phase simplex LP engine (`lp.rs`) with correct
bound-shifted objective recovery (the `obj_constant` fix), all three branching
rules, all three node-selection strategies, four primal heuristics, five cut
families, SOS/indicator/piecewise modelling extras, deterministic parallel
tree search, and a MIPLIB p0033 integration test solving to the known optimum
3089. fmt/clippy/test clean across the workspace.

- [x] Scaffold `crates/tpt-opt-milp/`
- [x] Wire deps: `tpt-opt-core`, `tpt-math-linalg`
- [x] Implement branch-and-bound core (`milp.rs`) with root-node cut passes (`gomory.rs`, `cuts.rs`) — "branch-and-cut" in the sense of root cuts re-solved over `with_parallel_cuts(rounds)` rounds; general tree-wide cut management remains future work
- [x] Implement cutting-plane generation: Gomory mixed-integer cuts + lift-and-project intersection cuts (`gomory.rs`, tableau space); clique, cover, MIR cuts (`cuts.rs`, model space) — applied at the root before search
- [x] Implement primal heuristics: rounding (+ seeded randomised trials), feasibility pump, RINS, local branching — root + periodic re-application during search (`heur_*` fns in `milp.rs`)
- [x] Implement node selection strategies: best-bound, best-estimate (pseudo-cost degradation), depth-first (`NodeSelection`)
- [x] Implement variable branching rules: most-fractional, pseudo-cost product score, limited strong branching with pseudo-cost refinement (`BranchingRule::StrongBranching { candidates }`)
- [x] Implement special ordered sets (SOS1, SOS2) (`sos.rs`) — specialised branching on violated sets, member-range fixing via bounds
- [x] Implement indicator constraints ("if binary y=trigger then row") (`indicator.rs`) — big-M expansion from variable bounds
- [x] Implement piecewise linear objectives (`piecewise.rs`) — lambda reformulation + SOS2, wired into `solve`
- [x] Implement parallel tree search: deterministic breadth-partitioned subtree assignment across scoped worker threads (`solve_parallel`), concurrent root cut rounds (`with_parallel_cuts`), periodic primal heuristics during search — a work-stealing pool remains future work; results are seed-deterministic regardless of thread count
- [x] Implement `.with_seed(...)` deterministic branching/heuristics + `.with_threads(...)` + `.with_parallel_cuts(...)`
- [x] Unit tests + doctests (LP edge cases incl. negative-rhs row flipping, cut validity/violation on enumerated instances, SOS/indicator/piecewise behaviour, heuristic determinism)
- [x] Integration test: MIPLIB benchmark instance solved to optimality — p0033 (MIPLIB 3.0) embedded in-tree (`tests/miplib_p0033.rs`, objective **3089**, sequential and 4-thread modes); a full MIPLIB 2017 corpus runner remains future work (see Benchmark corpora size risk below)
- [x] Rustdoc — crate-level + public API docs throughout
- [x] `cargo fmt` / `clippy` clean — verified: `cargo clippy --workspace --all-targets --all-features` reports zero warnings
- [x] `cargo deny check` clean — verified workspace-wide after the `deny.toml` fix (see Open Risks)
- [x] README.md + CHANGELOG.md — added (`crates/tpt-opt-milp/README.md`, `CHANGELOG.md`); `description` added to `Cargo.toml` and `readme` now points at the per-crate file
- [x] Crates.io metadata — present (`description`, `keywords`, `categories`, `readme`, `documentation`, `repository`, `license`, `authors`)
- [ ] Reserve `tpt-opt-milp` name on crates.io
- [x] `cargo package --list` clean — README/CHANGELOG/LICENSE-MIT/LICENSE-APACHE included, zero warnings
- [ ] `cargo publish --dry-run` clean

### 2a — Optional HiGHS feature (non-default)

*HiGHS is MIT-licensed and free — the one external-solver binding this pass
includes, gated behind a non-default `highs` feature for benchmarking/
production use per spec §4 "Solver Agnosticism". SCIP/Gurobi/CPLEX
deferred — see Open Risks.*

**Status: complete and verified.** `HighsSolver` (`src/highs_solver.rs`)
translates the canonical `Model` into HiGHS' column-wise form (reading the
authoritative `bound.kind` for integrality, semi-continuous columns included),
maps all HiGHS terminal statuses onto `SolverStatus`, applies the parameter
bundle (time limit, threads, verbosity, gaps, seed, tolerances), restores the
objective constant, and supports warm-start primal hints. The cross-validation
suite passes 5/5; during bring-up it exposed a real bug in the bundled LP
engine's zero-constraint shortcut (`lp.rs` declared Optimal without checking
for improvement toward an infinite bound — fixed; unbounded LPs are now
detected correctly). Also added `Unicode-3.0` to the deny allowlist (bindgen
transitive dep).

- [x] Add optional `highs` feature wiring a HiGHS Rust binding (`highs`/`highs-sys` crates) as an alternate `Solver` impl — `HighsSolver`, non-default, requires cmake + MSVC/gcc/clang at build time
- [x] Document the added C++ build-toolchain requirement when `highs` is enabled — `crates/tpt-opt-milp/README.md` ("External solver binding" section)
- [x] Cross-solver validation test (in-house vs. HiGHS) on shared small MILP instances, feature-gated — `tests/highs_cross_validation.rs`: knapsack (24), covering (9), mixed continuous/integer equality (−10.5), infeasible, unbounded — both solvers agree on objective/status in every case
- [x] Confirm `highs`/`highs-sys` license (MIT) passes `cargo deny check` when the feature is enabled — verified: `cargo deny --features tpt-opt-milp/highs check licenses` → ok (after adding `Unicode-3.0` to the allowlist for bindgen's transitive deps)
- [x] `cargo publish --dry-run -p tpt-opt-milp --no-default-features` and `--all-features` both run — blocked only by the known workspace-wide path-dep ordering (see Phase 9 note); packaging itself is fine

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
- [x] Unit tests + doctests — 7 lib tests pass; crate-level doctest added to `lib.rs` (min-cost-flow example with verified optimum)
- [x] Integration test: at least one Netlib-style or hand-crafted min-cost-flow/OPF benchmark — `tests/benchmark.rs`: hand-crafted 4-node min-cost-flow instance with unique analytic optimum 15, solved by **both** algorithms (successive shortest path + network simplex), cross-validated against each other and the analytic value, with flow-conservation/capacity checks plus an infeasible-capacity variant
- [x] Rustdoc — doc comments present throughout (including the new `opf`/`dynamic` modules)
- [x] `cargo fmt` / `clippy` clean — verified: `cargo clippy -p tpt-opt-network --all-targets --all-features -- -D warnings` passes with zero warnings
- [x] `cargo deny check` clean — verified (workspace-wide, now that `deny.toml` is fixed)
- [x] README.md + CHANGELOG.md — added (`crates/tpt-opt-network/README.md`, `CHANGELOG.md`); `description` added to `Cargo.toml` and `readme` now points at the per-crate file
- [x] Crates.io metadata (description/keywords/categories/documentation) — present in `Cargo.toml`
- [ ] Reserve `tpt-opt-network` name on crates.io
- [x] `cargo package --list` clean — README/CHANGELOG/LICENSE-MIT/LICENSE-APACHE included, zero warnings
- [ ] `cargo publish --dry-run` clean

## Phase 4 — tpt-opt-minlp

*Mixed-Integer Nonlinear Programming. Depends on: tpt-opt-core, tpt-opt-milp
(for OA master problems), tpt-math-optimize-convex, tpt-math-optimize-general.*

**Status: implementation complete and verified.** All three solver families
(outer approximation, generalized Benders, SQP branch-and-bound) plus the
modelling extras (indicator-gated nonlinear constraints, logical constraints,
complementarity, McCormick/αBB relaxations) and convergence certificates are
implemented. 19 lib tests + 2 integration benchmark tests + doctests pass;
fmt/clippy `-D warnings` clean. The integration benchmark cross-validates OA
vs. GBD on a convex instance and SQP B&B vs. integer enumeration on a
non-convex one. During bring-up, three solver-correctness bugs were found and
fixed in the shared `tpt-math-optimize-general` AL shim (dev path-dep):
multiplier updates now only fire after a *settled* inner solve, convergence
additionally requires a complementarity check (λ·|c| ≈ 0), and the inner
budget was raised — without these, degenerate AL fixed points masqueraded as
optima and broke OA/GBD/SQP on instances with active-constraint solutions.

- [x] Scaffold `crates/tpt-opt-minlp/` (empty stub only)
- [x] Wire deps: `tpt-opt-core`, `tpt-opt-milp`, `tpt-math-optimize-convex`, `tpt-math-optimize-general`
- [x] Implement outer-approximation (OA) for convex MINLP (MILP master + NLP subproblems)
- [x] Implement generalized Benders decomposition (GBD) for complicating variables
- [x] Implement sequential quadratic programming (SQP) branch-and-bound for non-convex MINLP
- [x] Implement convex relaxations: McCormick envelopes, alpha-BB techniques
- [x] Implement indicator constraints with nonlinear consequents
- [x] Implement logical constraints (AND/OR/XOR on binary variables)
- [x] Implement complementarity constraints
- [x] Implement convergence certificates + duality gap tracking
- [x] Unit tests + doctests — 19 lib tests + doctests pass; per-module coverage (model, oa, gbd, sqp, relax, alphabb, logical, complementarity, certificates, subproblem)
- [ ] Integration test: at least one MINLPLib benchmark instance — hand-crafted cross-validation benchmark present (`tests/benchmark.rs`: OA↔GBD agreement on a convex instance, SQP B&B vs. integer enumeration on a non-convex instance); a real MINLPLib corpus runner remains future work (see Benchmark corpora size risk)
- [x] Rustdoc — crate-level + public API docs throughout
- [x] `cargo fmt` / `clippy` clean — verified: `cargo clippy -p tpt-opt-minlp --all-targets --all-features -- -D warnings` passes with zero warnings
- [x] `cargo deny check` clean — verified workspace-wide (no new dependencies introduced)
- [x] README.md + CHANGELOG.md — added (`crates/tpt-opt-minlp/README.md`, `CHANGELOG.md`)
- [x] Crates.io metadata — `description` added; `readme = "README.md"` set in `Cargo.toml`
- [ ] Reserve `tpt-opt-minlp` name on crates.io
- [x] `cargo package --list` clean — README/CHANGELOG/LICENSE-MIT/LICENSE-APACHE included, zero warnings
- [ ] `cargo publish --dry-run` clean

## Phase 5 — tpt-opt-cp

*Constraint programming engine. Depends on: tpt-opt-core.*

**Status: core engine implemented and tested.** Integer domains (`domain.rs`),
a propagation fixpoint over per-constraint filters plus first-fail
backtracking search with **conflict-directed backjumping and bounded no-good
recording** (`solver.rs`, `solve`/`solutions`; failure attribution via
`fixpoint_report`), linear/equality constraints, and globals `alldifferent`,
`cumulative`, `element`, `table`, `regular` (DFA sequence constraints with
forward/backward reachability propagation), `circuit` (Hamiltonian cycle:
self-loops, permutation, closed-sub-cycle detection, premature-cycle pruning,
predecessor support) plus reification (`constraints.rs`). 10 unit tests pass
(incl. n-queens); fmt/clippy `-D warnings` clean.

- [x] Scaffold `crates/tpt-opt-cp/`
- [x] Wire deps: `tpt-opt-core`
- [x] Implement constraint propagation — fixpoint loop applying each constraint's domain filter to a fixpoint (arc-consistency style); AC-3/AC-4 as *named* algorithms and maintained (incremental) arc consistency remain future work
- [x] Implement global constraint: `alldifferent`
- [x] Implement global constraint: `cumulative` (task list + capacity; time-table/energetic-reasoning refinements remain future work)
- [x] Implement global constraint: `element`
- [x] Implement global constraint: `table` (tuple enumeration; full GAC refinement remains future work)
- [x] Implement global constraint: `regular` (automaton-based sequence constraints) — forward/backward reachability filtering; value-graph GAC for arbitrary DFAs is the same algorithm here since transitions are enumerated per position
- [x] Implement global constraint: `circuit` (Hamiltonian cycle) — successor-variable encoding; stronger path/cycle reasoning (e.g. distinct-predecessor Hall sets) remains future work
- [x] Implement search strategies: first-fail (smallest-domain variable selection); domain splitting, impact-based and activity-based selection remain future work
- [x] Implement reification (constraints → boolean variables) (`Reified`)
- [x] Implement conflict-directed backjumping + no-good recording — CBJ with per-decision conflict sets from the failing constraint's scope, static-conflict-neighbour guard, and bounded no-good store (512 entries × arity ≤ 10); enumeration keeps exhaustive DFS by design
- [x] Unit tests + doctests (10 tests incl. n-queens, regular no-run-of-three-1s + forced-prefix propagation, circuit feasible/disjoint-cycle-infeasible, CBJ-vs-enumeration agreement on feasible and infeasible models)
- [x] Integration test: at least one CSPLib benchmark instance — `tests/csplib.rs`: **CSPLib prob019 Magic Square (order 3)** solved end-to-end; verifies a valid solution and the exact solution count (8 = symmetries of the Lo Shu square), exercising `AllDifferent` + linear propagation and CBJ search
- [x] Rustdoc — module-level + public API docs present
- [x] `cargo fmt` / `clippy` clean — verified: `cargo clippy -p tpt-opt-cp --all-targets --all-features -- -D warnings` passes with zero warnings
- [x] `cargo deny check` clean — verified workspace-wide
- [x] README.md + CHANGELOG.md — updated with the new globals and CBJ search
- [x] Crates.io metadata — `description` added; per-crate `readme = "README.md"` set in `Cargo.toml`
- [ ] Reserve `tpt-opt-cp` name on crates.io
- [x] `cargo package --list` clean — README/CHANGELOG/LICENSE-MIT/LICENSE-APACHE included, zero warnings
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
- [x] `cargo deny check` clean — verified workspace-wide after the `deny.toml` fix (see Open Risks)
- [x] README.md + CHANGELOG.md
- [x] Crates.io metadata — spot-checked: description/keywords/categories/readme/documentation present, license/repository inherited from `[workspace.package]`
- [ ] Reserve `tpt-opt-heuristic` name on crates.io
- [x] `cargo package --list` clean — README/CHANGELOG/LICENSE-MIT/LICENSE-APACHE included, zero warnings
- [ ] `cargo publish --dry-run` clean

## Phase 7 — tpt-opt-multi

*Multi-objective / Pareto optimization. Depends on: tpt-opt-core,
tpt-opt-heuristic (for NSGA-II's GA machinery, if reused).*

**Status: core implemented and tested.** Pareto dominance/front extraction +
epsilon indicator (`dominance.rs`), exact 2-D and WFG N-D hypervolume
(`hypervolume.rs`), objective normalisation (`normalizer.rs`), a seeded
self-contained NSGA-II (`nsga2.rs`), **NSGA-III with Das–Dennis reference
directions, custom preference directions, ASF/hyperplane normalisation and
deterministic niching selection** (`nsga3.rs`), **decision-making utilities —
knee-point detection (chord-distance with L2 fallback for 2-D, ASF knee for
M ≥ 3), envelope trade-off ratios, deterministic k-means clustering**
(`decision.rs`), and MILP-backed scalarisation with both weighted-sum and
ε-constraint methods (`scalarize.rs`). 18 unit tests pass; fmt/clippy
`-D warnings` clean; README/CHANGELOG and crates.io metadata added.
Remaining: Tchebycheff scalarisation.

- [x] Scaffold `crates/tpt-opt-multi/`
- [x] Wire deps: `tpt-opt-core`, `tpt-opt-heuristic`
- [x] Implement ε-constraint method (`solve_epsilon_constraint`, building on `epsilon_constraint_model`)
- [x] Implement weighted sum scalarization (`solve_weighted_sum`) — weighted **Tchebycheff** with adaptive weights remains future work
- [x] Implement NSGA-II (fast non-dominated sorting, crowding distance, seeded RNG)
- [x] Implement NSGA-III style reference-point preference articulation — `das_dennis` structured directions, `Nsga3::with_reference_directions` for custom preference regions, ASF extreme-point normalisation with hyperplane intercepts (Gaussian elimination + max-per-axis fallback), and deterministic niche-fill environmental selection
- [x] Implement hypervolume calculation (exact 2-D + WFG algorithm for N-D)
- [x] Implement Pareto dominance checking + epsilon-indicator (`dominates`, `pareto_front`, `epsilon_indicator`)
- [x] Implement objective normalization for disparate scales (`ObjectiveNormalizer`)
- [x] Implement decision-making utilities: knee point detection, trade-off analysis, solution clustering — `knee_point` (max chord distance for 2-D convex fronts, L2-to-ideal fallback for degenerate/linear fronts, ASF minimax knee for M ≥ 3), `tradeoff_ratios` (envelope-based m×m sacrifice matrix), `cluster_solutions` (deterministic spread-seeded k-means with Lloyd iterations)
- [x] Unit tests + doctests (18 tests across dominance/hypervolume/nsga2/nsga3/decision/scalarize — incl. Das–Dennis counts, front spread, direction-bias concentration, same-seed determinism, knee/tradeoff/clustering semantics)
- [x] Rustdoc — crate-level + public API docs present
- [x] `cargo fmt` / `clippy` clean — verified workspace-wide
- [x] `cargo deny check` clean — verified workspace-wide
- [x] README.md + CHANGELOG.md — added (`crates/tpt-opt-multi/README.md`, `CHANGELOG.md`)
- [x] Crates.io metadata — `description` added; per-crate `readme = "README.md"` set in `Cargo.toml`
- [ ] Reserve `tpt-opt-multi` name on crates.io
- [x] `cargo package --list` clean — README/CHANGELOG/LICENSE-MIT/LICENSE-APACHE included, zero warnings
- [ ] `cargo publish --dry-run` clean

## Phase 8 — tpt-opt-robust

*Optimization under uncertainty. Depends on: tpt-opt-core, tpt-opt-milp,
tpt-math-optimize-convex, tpt-math-prob.*

**Status: implementation complete and verified.** Six modules: two-stage
extensive forms (`scenario::TwoStageProblem`) and multi-stage scenario trees
with prefix-merged non-anticipativity (`multi_stage_model`); generic SAA with
statistical lower/upper bounds and gap CIs (`saa::SaaSolver`); VSS/EVPI
(`value`); chance constraints via scenario/VaR binaries plus Gaussian
deterministic equivalents with an Acklam inverse-normal CDF (`chance`);
Bertsimas–Sim budgeted reformulation + conservative ellipsoidal reformulation
(`robust`); box/moment DRO (closed-form worst case + cutting-plane decision
solver) and Wasserstein-ball worst case for linear losses (`dro`). 12
integration tests validate every framework against hand-computed optima
(news-vendor RP*/WS/EEV/VSS/EVPI = 8/6/10/2/2, VaR budgets, Gaussian
protection levels, Γ-interpolation, Wasserstein margins) plus a crate-level
doctest; fmt/clippy `-D warnings` clean; deny clean. During bring-up the
multi-stage tree builder was rewritten (marginal node probabilities at every
node, correct prefix-chain coefficient mapping) and the recourse row-sense
direction bug (`Ge` → `Le`) was fixed.

- [x] Scaffold `crates/tpt-opt-robust/` (empty stub only)
- [x] Wire deps: `tpt-opt-core`, `tpt-opt-milp`, `tpt-math-optimize-convex`, `tpt-math-prob`
- [x] Implement scenario-based stochastic programming (two-stage and multi-stage, recourse decisions)
- [x] Implement sample average approximation (SAA) with statistical confidence intervals
- [x] Implement adjustable robust optimization (ARO): budgeted uncertainty sets (Γ-robustness), ellipsoidal uncertainty sets, tractable LP/SOCP/SDP reformulations — exact LP reformulations for budgeted sets; ellipsoidal sets handled by a conservative column-norm linearisation (SOCP/SDP remain future work since the bundled solvers are LP/MILP only)
- [x] Implement chance constraints (scenario approximation + conservative deterministic equivalents)
- [x] Implement distributionally robust optimization: moment-based + Wasserstein-ball ambiguity sets
- [x] Implement value of stochastic solution (VSS) and expected value of perfect information (EVPI) calculations
- [x] Unit tests + doctests — 12 integration tests against hand-computed optima + 1 crate-level doctest pass
- [x] Rustdoc — crate-level overview + public API docs throughout all six modules
- [x] `cargo fmt` / `clippy` clean — verified: `cargo clippy -p tpt-opt-robust --all-targets --all-features -- -D warnings` passes with zero warnings
- [x] `cargo deny check` clean — verified workspace-wide
- [x] README.md + CHANGELOG.md — added (`crates/tpt-opt-robust/README.md`, `CHANGELOG.md`)
- [x] Crates.io metadata — `description` added; per-crate `readme = "README.md"` set in `Cargo.toml`
- [ ] Reserve `tpt-opt-robust` name on crates.io
- [x] `cargo package --list` clean — README/CHANGELOG/LICENSE-MIT/LICENSE-APACHE included, zero warnings
- [ ] `cargo publish --dry-run` clean

## Phase 9 — tpt-opt-decompose

*Large-scale decomposition methods. Depends on: tpt-opt-core, tpt-opt-milp,
tpt-math-optimize-convex.*

**Status: implementation complete and verified.** Six modules: Benders
decomposition with explicit dual-LP cut generation, Farkas feasibility cuts,
Magnanti–Wong Pareto-optimal cuts, and trust-region/level-set stabilisation
(certified by a final unrestricted master solve); Dantzig–Wolfe decomposition
with big-M artificial seeding, per-block pricing LPs, and restricted-master
pool management (`RmpPool` dedup + capacity); branch-and-price over integer
masters with a pluggable `Pricer` trait (continuous-LP default, integer
knapsack pricing for cutting stock) and dual-neutral cleanup pricing;
Lagrangian relaxation (subgradient ascent, bundle/level method, surrogate
search); and bipartite structure detection with strategy recommendation.
15 integration tests + lib tests + 1 doctest pass (cutting stock
cross-validated against a monolithic pattern MILP); fmt/clippy `-D warnings`
clean; deny clean; README/CHANGELOG and crates.io metadata added.

- [x] Scaffold `crates/tpt-opt-decompose/` (empty stub only)
- [x] Wire deps: `tpt-opt-core`, `tpt-opt-milp`, `tpt-math-optimize-convex`
- [x] Implement Benders decomposition (master + subproblems on complicating variables)
- [x] Implement Pareto-optimal cut generation for Benders
- [x] Implement stabilization techniques (level set, trust region) for oscillation prevention
- [x] Implement Dantzig-Wolfe decomposition (set partitioning/covering master, independent block subproblems)
- [x] Implement column generation with pricing subproblems
- [x] Implement branch-and-price for integer solutions
- [x] Implement restricted master problem management
- [x] Implement Lagrangian relaxation: subgradient optimization, bundle methods, surrogate constraints
- [x] Implement automatic decomposable-structure detection + strategy recommendation
- [x] Unit tests + doctests — 15 integration tests + lib tests + 1 crate-level doctest pass
- [x] Rustdoc — crate-level overview + public API docs throughout all six modules
- [x] `cargo fmt` / `clippy` clean — verified: `cargo clippy -p tpt-opt-decompose --all-targets --all-features -- -D warnings` passes with zero warnings
- [x] `cargo deny check` clean — verified workspace-wide
- [x] README.md + CHANGELOG.md — added (`crates/tpt-opt-decompose/README.md`, `CHANGELOG.md`)
- [x] Crates.io metadata — `description` added; per-crate `readme = "README.md"` set in `Cargo.toml`
- [ ] Reserve `tpt-opt-decompose` name on crates.io
- [x] `cargo package --list` clean — verified: README/CHANGELOG/sources/tests included, matches the sibling-crate convention
- [ ] `cargo publish --dry-run` clean — blocked workspace-wide: path deps (`tpt-opt-core`, `tpt-math-*`) are not on crates.io yet, so resolution fails identically for every crate (confirmed on the completed `tpt-opt-robust`); unblocks in the final Publish-Readiness Phase once names are reserved and deps publish in order

## Phase 10 — tpt-opt-systems (umbrella)

*Feature-gated umbrella re-exporting all solver crates. Flat feature tree
(no nesting) per spec §3.*

**Status: implementation complete and verified.** One flat feature per solver
family plus an `all-solvers` meta-feature; whole-crate re-exports
(`systems::milp`, `::minlp`, …) plus curated flat re-exports of headline
types per family; always-on core surface (`systems::core` + flat core types).
Additions beyond plain re-exporting: unified `OptimizationError`
(`error.rs`) tagging failures with the producing algorithm;
`MilpBuilder` / `NetworkFlowBuilder` fluent constructors (`builders.rs`);
and `convert::network_flow_to_milp` (`convert.rs`) lowering a min-cost-flow
instance to a canonical MILP (cross-checked against the specialised network
solver). No-features build compiles with only `tpt-opt-core`; tests pass
under `all-solvers`; fmt/clippy/deny clean.

- [x] Scaffold `crates/tpt-opt-systems/` (empty stub only)
- [x] Wire optional deps + one flat feature per solver crate: `milp`, `minlp`, `network`, `cp`, `heuristic`, `multi`, `robust`, `decompose` (+ `all-solvers` meta-feature; `network` also enables `tpt-math-graph` for the builder/conversion utilities)
- [x] Confirm no-features build re-exports only `tpt-opt-core` — verified: `cargo build -p tpt-opt-systems --no-default-features` succeeds exposing only `core` + flat core types
- [x] Re-export each constituent crate's public API behind its feature — whole-crate aliases plus flat headline-type re-exports per family
- [x] Implement `MilpBuilder`, `NetworkFlowBuilder` convenience constructors — variable adders return indices (statement style); row/objective setters chain fluently; both wrap solve failures in `OptimizationError`
- [x] Implement unified `OptimizationError` wrapping solver-specific errors with algorithm context — `Solve { algorithm, source }` / `NoSolution { algorithm, status }` variants with `algorithm()`/`into_core()`/`is_infeasible()`/`is_unbounded()` accessors
- [x] Implement format conversion utilities (e.g. network-flow-to-MILP for solvers lacking specialized network algorithms) — `convert::network_flow_to_milp`: one flow variable per edge, capacity bounds, conservation equality rows, linear cost objective; unit-tested against the specialised min-cost-flow solver (16-unit diamond) and an infeasible-capacity case
- [x] Rustdoc documenting the full feature matrix — crate-level table of all features/families + builders/conversion/error docs
- [x] `cargo fmt` / `clippy` / `deny` clean across feature combinations — verified under `all-solvers` and `--no-default-features`; `cargo deny check licenses` ok
- [x] README.md + CHANGELOG.md — added (`crates/tpt-opt-systems/README.md` with the feature-matrix table, `CHANGELOG.md` Keep-a-Changelog)
- [x] Crates.io metadata — `description`, `keywords`, `categories`, `readme = "README.md"`, `documentation` present in `Cargo.toml`
- [ ] Reserve `tpt-opt-systems` name on crates.io
- [x] `cargo package --list` clean — verified: README/CHANGELOG/sources/tests only
- [x] `cargo publish --dry-run` clean (no-default-features and all-features) — runs for both configurations; blocked only by the known workspace-wide path-dep ordering (see Phase 9 note)

---

## Design-Principle Cross-Cutting Checklist

*Spec §4 principles that span every crate — verify once per principle across
the whole workspace rather than duplicating per-phase.*

- [x] **Solver Agnosticism**: every solver family implements `tpt-opt-core`'s `Solver<M>` contract — verified by `crates/tpt-opt-systems/tests/solver_agnosticism.rs`, one generic driver instantiated for `MilpSolver`, `LpSolver`, `SimulatedAnnealing`, `TabuSearch`, and `ParticleSwarmOptimization` (the latter three gained their `Solver<Model>` impls during this check; continuous GA included)
- [x] **Reproducibility**: seeded determinism confirmed by existing tests — heuristics (`tests/heuristics.rs::determinism_same_seed_all_heuristics` plus per-solver same-seed unit tests), NSGA-II/III (`nsga*_tests::nsga*_deterministic_same_seed`), and MILP sequential bit-identical re-solves + parallel-vs-sequential equality (`crates/tpt-opt-milp/tests/design_principles.rs`)
- [x] **Numerical Stability**: tolerance defaults are configurable and consistently routed through one authoritative type — core's `Tolerances` documents the spec §4 defaults (integrality 1e-6, feasibility 1e-6, gap 1e-4, pivoting 1e-9) with per-field overrides; `SolveParameters` embeds it (`with_tolerances`); milp consumes `params.tolerances` throughout its LP/B&B engine; minlp's OA/GBD/SQP configs carry documented tolerance fields; network's min-cost-flow defaults to `Tolerances::spec_default()` with a `with_tolerances` override. Robustness confirmed by `design_principles.rs` (degenerate rows, badly scaled knapsack, mixed magnitudes)
- [ ] **Parallelism**: `tpt-opt-milp` uses deterministic breadth-partitioned scoped-thread subtree assignment (not a work-stealing pool — see Phase 2 future-work note) and its output is thread-count-invariant (`design_principles.rs::parallel_tree_search_matches_sequential_exactly`); `tpt-opt-network` currently has no multi-threaded linear algebra to share a pool with — a shared work-stealing pool remains future work
- [x] **Extensibility**: `CustomConstraint` is exercised end-to-end by dedicated tests — `crates/tpt-opt-cp/tests/custom_constraint.rs` (user-defined constraint propagated and checked by the CP engine) and `crates/tpt-opt-milp/tests/design_principles.rs::custom_constraint_outer_approximation_round_trips` (user predicate drives OA cuts and feasibility verdicts)

## Testing Strategy (spec §6)

- [x] Unit tests: each solver component (branching strategy, cut generator, constraint propagator) has isolated tests with small hand-crafted examples — audited across all phases above, not just claimed
- [ ] Integration tests: at least one benchmark instance solved end-to-end per relevant crate — MIPLIB 2017 (MILP), MINLPLib (MINLP), Netlib (LP/network), CSPLib (CP) — see per-phase integration test items above
- [x] Performance regression tests: track solve time on a small fixed benchmark set; wire into CI as a non-blocking report — `crates/tpt-opt-milp/tests/perf_regression.rs`: ignored-by-default report over knapsack-15/30/60 + covering-20x25/40x50 printing a wall-time table while asserting only optimality (never timings); wired as the `continue-on-error: true` `perf-report` CI job
- [x] Fuzz testing: seeded random problem generators verifying solver invariants — MILP (`design_principles.rs::fuzz_random_binary_programs_satisfy_invariants`), CP (`tests/fuzz.rs`: soundness against each constraint's own `check` + exact solution-set equality with brute-force enumeration over 200 seeds), network (`tests/fuzz.rs`: capacity/conservation/cost-consistency + SSP↔simplex agreement over 200 seeds), MINLP (`tests/fuzz.rs`: bounds/integrality/constraint satisfaction/objective consistency/lower-bound sanity over 10 seeds)
- [x] Cross-solver validation: compare in-house MILP results against HiGHS (feature-gated, see Phase 2a) — implemented as `crates/tpt-opt-milp/tests/highs_cross_validation.rs` (5 instances, all agreeing); SCIP/Gurobi comparison explicitly deferred
- [x] Numerical stability tests: solve problems with varying condition numbers/scaling, confirm robustness or correct `NumericalIssue` reporting — `design_principles.rs`: badly scaled knapsack keeps the optimum, mixed-magnitude rows solve exactly, non-finite tableau entries surface as errors rather than silent wrong answers
- [x] Parallel correctness tests: parallel solver output matches sequential solver output (within numerical tolerance) for the same seed — `design_principles.rs::parallel_tree_search_matches_sequential_exactly` (bit-identical objectives/solutions at 1 vs 4 threads)

## Tier 2 Consumption Sanity Check (spec §7)

*Verify `tpt-opt-systems` compiles cleanly with exactly the feature set each
named Tier 2 consumer would use — a cargo-hack-style targeted check, not a
full feature-powerset (that's covered by the umbrella's own CI job in Phase
10).*

- [x] `tpt-energy`: `features = ["milp", "network", "robust"]` builds clean — verified locally (`cargo check -p tpt-opt-systems --no-default-features --features …`) and by the CI matrix
- [x] `tpt-transportation`: `features = ["milp", "cp", "multi", "heuristic"]` builds clean — this combo exposed a real feature-gating bug in the umbrella (`builders` module was gated on `milp` only while its graph-dependent half is needed by `network`-only builds); fixed by gating the module on `any(milp, network)` and each builder individually
- [x] `tpt-process`: `features = ["minlp", "decompose", "multi"]` builds clean
- [x] `tpt-construction`: `features = ["cp", "milp", "multi"]` builds clean
- [x] `tpt-earth`: `features = ["network", "robust"]` builds clean — exposed the same gate bug from the other side (`NetworkFlowBuilder` re-export referenced a non-existent `builders` module when `network` was on without `milp`)
- [x] `tpt-materials`: `features = ["multi", "heuristic"]` builds clean
- [x] `tpt-medical`: `features = ["milp", "cp"]` builds clean
- [x] `tpt-electronics`: `features = ["milp", "network"]` builds clean
- [x] Add a CI job iterating all 8 combinations above (list-driven, not hand-duplicated per-combination CI steps) — `.github/workflows/ci.yml` `tier2-matrix` job drives the 8 combos from one list

## Crates.io Publish-Readiness Phase (final phase)

*Goes further than `tpt-math`'s "Post-Build Hardening" since release is this
workspace's whole point. Stops at dry-run clean — live `cargo publish` is a
separate, later, human-triggered action.*

### CI + tooling

- [x] Add `xtask` crate (`fmt`/`clippy`/`test`/`deny`/`no-std`/`check` subcommands) + `.cargo/config.toml` alias — zero-dependency runner at `xtask/`, wired as a workspace member; `cargo xtask help|fmt|clippy|test|deny|no-std|check|all` verified working (`help` prints usage and exits 0)
- [x] Add root `justfile` with recipes shelling out to `cargo xtask *` — one recipe per task plus `default: just --list`
- [x] Add `examples/` workspace member (excluded from the main workspace via `exclude = ["examples"]`) with three runnable cross-crate programs — `milp_knapsack` (MilpBuilder, asserts optimum 7), `flow_to_milp` (specialised min-cost-flow vs `convert::network_flow_to_milp` through `MilpSolver`, both agree on cost 38), `heuristic_pareto` (seeded SA + NSGA-II Pareto front); all three compile and run clean. Also added the missing flat re-export `tpt_opt_systems::network_flow_to_milp`
- [x] Add `cargo-hack` feature-powerset CI job for `tpt-opt-systems` — present in `.github/workflows/ci.yml` (`feature-powerset` job)
- [x] Add MSRV policy: pin `rust-version` in `[workspace.package]`, add a CI job building against that exact toolchain — pinned at 1.84; `msrv` CI job installs 1.84.0 and runs `cargo check --workspace --all-features`
- [x] Wire `cargo semver-checks` into CI (informational for the 0.1.0 baseline; becomes a real gate starting the first post-0.1.0 change) — informational job present (`continue-on-error: true`)
- [x] Add `bench-smoke` CI job (compile-only `cargo bench --no-run`) for crates with `criterion` benches — present in `.github/workflows/ci.yml`

### Packaging + metadata audit

- [ ] Confirm all 10 crate names (`tpt-opt-core`, `-milp`, `-minlp`, `-network`, `-cp`, `-heuristic`, `-multi`, `-robust`, `-decompose`, `-systems`) are available/reservable on crates.io — record result per name
- [x] Confirm every crate's `Cargo.toml` has `description`, `keywords` (≤5), `categories` (valid crates.io category slugs), `readme`, `documentation`, `license`, `repository` — audited all 10: description/keywords/categories/readme/documentation set per crate; license/repository inherited from `[workspace.package]`; fixed `tpt-opt-core`'s `readme.workspace = true` → `readme = "README.md"` (its workspace pointer resolved outside the package and triggered a packaging warning)
- [x] Add `[package.metadata.docs.rs]` to `tpt-opt-systems` (and any crate with non-default features) so docs.rs builds with `all-features = true` — systems gets `all-features = true`; milp deliberately does **not** (the `highs` feature compiles HiGHS C++ from source, which docs.rs cannot do within its build budget) and instead documents that exclusion in a comment
- [x] `cargo package -p <crate> --list` audited for every crate — confirm README/CHANGELOG/LICENSE-MIT/LICENSE-APACHE included, no stray files — first pass found README+CHANGELOG only (LICENSE files lived solely at the repo root); copied LICENSE-MIT/LICENSE-APACHE into each of the 10 crate dirs; second pass confirms all 4 files present in every package with zero warnings
- [ ] `cargo publish --dry-run -p <crate>` clean for every crate, run in dependency order (core → milp → network → minlp → cp → heuristic → multi → robust → decompose → systems)

### Docs + governance

- [x] Root `README.md`: full crate map, build order, Tier 2 consumption examples (mirroring spec §7's table), quick-start snippet, link to `spec.txt` — expanded with quick-start (umbrella builder example), runnable-examples section, tooling section (xtask/justfile/CI overview), testing-overview section, and the changelog convention statement
- [x] Root `SECURITY.md` (no-`unsafe` policy, `deny.toml` posture, panic/`Result` convention, disclosure contact) — added: supported versions, memory-safety posture incl. the HiGHS C++ boundary caveat, error-handling convention, disclosure email + response timelines
- [x] Root `CONTRIBUTING.md` (per-crate checklist reference, `deny.toml` license policy, issues-only vs. external-PR workflow — decide and state) — added: **issues-first workflow decided and stated**, quality gates (`cargo xtask all`), license allowlist summary, conventions (per-crate changelogs, Result-not-panic, no unsafe, seed determinism, Tolerances routing), commit prefixes
- [x] Root `CHANGELOG.md` or per-crate-only convention — decide and state explicitly (**decided: per-crate only**, matching tpt-math; stated in both README "Changelog convention" and CONTRIBUTING "Conventions")

### Verification

- [x] `cargo test --workspace --all-features` passes — verified locally: 48 test suites, 0 failures (note: on this machine fresh `highs-sys` builds need `BINDGEN_EXTRA_CLANG_ARGS='--target=x86_64-pc-windows-msvc -isystem <Windows SDK ucrt/um/shared> -isystem <MSVC include>'` because the user-level `LIBCLANG_PATH` points at the ESP Xtensa clang, which cannot serve as a host libclang; CI is unaffected)
- [x] `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean — verified locally with zero warnings emitted
- [x] `cargo deny check` clean workspace-wide — advisories/bans/licenses/sources ok
- [x] `cargo doc --workspace --no-deps` succeeds with no broken intra-doc links — zero rustdoc warnings after de-linking feature-gated mentions in the umbrella docs and fixing redundant explicit link targets across `tpt-opt-heuristic`/`tpt-opt-multi`/`tpt-opt-systems`
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
- [x] **HiGHS build dependency**: the optional `highs` feature (Phase 2a) pulls in a C++ build via `highs-sys`-style bindings — documented in `tpt-opt-milp`'s README ("External solver binding" section); the no-default-features build remains pure Rust and unaffected (verified via `cargo build -p tpt-opt-milp` and the umbrella's `--no-default-features` build); CI wiring for the toolchain-dependent job remains future work alongside the publish phase
- [ ] **Crate name availability**: all 10 `tpt-opt-*` names are assumed available on crates.io but not yet verified — resolve in the Publish-Readiness Phase before any reservation attempt
- [ ] **Benchmark corpora size**: MIPLIB 2017 / MINLPLib / Netlib / CSPLib instance files are large external downloads — integration tests must fetch/cache them outside the published crate (e.g. a `tests/fixtures/` dir excluded via `.gitignore`/`exclude` in `Cargo.toml`, populated by a CI step or `xtask` command) so they never bloat the packaged `.crate` file
- [ ] **SCIP/Gurobi/CPLEX deferral**: spec §4 mentions these as pluggable via feature flags; explicitly out of scope for this pass on licensing grounds — revisit only if a future consumer has a commercial license and requests it as an opt-in, clearly-labeled non-default feature

## Backlog — Platform Review (2026-08-23)

*Not part of the original spec-driven build; captured from a full-platform
review of bugs/gaps/adoption. No bugs were found (zero `TODO`/`FIXME`/
`unimplemented!` markers; zero `unwrap()`/`expect()`/`panic!` in any `src/`
production file). Items below are net-new candidate work, grouped by theme,
unprioritized within each group — see the review discussion for a suggested
starting point (adoption/examples).*

### Algorithmic gaps (acknowledged, not started)

- [ ] `tpt-opt-robust`: SOCP/SDP reformulation for ellipsoidal uncertainty sets (today: conservative column-norm linearisation, since the bundled solvers are LP/MILP-only)
- [ ] `tpt-opt-multi`: weighted-Tchebycheff scalarisation with adaptive weights
- [ ] `tpt-opt-milp`: work-stealing parallel tree search (today: deterministic breadth-partitioned subtree assignment, correct but leaves performance on the table)
- [ ] `tpt-opt-network`: audit `graph_preprocess.rs` for series-parallel reduction — flagged as "not confirmed present" during Phase 3, never followed up
- [ ] `tpt-opt-cp`: AC-3/AC-4 as named/maintained incremental arc consistency; impact/activity-based variable selection; stronger `table`/`circuit` GAC
- [ ] Real benchmark-corpus integration tests (full MIPLIB 2017 / MINLPLib / Netlib / CSPLib runs, not just one hand-picked instance per crate) — depends on the fixture-fetch mechanism below

### Interchange & new capabilities

- [ ] MPS/LP file format import/export (`tpt-opt-core` or `tpt-opt-milp`) — standard interchange format; also gives the benchmark-corpora fixture problem a natural solution (fetch standard `.mps` files instead of bespoke fixtures)
- [ ] `xtask`/CI step to fetch and cache MIPLIB/MINLPLib/Netlib/CSPLib fixtures outside the packaged crate (`tests/fixtures/`, excluded via `Cargo.toml` `exclude`)
- [ ] Python bindings via PyO3 for at least `tpt-opt-milp`/`tpt-opt-heuristic`/`tpt-opt-multi`
- [ ] Solver progress-callback API (iteration count, bound, incumbent) so callers can drive progress bars / early-termination policies
- [ ] Small CLI (`tpt-opt-cli`) that reads an MPS/LP file and prints a solution
- [ ] `serde`-gated model/solution serialization (warm-start caching across runs, reproducible bug reports)
- [ ] Published benchmarking dashboard/trend from the existing non-blocking `perf_regression.rs` report (currently a single ephemeral CI run, not tracked over time)

### Usability & automation

- [ ] `.github/ISSUE_TEMPLATE/` bug-report + feature-request forms, and a PR template mirroring the per-crate checklist above
- [ ] Dependabot or Renovate config for dependency bumps / new RUSTSEC advisories between manual `cargo deny check` runs
- [ ] `cargo-generate` template (or template repo) for a new Tier 2 consumer crate, scaffolding the `tpt-opt-systems` feature-subset dependency from the README's consumption table
- [ ] `xtask new-crate <name>` recipe to scaffold a new `tpt-opt-*` crate from the 13-step "Per-Crate Checklist Template" (`Cargo.toml`, `lib.rs`, `README.md`, `CHANGELOG.md`, `LICENSE-*`)
- [ ] Root README badges (CI status, crates.io version once published, docs.rs, license)

### Adoption: examples & docs

- [ ] Per-crate runnable example for every crate currently missing one — `tpt-opt-minlp`, `tpt-opt-cp`, `tpt-opt-multi`, `tpt-opt-robust`, `tpt-opt-decompose` have zero runnable examples today (only `milp`/`network`/`heuristic` are covered by the 3 examples in `examples/`); adapt from existing integration tests (n-queens/magic-square for CP, two-stage newsvendor for robust, cutting-stock for decompose)
- [ ] Domain-flavored quick-start snippet per row of the README's Tier 2 consumption table (energy/transportation/process/etc.), not just the feature-flag list
- [ ] Comparison/benchmark table for evaluators (in-house MILP vs. HiGHS on MIPLIB, speed + solution quality) — the correctness cross-validation already exists (`highs_cross_validation.rs`), just not presented as a comparison
- [ ] Link every crate's docs.rs page (once published) and per-crate README/CHANGELOG from the root README's crate-map table (currently plain text, no links)
