# Contributing to tpt-systems-optimisation

## Workflow

This repository currently follows an **issues-only** workflow for external
contributors: please open an issue describing the bug or proposed change
before submitting code. Internal developers push feature branches and merge
via pull request; external PRs are accepted once they map to a tracked issue.

## Per-crate checklist

Every crate follows the same shape (see `todo.md` for the authoritative
checklist): scaffold → wire deps → implement scope → unit tests + doctests →
rustdoc → fmt/clippy clean → `cargo deny` clean → README + CHANGELOG →
crates.io metadata. Keep the checklist updated as you complete steps.

## Quality gates

A PR is mergeable when all of the following pass locally (or via
`cargo xtask all` / `just all`):

```text
cargo xtask fmt      # rustfmt --check over the workspace
cargo xtask clippy   # clippy --all-targets --all-features -- -D warnings
cargo xtask test     # tests + doctests, --all-features
cargo xtask deny     # advisories / bans / licenses / sources
```

CI runs these plus no_std cross-checks, feature-powerset sweeps of the
umbrella crate, Tier 2 feature-combination builds, MSRV (Rust **1.84**) build,
informational semver checks, bench compile-smoke, and a non-blocking MILP
performance report.

## License policy

All code is dual-licensed **MIT OR Apache-2.0**. Dependencies must be
permissively licensed — the allowlist lives in [`deny.toml`](./deny.toml)
(MIT, Apache-2.0, BSD, ISC, Unicode-3.0, Zlib, CC0-1.0). Copyleft licenses are
rejected by CI. Proprietary solver bindings (SCIP/Gurobi/CPLEX) stay out of
scope on licensing grounds; HiGHS (MIT) is the only external solver binding,
behind a non-default feature.

## Conventions

- **Changelogs**: per crate only (`crates/<name>/CHANGELOG.md`,
  Keep-a-Changelog format). There is intentionally no root changelog.
- **Errors**: solvers return `Result`/typed statuses instead of panicking;
  reachable panics on well-formed input are bugs.
- **No `unsafe`** in the optimisation crates (see [`SECURITY.md`](./SECURITY.md)).
- **Determinism**: every stochastic component takes a seed; same seed ⇒ same
  result, regardless of thread count.
- **Tolerances**: route numeric comparisons through
  `tpt_opt_core::Tolerances` rather than ad-hoc constants.

## Commit style

Conventional-commit-ish prefixes (`feat:`, `fix:`, `docs:`, `test:`, `ci:`,
`chore:`) keep the history scannable.