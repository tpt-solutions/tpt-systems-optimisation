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
umbrella crate, Tier 2 feature-combination builds, MSRV (Rust **1.85**) build,
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
## Development setup

The devcontainer in [`.devcontainer/`](./.devcontainer) installs everything
below. To set it up by hand:

- Rust **1.85** (`rust-toolchain.toml` pins it), with `rustfmt` and `clippy`.
- `cmake` and a C++ toolchain, only if you build the `highs` feature.
- Optional tools: `just`, `cargo-deny`, `cargo-nextest`, `mdbook`,
  `cargo-llvm-cov`.

```text
cargo test --workspace          # all tests and doctests
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
mdbook build docs               # the user guide, output in docs/book/
```

## Documentation

- **API reference**: rustdoc. Build it with `cargo doc --workspace --no-deps`.
  CI fails on rustdoc warnings, so broken intra-doc links break the build.
- **User guide**: the mdBook in [`docs/`](./docs). Chapters are in
  `docs/src/`, and `SUMMARY.md` controls the order.
- **Per-crate**: each crate's `README.md` and `CHANGELOG.md`.

## Adding a crate

Use the scaffolding that matches the per-crate checklist:

```text
cargo xtask new-crate tpt-opt-example
```

It creates `Cargo.toml` (inheriting workspace fields), a `lib.rs` stub,
`README.md`, `CHANGELOG.md`, and the licence files. Then work through the
checklist in `todo.md`.

## Examples

Umbrella examples live in `crates/tpt-opt-systems/examples/`. Each one names
the feature it needs in `Cargo.toml`, so `cargo test` skips it when the
feature is off. Cross-crate examples are in `examples/`, run with
`cargo run --manifest-path examples/Cargo.toml --example <name>`.

## Releasing the CLI

Tag the commit with `tpt-opt-cli-v<version>`. The `release.yml` workflow builds
archives for Linux, macOS (Apple Silicon), and Windows, then attaches them to
the GitHub release. Publishing to crates.io stays a separate, manual step.
