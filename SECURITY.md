# Security Policy

## Supported versions

Only the latest release line (`0.1.x`) receives security fixes while the
workspace is pre-1.0.

## Memory safety posture

- The optimisation crates themselves contain **no `unsafe` code**. Any future
  `unsafe` block must be documented with a `// SAFETY:` comment explaining
  every invariant, and justified in review against this policy.
- The optional `highs` feature pulls in HiGHS C++ bindings via
  `highs-sys`/bindgen; that boundary is third-party code outside this
  workspace's control. It is non-default and can be avoided entirely by not
  enabling the feature.
- `cargo deny` enforces advisories (RustSec), a permissive-license allowlist,
  source restrictions (no unknown registries/git), and bans yanked/duplicate
  versions. CI fails on violations.

## Error handling convention

Solvers report failures as `Result`/typed status values (`OptError`,
`SolverStatus::NumericalIssue`, …) rather than panicking. Panics are treated
as bugs: please report any reachable panic on well-formed input as a defect.

## Reporting a vulnerability

Email **security@tpt.solutions** (or open a private GitHub security advisory
on this repository). Please include a minimal reproducer and affected crate
versions. We aim to acknowledge within 7 days and publish a fix or mitigation
within 90 days, crediting reporters by default.