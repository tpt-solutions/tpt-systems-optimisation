<!--
PRs map to a tracked issue (issues-first workflow — see CONTRIBUTING.md).
Reference it below. Internal or external, the same quality gates apply.
-->

## Summary

<!-- What changes, and why? Link the tracked issue: "Closes #N". -->

Closes #

## Checklist

Mirror of the per-crate checklist in `todo.md` — tick what applies.

- [ ] Scope implemented (or change confined to an existing module)
- [ ] Unit tests + doctests added/updated
- [ ] Rustdoc updated (crate-level + public API)
- [ ] `cargo xtask fmt` clean
- [ ] `cargo xtask clippy` clean (`--all-targets --all-features -- -D warnings`)
- [ ] `cargo xtask test` passes (`--all-features`, includes doctests)
- [ ] `cargo xtask deny` clean (advisories / bans / licenses / sources)
- [ ] no_std target still builds if `tpt-opt-core` was touched
      (`cargo xtask no-std`)
- [ ] README.md + CHANGELOG.md updated (per-crate, Keep-a-Changelog)
- [ ] crates.io metadata intact if `Cargo.toml` changed
      (description / keywords / categories / readme / documentation)
- [ ] Determinism preserved: any new randomness is seeded
- [ ] No `unsafe`; errors returned as `Result`/typed statuses, not panics
- [ ] Numeric comparisons routed through `Tolerances` where applicable

## Verification

<!-- Paste the tail of your local gate run, e.g. `cargo xtask all` output. -->

```text
(paste output here)
```
