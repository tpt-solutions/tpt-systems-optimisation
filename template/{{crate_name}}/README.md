# {{crate_name}}

{{description}}

Scaffolded from the `tpt-systems-optimisation` Tier 2 template with the
`tpt-opt-systems` feature subset: {{tpt_features}}.

## Quick start

See `src/lib.rs` for a compiling example using the always-available core
model types, and the [workspace README](https://github.com/tpt-solutions/tpt-systems-optimisation)
for per-domain feature recommendations and runnable examples.

## Quality gates

Mirror the workspace tooling:

```sh
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## License

MIT OR Apache-2.0.