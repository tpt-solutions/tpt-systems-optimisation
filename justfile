# tpt-systems-optimisation — task recipes.
#
# All recipes shell out to `cargo xtask` (see xtask/ and .cargo/config.toml),
# so the logic lives in one place. Requires https://github.com/casey/just.

default:
    @just --list

# Format the whole workspace
fmt:
    cargo xtask fmt

# Clippy with -D warnings, all features
clippy:
    cargo xtask clippy

# Tests + doctests, all features
test:
    cargo xtask test

# cargo-deny (advisories / bans / licenses / sources)
deny:
    cargo xtask deny

# Cross-check tpt-opt-core for thumbv6m-none-eabi
no-std:
    cargo xtask no-std

# Fast compile check, all features
check:
    cargo xtask check

# Everything above, in order
all:
    cargo xtask all