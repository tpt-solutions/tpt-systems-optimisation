//! Workspace task runner for `tpt-systems-optimisation`.
//!
//! Invoked via the `cargo xtask` alias (see `.cargo/config.toml`):
//!
//! ```text
//! cargo xtask fmt      # rustfmt over the whole workspace
//! cargo xtask clippy   # clippy with -D warnings, all features
//! cargo xtask test     # unit/integration tests + doctests
//! cargo xtask deny     # cargo-deny (advisories/bans/licenses/sources)
//! cargo xtask no-std   # cross-check tpt-opt-core for thumbv6m-none-eabi
//! cargo xtask check    # fast compile check, all features
//! cargo xtask all      # everything above, in order
//! ```

use std::process::{Command, ExitCode};

const NO_STD_TARGET: &str = "thumbv6m-none-eabi";

const TASKS: &[(&str, &str)] = &[
    ("fmt", "rustfmt over the whole workspace"),
    ("clippy", "clippy --all-targets --all-features with -D warnings"),
    ("test", "tests + doctests (--all-features)"),
    ("deny", "cargo-deny advisories/bans/licenses/sources"),
    ("no-std", "cross-check tpt-opt-core for thumbv6m-none-eabi"),
    ("check", "fast compile check (--all-features)"),
    ("all", "run every task above in order"),
];

fn main() -> ExitCode {
    let task = std::env::args().nth(1).unwrap_or_else(|| "help".to_string());
    let ok = run(&task);
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn run(task: &str) -> bool {
    match task {
        "help" => {
            print_help();
            true
        }
        "fmt" => cargo(&["fmt", "--all"]),
        "clippy" => cargo(&[
            "clippy",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--",
            "-D",
            "warnings",
        ]),
        "test" => {
            cargo(&["test", "--workspace", "--all-features"])
                && cargo(&["test", "--workspace", "--all-features", "--doc"])
        }
        "deny" => cargo(&["deny", "check"]),
        "no-std" => {
            cargo(&[
                "check",
                "-p",
                "tpt-opt-core",
                "--target",
                NO_STD_TARGET,
                "--no-default-features",
            ]) && cargo(&[
                "check",
                "-p",
                "tpt-opt-core",
                "--target",
                NO_STD_TARGET,
                "--no-default-features",
                "--features",
                "alloc",
            ])
        }
        "check" => cargo(&["check", "--workspace", "--all-features"]),
        "all" => ["fmt", "clippy", "test", "deny", "no-std", "check"].iter().all(|t| run(t)),
        other => {
            eprintln!("unknown task: {other}");
            print_help();
            false
        }
    }
}

/// Run `cargo <args>` in the workspace root, inheriting stdio.
fn cargo(args: &[&str]) -> bool {
    println!("+ cargo {}", args.join(" "));
    Command::new("cargo").args(args).status().map(|s| s.success()).unwrap_or(false)
}

fn print_help() {
    println!(
        "usage: cargo xtask <task>

tasks:"
    );
    for (name, desc) in TASKS {
        println!("  {name:<8} {desc}");
    }
}
