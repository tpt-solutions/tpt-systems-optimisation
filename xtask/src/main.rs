//! Workspace task runner for `tpt-systems-optimisation`.
//!
//! Invoked via the `cargo xtask` alias (see `.cargo/config.toml`):
//!
//! ```text
//! cargo xtask fmt              # rustfmt over the whole workspace
//! cargo xtask clippy           # clippy with -D warnings, all features
//! cargo xtask test             # unit/integration tests + doctests
//! cargo xtask deny             # cargo-deny (advisories/bans/licenses/sources)
//! cargo xtask no-std           # cross-check tpt-opt-core for thumbv6m-none-eabi
//! cargo xtask check            # fast compile check, all features
//! cargo xtask all              # everything above, in order
//! cargo xtask new-crate <name> # scaffold a new tpt-opt-* crate
//! cargo xtask fetch-fixtures   # download benchmark corpora into tests/fixtures/
//! ```

use std::fs;
use std::process::{Command, ExitCode};

const NO_STD_TARGET: &str = "thumbv6m-none-eabi";

const TASKS: &[(&str, &str)] = &[
    ("fmt", "rustfmt over the whole workspace"),
    ("clippy", "clippy --all-targets --all-features with -D warnings"),
    ("test", "tests + doctests (--all-features)"),
    ("deny", "cargo-deny advisories/bans/licenses/sources"),
    ("no-std", "cross-check tpt-opt-core for thumbv6m-none-eabi"),
    ("check", "fast compile check (--all-features)"),
    ("new-crate", "scaffold a new tpt-opt-* crate: new-crate <name>"),
    ("fetch-fixtures", "download benchmark fixtures: fetch-fixtures [--list] [--force] [suite...]"),
    ("all", "run every task above in order"),
];

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let task = args.next().unwrap_or_else(|| "help".to_string());
    let ok = match task.as_str() {
        "new-crate" => match args.next() {
            Some(name) => new_crate(&name),
            None => {
                eprintln!("usage: cargo xtask new-crate <name>   (e.g. tpt-opt-scheduling)");
                false
            }
        },
        "fetch-fixtures" => {
            let mut suites: Vec<String> = Vec::new();
            let mut force = false;
            let mut list = false;
            for a in args {
                match a.as_str() {
                    "--force" => force = true,
                    "--list" => list = true,
                    s => suites.push(s.to_string()),
                }
            }
            fetch_fixtures(&suites, force, list)
        }
        other => run(other),
    };
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
        println!("  {name:<10} {desc}");
    }
}

// ---------------------------------------------------------------------------
// Benchmark fixture fetching (kept OUTSIDE every crate directory so the
// downloads never enter a packaged `.crate` file; see Open Risks in todo.md)
// ---------------------------------------------------------------------------

/// One downloadable benchmark instance.
struct Fixture {
    /// Suite name (subdirectory under `tests/fixtures/`).
    suite: &'static str,
    /// Instance name (file stem).
    name: &'static str,
    /// Download URL.
    url: &'static str,
    /// `true` when the URL serves gzip-compressed content that must be
    /// decompressed after download (via the system `tar`, which understands
    /// single-file gzip archives on Windows 10+/Linux/macOS alike).
    gzipped: bool,
}

/// Curated small instances per benchmark family. Everything here is a plain
/// HTTP(S) download so the step works without credentials or logins.
///
/// Sources: the HiGHS project's check-instance mirror hosts expanded
/// plain-text MPS copies of classic Netlib LP and MIPLIB MILP instances
/// (netlib.org's canonical distribution uses a packed fixed format no
/// free-format reader can ingest, and miplib.zib.de gates downloads behind
/// a session — hence this mirror).
const FIXTURES: &[Fixture] = &[
    // Netlib LP collection (classic continuous test set).
    Fixture {
        suite: "netlib",
        name: "afiro",
        url: "https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances/afiro.mps",
        gzipped: false,
    },
    Fixture {
        suite: "netlib",
        name: "adlittle",
        url:
            "https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances/adlittle.mps",
        gzipped: false,
    },
    Fixture {
        suite: "netlib",
        name: "e226",
        url: "https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances/e226.mps",
        gzipped: false,
    },
    // MIPLIB classics (mixed-integer test set).
    Fixture {
        suite: "miplib",
        name: "flugpl",
        url: "https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances/flugpl.mps",
        gzipped: false,
    },
    Fixture {
        suite: "miplib",
        name: "gt2",
        url: "https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances/gt2.mps",
        gzipped: false,
    },
    Fixture {
        suite: "miplib",
        name: "bell5",
        url: "https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances/bell5.mps",
        gzipped: false,
    },
    Fixture {
        suite: "miplib",
        name: "egout",
        url: "https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances/egout.mps",
        gzipped: false,
    },
];

const FIXTURES_ROOT: &str = "tests/fixtures";

fn fetch_fixtures(suites: &[String], force: bool, list: bool) -> bool {
    if list {
        println!("available fixtures:");
        for f in FIXTURES {
            println!("  {:<8} {:<10} {}", f.suite, f.name, f.url);
        }
        println!("\nusage: cargo xtask fetch-fixtures [--force] [suite ...]  (no suites = all)");
        return true;
    }

    let selected: Vec<&Fixture> = FIXTURES
        .iter()
        .filter(|f| suites.is_empty() || suites.iter().any(|s| s == f.suite || s == f.name))
        .collect();
    if selected.is_empty() {
        eprintln!(
            "no fixtures match {:?}; run `cargo xtask fetch-fixtures --list` for options",
            suites
        );
        return false;
    }

    let root = std::path::Path::new(FIXTURES_ROOT);
    let mut ok = true;
    for f in &selected {
        let dir = root.join(f.suite);
        if let Err(e) = fs::create_dir_all(&dir) {
            eprintln!("create {}: {e}", dir.display());
            ok = false;
            continue;
        }
        let dest = dir.join(format!("{}.mps", f.name));
        if dest.exists() && !force {
            println!("= cached {}", dest.display());
            continue;
        }
        print!("+ fetching {} -> {} ... ", f.url, dest.display());
        if !download_fixture(f, &dest) {
            println!("FAILED");
            ok = false;
        } else {
            println!("ok");
        }
    }

    // Keep the downloaded corpus out of version control and out of any
    // packaged crate (the directory lives at the repo root, outside crates/).
    ignore_fixtures_dir();

    if ok {
        println!(
            "\nfixtures ready under {FIXTURES_ROOT}/. Integration tests may reference them via \
             relative paths; they are gitignored and excluded from packaging by construction."
        );
    } else {
        eprintln!("\nsome fixtures failed; check network access and retry with --force");
    }
    ok
}

/// Download one fixture (decompressing gzip payloads through system `tar`)
/// and sanity-check the result looks like an MPS model.
fn download_fixture(f: &Fixture, dest: &std::path::Path) -> bool {
    let tmp = dest.with_extension("part");
    let out =
        Command::new("curl").args(["-fsSL", "--retry", "2", "-o"]).arg(&tmp).arg(f.url).output();
    match out {
        Ok(o) if o.status.success() => {}
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            eprintln!();
            eprintln!("  curl failed (exit {:?}): {}", o.status.code(), stderr.trim());
            let _ = fs::remove_file(&tmp);
            return false;
        }
        Err(e) => {
            eprintln!();
            eprintln!("  could not spawn curl: {e} (is curl.exe on PATH?)");
            return false;
        }
    }

    if f.gzipped {
        // `tar -xzf <file> -O` streams the decompressed content to stdout.
        let out = Command::new("tar").args(["-xzf"]).arg(&tmp).arg("-O").output();
        let _ = fs::remove_file(&tmp);
        match out {
            Ok(o) if o.status.success() && !o.stdout.is_empty() => {
                if fs::write(dest, &o.stdout).is_err() {
                    return false;
                }
            }
            Ok(o) => {
                eprintln!();
                eprintln!(
                    "  tar decompression failed: {}",
                    String::from_utf8_lossy(&o.stderr).trim()
                );
                return false;
            }
            Err(e) => {
                eprintln!();
                eprintln!("  could not spawn tar: {e}");
                return false;
            }
        }
    } else if fs::rename(&tmp, dest).is_err() {
        let _ = fs::remove_file(&tmp);
        return false;
    }

    // Sanity check: MPS models carry a ROWS section.
    match fs::read_to_string(dest) {
        Ok(text) => text.to_ascii_uppercase().contains("ROWS"),
        Err(_) => false,
    }
}

/// Ensure `/tests/fixtures/` is listed in the root `.gitignore`.
fn ignore_fixtures_dir() {
    const MARKER: &str = "/tests/fixtures/";
    let path = std::path::Path::new(".gitignore");
    let existing = fs::read_to_string(path).unwrap_or_default();
    if existing.lines().any(|l| l.trim() == MARKER) {
        return;
    }
    let mut out = existing.clone();
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(&format!("{MARKER} # downloaded benchmark corpus (cargo xtask fetch-fixtures)\n"));
    let _ = fs::write(path, out);
    println!("+ added {MARKER} to .gitignore");
}

// ---------------------------------------------------------------------------
// new-crate scaffolding (Per-Crate Checklist Template steps 1-2, 9)
// ---------------------------------------------------------------------------

/// Scaffold `crates/<name>/` from the per-crate checklist template: Cargo.toml
/// inheriting the workspace fields, a documented `lib.rs` stub, README,
/// Keep-a-Changelog CHANGELOG, and copies of both license files. The remaining
/// template steps (implement scope, tests, fmt/clippy/deny gates, crates.io
/// metadata polish, name reservation, package audit, publish dry-run) are
/// printed as a reminder list.
fn new_crate(name: &str) -> bool {
    if !name.starts_with("tpt-opt-") || name.len() <= "tpt-opt-".len() {
        eprintln!("crate name must start with `tpt-opt-` (got `{name}`)");
        return false;
    }
    if !name["tpt-opt-".len()..].chars().all(|c| c.is_ascii_lowercase() || c == '-') {
        eprintln!("crate name suffix must be lowercase ascii letters/hyphens (got `{name}`)");
        return false;
    }
    let dir = std::path::Path::new("crates").join(name);
    if dir.exists() {
        eprintln!("{} already exists", dir.display());
        return false;
    }

    let short = &name["tpt-opt-".len()..];
    let files: Vec<(std::path::PathBuf, String)> = vec![
        (
            dir.join("Cargo.toml"),
            format!(
                r#"[package]
name = "{name}"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
authors.workspace = true
repository.workspace = true
homepage.workspace = true
readme = "README.md"
description = "TODO: one-line description of {name} for crates.io."
keywords = ["optimization", "{short}"]
categories = ["science::math", "algorithms"]
documentation = "https://docs.rs/{name}"

[dependencies]
    tpt-opt-core = {{ workspace = true }}

[features]
default = ["std"]
std = []
"#
            ),
        ),
        (
            dir.join("src").join("lib.rs"),
            format!(
                r#"//! TODO: crate-level overview for `{name}`.
//!
//! Describe what this solver family does, which spec section it implements,
//! and link the sibling crates it builds on (`tpt-opt-core`, ...).
//!
//! # Examples
//!
//! ```
//! // TODO: a doctest that compiles once real functionality lands.
//! assert_eq!(2 + 2, 4);
//! ```

// Implementation goes here. Conventions (see CONTRIBUTING.md):
// - Result-returning APIs; never panic on bad input.
// - Route numeric tolerances through `tpt_opt_core::Tolerances`.
// - Seedable determinism for anything randomised.
#[cfg(test)]
mod tests {{
    #[test]
    fn scaffold_smoke() {{
        assert!(true);
    }}
}}
"#
            ),
        ),
        (
            dir.join("README.md"),
            format!(
                "# {name}\n\nTODO: what this crate provides, its place in the\n`tpt-systems-optimisation` workspace, and a quick-start snippet.\n\n## Status\n\nScaffolded via `cargo xtask new-crate`; implementation pending.\n\n## License\n\nMIT OR Apache-2.0 (see LICENSE-MIT / LICENSE-APACHE).\n"
            ),
        ),
        (
            dir.join("CHANGELOG.md"),
            format!(
                "# Changelog\n\nAll notable changes to this crate are documented in this file.\nFormat based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);\nthis project adheres to [Semantic Versioning](https://semver.org/).\n\n## [Unreleased]\n\n### Added\n\n- Initial scaffold generated by `cargo xtask new-crate`.\n"
            ),
        ),
    ];

    for (path, content) in &files {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .unwrap_or_else(|e| panic!("create {}: {e}", parent.display()));
        }
        fs::write(path, content).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        println!("+ created {}", path.display());
    }

    // License copies from the repo root (template step 9/12 expectation:
    // every packaged crate carries both license texts).
    for lic in ["LICENSE-MIT", "LICENSE-APACHE"] {
        let src = std::path::Path::new(lic);
        let dst = dir.join(lic);
        match fs::copy(src, &dst) {
            Ok(_) => println!("+ copied {} -> {}", src.display(), dst.display()),
            Err(e) => {
                eprintln!("could not copy {lic}: {e}; add it manually before packaging");
                return false;
            }
        }
    }

    println!(
        "\nScaffold complete. Remaining per-crate checklist steps:\n\
         3. Implement scope in src/\n\
         4. Unit tests + doctests\n\
         5. Rustdoc (crate-level + public API)\n\
         6. cargo xtask fmt && cargo xtask clippy\n\
         7. cargo xtask deny\n\
         8. no_std verification (only if the spec marks this crate no_std)\n\
         9. Polish README.md + CHANGELOG.md\n\
         10. Finalise crates.io metadata (description/keywords/categories)\n\
         11. Confirm name availability on crates.io\n\
         12. cargo package -p {name} --list\n\
         13. cargo publish --dry-run -p {name}"
    );
    true
}
