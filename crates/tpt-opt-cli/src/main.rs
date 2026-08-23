//! `tpt-opt-cli` — read an MPS or CPLEX-LP model, solve it with the bundled
//! `tpt-opt-milp` branch-and-cut solver, and print the solution.
//!
//! ```text
//! tpt-opt-cli model.mps
//! tpt-opt-cli model.lp --time-limit 30 --threads 4 --seed 7 --cuts
//! tpt-opt-cli model.mps --export model.lp   # format conversion only
//! ```
//!
//! Exit codes: `0` the run completed (any terminal status, including
//! infeasible/unbounded, is a successful *run*); `1` I/O or parse failure;
//! `2` usage error.

use std::process::ExitCode;

use tpt_opt_core::solver::{Solver, SolverStatus};
use tpt_opt_core::Model;
use tpt_opt_milp::{read_lp, read_mps, write_lp, write_mps, MilpSolver};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Mps,
    Lp,
}

struct Options {
    path: String,
    time_limit: Option<f64>,
    threads: usize,
    seed: Option<u64>,
    cuts: bool,
    export: Option<String>,
    quiet: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            path: String::new(),
            time_limit: None,
            threads: 1,
            seed: None,
            cuts: false,
            export: None,
            quiet: false,
        }
    }
}

fn main() -> ExitCode {
    let opts = match parse_args(std::env::args().skip(1).collect()) {
        Ok(Some(o)) => o,
        Ok(None) => return ExitCode::SUCCESS, // help printed
        Err(msg) => {
            eprintln!("error: {msg}");
            eprint_usage();
            return ExitCode::from(2);
        }
    };

    let text = match std::fs::read_to_string(&opts.path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: cannot read {}: {e}", opts.path);
            return ExitCode::FAILURE;
        }
    };

    let format = detect_format(&opts.path, &text);
    let parse = match format {
        Format::Mps => read_mps(&text),
        Format::Lp => read_lp(&text),
    };
    let model: Model = match parse {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: failed to parse {} as {format:?}: {e}", opts.path);
            return ExitCode::FAILURE;
        }
    };

    // Export-only mode: convert between formats without solving.
    if let Some(out_path) = &opts.export {
        let out_format = detect_format(out_path, "");
        let rendered = match out_format {
            Format::Mps => write_mps(&model),
            Format::Lp => write_lp(&model),
        };
        if let Err(e) = std::fs::write(out_path, rendered) {
            eprintln!("error: cannot write {out_path}: {e}");
            return ExitCode::FAILURE;
        }
        println!("wrote {out_path} ({out_format:?})");
        return ExitCode::SUCCESS;
    }

    let mut solver = MilpSolver::new();
    if let Some(secs) = opts.time_limit {
        solver = solver.with_time_limit(std::time::Duration::from_secs_f64(secs));
    }
    if opts.threads > 1 {
        solver = solver.with_threads(opts.threads);
    }
    if let Some(seed) = opts.seed {
        solver = solver.with_seed(seed);
    }
    if opts.cuts {
        solver = solver.with_cuts(true);
    }

    let started = std::time::Instant::now();
    let sol = match solver.solve(&model) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: solve failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let elapsed = started.elapsed();

    if !opts.quiet {
        println!(
            "model:   {} ({} vars, {} rows)",
            opts.path,
            model.num_vars,
            model.num_constraints()
        );
    }
    println!("status:  {}", status_name(sol.status));
    println!("objective: {}", fmt_num(sol.objective_value));
    if !opts.quiet && sol.status.has_solution() {
        println!("variables:");
        for (i, v) in sol.primal.iter().enumerate() {
            let name =
                model.variables.get(i).map(|_| format!("x{i}")).unwrap_or_else(|| format!("x{i}"));
            println!("  {name} = {}", fmt_num(*v));
        }
    }
    if !opts.quiet {
        if let Some(nodes) = sol.iterations {
            println!("nodes:   {nodes}");
        }
        println!("time:    {:.3}s", elapsed.as_secs_f64());
    }
    ExitCode::SUCCESS
}

fn status_name(s: SolverStatus) -> &'static str {
    match s {
        SolverStatus::Optimal => "optimal",
        SolverStatus::Infeasible => "infeasible",
        SolverStatus::Unbounded => "unbounded",
        SolverStatus::TimeLimit => "time limit (best solution shown)",
        SolverStatus::NumericalIssue => "numerical issue",
        SolverStatus::Error => "error",
    }
}

/// Trim trailing zeros for compact output (`3.0` prints as `3`, `2.5` as `2.5`).
fn fmt_num(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// Decide MPS vs LP by extension first, then by content sniffing.
fn detect_format(path: &str, content: &str) -> Format {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".mps") || lower.ends_with(".mps.gz") {
        return Format::Mps;
    }
    if lower.ends_with(".lp") || lower.ends_with(".cplex") || lower.ends_with(".lp.gz") {
        return Format::Lp;
    }
    sniff_format(content)
}

/// Content sniffing used when the extension is unknown (or in export mode,
/// where only the target extension is known and the source is irrelevant).
fn sniff_format(content: &str) -> Format {
    let head: String = content
        .lines()
        .take(20)
        .collect::<Vec<_>>()
        .join(
            "
",
        )
        .to_ascii_lowercase();
    if head.contains("minimize") || head.contains("maximise") || head.contains("maximize") {
        Format::Lp
    } else {
        // MPS section headers or a NAME line; default to MPS (the more
        // common interchange format for benchmark libraries).
        Format::Mps
    }
}

fn parse_args(args: Vec<String>) -> Result<Option<Options>, String> {
    let mut o = Options::default();
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => {
                print_help();
                return Ok(None);
            }
            "--quiet" | "-q" => o.quiet = true,
            "--cuts" => o.cuts = true,
            "--time-limit" => {
                let v = it.next().ok_or("--time-limit needs a value (seconds)")?;
                o.time_limit =
                    Some(v.parse::<f64>().map_err(|_| format!("invalid --time-limit `{v}`"))?);
                if o.time_limit.is_some_and(|t| !(t > 0.0)) {
                    return Err("--time-limit must be positive".into());
                }
            }
            "--threads" => {
                let v = it.next().ok_or("--threads needs a value")?;
                o.threads = v.parse::<usize>().map_err(|_| format!("invalid --threads `{v}`"))?;
                if o.threads == 0 {
                    return Err("--threads must be >= 1".into());
                }
            }
            "--seed" => {
                let v = it.next().ok_or("--seed needs a value")?;
                o.seed = Some(v.parse::<u64>().map_err(|_| format!("invalid --seed `{v}`"))?);
            }
            "--export" => {
                let v = it.next().ok_or("--export needs a destination path")?;
                o.export = Some(v);
            }
            other => {
                if other.starts_with('-') {
                    return Err(format!("unknown option `{other}`"));
                }
                if !o.path.is_empty() {
                    return Err(format!("unexpected extra argument `{other}`"));
                }
                o.path = other.to_string();
            }
        }
    }
    if o.path.is_empty() {
        return Err("missing model file".into());
    }
    Ok(Some(o))
}

fn eprint_usage() {
    eprintln!("usage: tpt-opt-cli <model.(mps|lp)> [options]");
}

fn print_help() {
    println!(
        "tpt-opt-cli — solve an MPS or CPLEX-LP model with tpt-opt-milp

usage:
  tpt-opt-cli <model.(mps|lp)> [options]

options:
  --time-limit <secs>  wall-clock limit; reports the best incumbent on expiry
  --threads <n>        deterministic parallel tree search (n > 1)
  --seed <u64>         fix the heuristic/branching seed for reproducibility
  --cuts               enable the root cut suite (Gomory/L&P/clique/cover/MIR)
  --export <path>      convert the parsed model to `.mps`/`.lp` and exit
  -q, --quiet          print only status + objective
  -h, --help           this help

exit codes: 0 run completed · 1 I/O or parse failure · 2 usage error"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_formats_by_extension_and_content() {
        assert_eq!(detect_format("a.MPS", ""), Format::Mps);
        assert_eq!(detect_format("b.lp", ""), Format::Lp);
        assert_eq!(
            detect_format(
                "c.txt",
                "Minimize
 obj: x
"
            ),
            Format::Lp
        );
        assert_eq!(
            detect_format(
                "d",
                "NAME          TEST
ROWS
"
            ),
            Format::Mps
        );
        assert_eq!(detect_format("e", ""), Format::Mps); // default
    }

    #[test]
    fn parses_options_and_rejects_bad_input() {
        let o = parse_args(vec!["m.mps".into(), "--threads".into(), "4".into(), "--cuts".into()])
            .unwrap()
            .unwrap();
        assert_eq!(o.path, "m.mps");
        assert_eq!(o.threads, 4);
        assert!(o.cuts);

        assert!(parse_args(vec![]).is_err());
        assert!(parse_args(vec!["--bogus".into()]).is_err());
        assert!(parse_args(vec!["m.mps".into(), "--time-limit".into(), "x".into()]).is_err());
        assert!(parse_args(vec!["m.mps".into(), "--time-limit".into(), "0".into()]).is_err());
        assert!(parse_args(vec!["m.mps".into(), "extra.mps".into()]).is_err());
        // Help is not an error and consumes nothing else.
        assert!(parse_args(vec!["--help".into()]).unwrap().is_none());
    }

    #[test]
    fn numbers_print_compactly() {
        assert_eq!(fmt_num(17.0), "17");
        assert_eq!(fmt_num(-10.5), "-10.5");
        assert_eq!(fmt_num(0.0), "0");
    }

    #[test]
    fn end_to_end_solve_via_parsed_mps() {
        // Tiny knapsack written in free MPS, solved through the same code
        // path the CLI uses (parse -> MilpSolver).
        let mps = "\
NAME          KNAP
OBJSENSE
    MAX
ROWS
 N  COST
 L  CAP
COLUMNS
    X         COST      5.0        CAP       3.0
    Y         COST      4.0        CAP       2.0
RHS
    R         CAP       5.0
BOUNDS
 BV BND       X
 BV BND       Y
ENDATA
";
        let m = read_mps(mps).expect("parse");
        let mut solver = MilpSolver::new();
        let sol = solver.solve(&m).expect("solve");
        assert_eq!(sol.status, SolverStatus::Optimal);
        assert!((sol.objective_value - 9.0).abs() < 1e-6, "both items fit: 5+4");
    }
}
