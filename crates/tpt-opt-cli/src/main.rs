//! `tpt-opt-cli`: read an MPS or CPLEX-LP model and inspect, convert, presolve,
//! diagnose, or solve it with `tpt-opt-milp` (or HiGHS when built with the
//! `highs` feature).
//!
//! ```text
//! tpt-opt-cli model.mps                       # solve (the default subcommand)
//! tpt-opt-cli solve model.lp --time-limit 30 --threads 4 --seed 7 --cuts
//! tpt-opt-cli info model.mps --format json
//! tpt-opt-cli convert model.mps model.lp
//! tpt-opt-cli iis infeasible.lp
//! tpt-opt-cli presolve model.mps --export reduced.lp
//! ```
//!
//! Exit codes: `0` the command completed (any terminal solve status, including
//! infeasible or unbounded, is a completed run); `1` I/O, parse, or solver
//! failure; `2` usage error.

mod args;
mod output;

use std::process::ExitCode;
use std::time::Instant;

use tpt_opt_core::progress::{ProgressAction, ProgressEvent};
use tpt_opt_core::{Model, Solution, SolveParameters, Solver, Verbosity};
use tpt_opt_milp::{
    find_iis, presolve_model, read_lp, read_mps, read_opb, write_lp, write_mps, write_opb,
    MilpSolver,
};

use args::{Backend, Command, Options, OutputFormat, Parsed};
use output::{fmt_num, json_num, json_str};

const HELP: &str = "\
tpt-opt-cli — read, inspect, convert, presolve, diagnose, and solve MPS / CPLEX-LP models

usage:
  tpt-opt-cli [solve] <model> [options]        solve (default when no subcommand is given)
  tpt-opt-cli info <model> [--format json]     dimensions, variable kinds, class
  tpt-opt-cli classify <model>                 linear program or mixed-integer program
  tpt-opt-cli convert <in> <out>               convert between .mps, .lp and .opb
  tpt-opt-cli iis <model> [--format json]      irreducible infeasible subsystem
  tpt-opt-cli presolve <model> [--export <p>]  presolve report (optionally write the reduced model)

solve options:
  --time-limit <secs>  wall-clock limit; reports the best incumbent on expiry
  --threads <n>        deterministic parallel tree search (n > 1)
  --seed <u64>         fix the heuristic/branching seed for reproducibility
  --cuts               enable the root cut suite (MILP backend)
  --gap <abs>          absolute optimality gap
  --presolve           presolve before solving
  --solver <milp|highs>  backend (highs needs a build with --features highs)
  --log                print progress events to stderr
  --export <path>      write the parsed model to <path> (.mps or .lp) and exit
  --out <path>         also write the solution values to <path>

common options:
  --format <text|json> output format (info, classify, iis, presolve, solve)
  -q, --quiet          print only status and objective (text output)
  -h, --help           this help

exit codes: 0 run completed · 1 I/O, parse, or solver failure · 2 usage error";

enum Format {
    Mps,
    Lp,
    Opb,
}

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let opts = match args::parse(&raw) {
        Ok(Parsed::Run(o)) => o,
        Ok(Parsed::Help) => {
            println!("{HELP}");
            return ExitCode::SUCCESS;
        }
        Err(msg) => {
            eprintln!("error: {msg}");
            eprintln!("run `tpt-opt-cli --help` for usage");
            return ExitCode::from(2);
        }
    };

    match run(&opts) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn run(o: &Options) -> Result<(), String> {
    match o.command {
        Command::Convert => convert(&o.inputs[0], &o.inputs[1]),
        Command::Info => info(&load(&o.inputs[0])?, &o.inputs[0], o.format),
        Command::Classify => classify(&load(&o.inputs[0])?, o.format),
        Command::Iis => iis(&load(&o.inputs[0])?, o.format),
        Command::Presolve => presolve_command(o),
        Command::Solve => solve(o),
    }
}

/// Read and parse a model file, choosing the format by extension or content.
fn load(path: &str) -> Result<Model, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    match detect_format(path, &text) {
        Format::Mps => read_mps(&text).map_err(|e| format!("failed to parse {path} as MPS: {e}")),
        Format::Lp => read_lp(&text).map_err(|e| format!("failed to parse {path} as LP: {e}")),
        Format::Opb => read_opb(&text).map_err(|e| format!("failed to parse {path} as OPB: {e}")),
    }
}

/// Write `model` to `path`, choosing the format from the extension.
fn save(model: &Model, path: &str) -> Result<(), String> {
    let rendered = match extension_format(path) {
        Some(Format::Mps) => write_mps(model),
        Some(Format::Lp) => write_lp(model),
        Some(Format::Opb) => write_opb(model).map_err(|e| format!("cannot write {path}: {e}"))?,
        None => {
            return Err(format!("cannot tell the output format of {path}; use .mps, .lp or .opb"))
        }
    };
    std::fs::write(path, rendered).map_err(|e| format!("cannot write {path}: {e}"))
}

fn convert(input: &str, output: &str) -> Result<(), String> {
    let model = load(input)?;
    save(&model, output)?;
    println!("wrote {output}");
    Ok(())
}

fn info(model: &Model, path: &str, format: OutputFormat) -> Result<(), String> {
    let s = output::summarise(model);
    let class = output::class_text(model);
    let sense = output::sense_text(model.objective.sense);
    match format {
        OutputFormat::Text => {
            println!("file:        {path}");
            println!("sense:       {sense}");
            println!(
                "variables:   {} (continuous {}, integer {}, binary {}, semi-continuous {})",
                model.num_vars, s.continuous, s.integer, s.binary, s.semi_continuous
            );
            println!("constraints: {} ({} nonzeros)", model.num_constraints(), s.nonzeros);
            println!("class:       {class}");
        }
        OutputFormat::Json => println!(
            "{{\"file\":{},\"sense\":{},\"variables\":{},\"continuous\":{},\"integer\":{},\"binary\":{},\"semi_continuous\":{},\"constraints\":{},\"nonzeros\":{},\"class\":{}}}",
            json_str(path),
            json_str(sense),
            model.num_vars,
            s.continuous,
            s.integer,
            s.binary,
            s.semi_continuous,
            model.num_constraints(),
            s.nonzeros,
            json_str(class),
        ),
    }
    Ok(())
}

fn classify(model: &Model, format: OutputFormat) -> Result<(), String> {
    let class = output::class_text(model);
    match format {
        OutputFormat::Text => println!("{class}"),
        OutputFormat::Json => println!("{{\"class\":{}}}", json_str(class)),
    }
    Ok(())
}

fn iis(model: &Model, format: OutputFormat) -> Result<(), String> {
    let found = find_iis(model).map_err(|e| format!("IIS: {e}"))?;
    let rows: Vec<String> = found
        .constraints
        .iter()
        .map(|&i| {
            let c = &model.constraints[i];
            (i, output::row_text(&c.indices, &c.coeffs, c.lower, c.upper))
        })
        .map(|(i, text)| format!("c{i}: {text}"))
        .collect();
    match format {
        OutputFormat::Text => {
            println!("irreducible infeasible subsystem ({} constraints):", rows.len());
            for row in &rows {
                println!("  {row}");
            }
        }
        OutputFormat::Json => {
            let items: Vec<String> = found
                .constraints
                .iter()
                .zip(&rows)
                .map(|(i, row)| format!("{{\"index\":{i},\"row\":{}}}", json_str(row)))
                .collect();
            println!("{{\"constraints\":[{}]}}", items.join(","));
        }
    }
    Ok(())
}

fn presolve_command(o: &Options) -> Result<(), String> {
    let model = load(&o.inputs[0])?;
    let (reduced, report) = presolve_model(&model);
    match o.format {
        OutputFormat::Text => println!("{}", output::presolve_text(&report)),
        OutputFormat::Json => println!("{}", output::presolve_json(&report)),
    }
    if let Some(path) = &o.export {
        if report.infeasible {
            return Err("presolve proved the model infeasible; nothing written".into());
        }
        save(&reduced, path)?;
    }
    Ok(())
}

fn solve(o: &Options) -> Result<(), String> {
    let path = &o.inputs[0];
    let model = load(path)?;

    if let Some(out_path) = &o.export {
        save(&model, out_path)?;
        println!("wrote {out_path}");
        return Ok(());
    }

    let model = if o.presolve {
        let (reduced, report) = presolve_model(&model);
        if !o.quiet && o.format == OutputFormat::Text {
            println!(
                "presolve: {} rows removed, {} bounds tightened",
                report.removed_rows, report.tightened_bounds
            );
        }
        if report.infeasible {
            return report_status(o, &model, &infeasible_solution(), 0.0, path);
        }
        reduced
    } else {
        model
    };

    let params = SolveParameters {
        time_limit: o.time_limit,
        absolute_gap: o.gap,
        threads: o.threads,
        seed: o.seed,
        verbosity: if o.log { Verbosity::Verbose } else { Verbosity::Quiet },
        ..SolveParameters::defaults()
    };

    let started = Instant::now();
    let sol = solve_with(o, &model, &params)?;
    let elapsed = started.elapsed().as_secs_f64();
    report_status(o, &model, &sol, elapsed, path)
}

fn solve_with(o: &Options, model: &Model, params: &SolveParameters) -> Result<Solution, String> {
    match o.backend {
        Backend::Milp => {
            let mut solver = MilpSolver::new().with_cuts(o.cuts);
            if o.log {
                solver = solver.with_progress_callback(Box::new(log_progress));
            }
            solver.set_parameter(params).map_err(|e| e.to_string())?;
            solver.solve(model).map_err(|e| format!("solve failed: {e}"))
        }
        Backend::Highs => solve_highs(model, params),
    }
}

#[cfg(feature = "highs")]
fn solve_highs(model: &Model, params: &SolveParameters) -> Result<Solution, String> {
    let mut solver = tpt_opt_milp::HighsSolver::new();
    solver.set_parameter(params).map_err(|e| e.to_string())?;
    solver.solve(model).map_err(|e| format!("solve failed: {e}"))
}

#[cfg(not(feature = "highs"))]
fn solve_highs(_model: &Model, _params: &SolveParameters) -> Result<Solution, String> {
    Err("this build has no HiGHS support".into())
}

fn log_progress(ev: &ProgressEvent) -> ProgressAction {
    let inc = ev.incumbent.map_or("-".into(), fmt_num);
    let bound = ev.bound.map_or("-".into(), fmt_num);
    eprintln!(
        "[{:>8.3}s] nodes {:>8}  incumbent {:>14}  bound {:>14}",
        ev.elapsed.as_secs_f64(),
        ev.iterations,
        inc,
        bound
    );
    ProgressAction::Continue
}

/// Placeholder used when presolve already proved infeasibility.
fn infeasible_solution() -> Solution {
    Solution::new(Vec::new(), f64::NAN, tpt_opt_core::SolverStatus::Infeasible)
}

fn report_status(
    o: &Options,
    model: &Model,
    sol: &Solution,
    elapsed: f64,
    path: &str,
) -> Result<(), String> {
    if let Some(out_path) = &o.out {
        let mut body = String::new();
        for (i, v) in sol.primal.iter().enumerate() {
            body.push_str(&format!("x{i} {}\n", fmt_num(*v)));
        }
        std::fs::write(out_path, body).map_err(|e| format!("cannot write {out_path}: {e}"))?;
    }

    let shown_values = sol.status.has_solution();
    match o.format {
        OutputFormat::Text => {
            if !o.quiet {
                println!(
                    "model:   {path} ({} vars, {} rows)",
                    model.num_vars,
                    model.num_constraints()
                );
            }
            println!("status:  {}", output::status_text(sol.status));
            if sol.objective_value.is_finite() {
                println!("objective: {}", fmt_num(sol.objective_value));
            } else {
                println!("objective: -");
            }
            if !o.quiet && shown_values {
                println!("variables:");
                for (i, v) in sol.primal.iter().enumerate() {
                    println!("  x{i} = {}", fmt_num(*v));
                }
            }
            if !o.quiet {
                if let Some(nodes) = sol.iterations {
                    println!("nodes:   {nodes}");
                }
                println!("time:    {elapsed:.3}s");
            }
        }
        OutputFormat::Json => {
            let values: Vec<String> = if shown_values {
                sol.primal.iter().map(|v| json_num(*v)).collect()
            } else {
                Vec::new()
            };
            let nodes = sol.iterations.map_or("null".into(), |n| n.to_string());
            println!(
                "{{\"file\":{},\"status\":{},\"objective\":{},\"primal\":[{}],\"nodes\":{},\"time_seconds\":{}}}",
                json_str(path),
                json_str(output::status_key(sol.status)),
                json_num(sol.objective_value),
                values.join(","),
                nodes,
                json_num(elapsed),
            );
        }
    }
    Ok(())
}

fn extension_format(path: &str) -> Option<Format> {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".mps") || lower.ends_with(".mps.gz") {
        Some(Format::Mps)
    } else if lower.ends_with(".lp") || lower.ends_with(".cplex") || lower.ends_with(".lp.gz") {
        Some(Format::Lp)
    } else if lower.ends_with(".opb") {
        Some(Format::Opb)
    } else {
        None
    }
}

/// Decide MPS vs LP by extension first, then by content sniffing.
fn detect_format(path: &str, content: &str) -> Format {
    extension_format(path).unwrap_or_else(|| sniff_format(content))
}

/// Content sniffing used when the extension is unknown.
fn sniff_format(content: &str) -> Format {
    let head: String = content.lines().take(20).collect::<Vec<_>>().join("\n").to_ascii_lowercase();
    if head.contains("#variable=")
        || head.contains("\nmin:")
        || head.contains("\nmax:")
        || head.starts_with("min:")
        || head.starts_with("max:")
    {
        Format::Opb
    } else if head.contains("minimize") || head.contains("maximise") || head.contains("maximize") {
        Format::Lp
    } else {
        // MPS section headers or a NAME line; MPS is the more common interchange format.
        Format::Mps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_formats_by_extension_and_content() {
        assert!(matches!(detect_format("a.MPS", ""), Format::Mps));
        assert!(matches!(detect_format("b.lp", ""), Format::Lp));
        assert!(matches!(detect_format("c.txt", "Minimize\n obj: x\n"), Format::Lp));
        assert!(matches!(detect_format("d", "NAME          TEST\nROWS\n"), Format::Mps));
        assert!(matches!(detect_format("e", ""), Format::Mps));
    }

    #[test]
    fn json_strings_are_escaped() {
        assert_eq!(json_str("a\"b\\c\n"), "\"a\\\"b\\\\c\\n\"");
        assert_eq!(json_str("\u{1}"), "\"\\u0001\"");
    }

    #[test]
    fn non_finite_numbers_become_null() {
        assert_eq!(json_num(f64::INFINITY), "null");
        assert_eq!(json_num(2.5), "2.5");
    }

    #[test]
    fn end_to_end_solve_via_parsed_mps() {
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
        let sol = MilpSolver::new().solve(&m).expect("solve");
        assert_eq!(sol.status, tpt_opt_core::SolverStatus::Optimal);
        assert!((sol.objective_value - 9.0).abs() < 1e-6, "both items fit: 5+4");
    }
}
