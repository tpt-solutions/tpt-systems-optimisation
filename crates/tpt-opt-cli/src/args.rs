//! Command-line parsing.
//!
//! The first argument selects a subcommand. When it is not a known subcommand
//! the arguments are treated as `solve <model> [options]`, which keeps the
//! original `tpt-opt-cli model.mps` form working.

/// What the user asked the CLI to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    /// Solve the model and print the solution.
    Solve,
    /// Print model dimensions and structure.
    Info,
    /// Print the structural class (LP or MILP).
    Classify,
    /// Convert between MPS and CPLEX-LP files.
    Convert,
    /// Find an irreducible infeasible subsystem.
    Iis,
    /// Run presolve and report what it changed.
    Presolve,
}

/// How results are printed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Human-readable text.
    Text,
    /// One JSON object on stdout.
    Json,
}

/// Which solver backend `solve` uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// The bundled branch-and-cut solver (`tpt-opt-milp`).
    Milp,
    /// HiGHS, available when built with the `highs` feature.
    Highs,
}

/// Parsed command line.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// Subcommand to run.
    pub command: Command,
    /// Positional arguments: one model path, or input and output for `convert`.
    pub inputs: Vec<String>,
    /// Wall-clock limit in seconds.
    pub time_limit: Option<f64>,
    /// Worker threads for the parallel tree search.
    pub threads: usize,
    /// Seed for heuristics and branching.
    pub seed: Option<u64>,
    /// Enable the root cut suite (MILP backend).
    pub cuts: bool,
    /// Absolute optimality gap.
    pub gap: f64,
    /// Print progress events to stderr while solving.
    pub log: bool,
    /// Print only the status and objective.
    pub quiet: bool,
    /// Output format.
    pub format: OutputFormat,
    /// Solver backend.
    pub backend: Backend,
    /// Presolve the model before solving.
    pub presolve: bool,
    /// Write the (possibly presolved) model here instead of solving.
    pub export: Option<String>,
    /// Write the solution values to this file.
    pub out: Option<String>,
}

/// Result of parsing: run a command, or show the help text.
#[derive(Debug, Clone, PartialEq)]
pub enum Parsed {
    /// Run with these options.
    Run(Options),
    /// Print the help text and exit successfully.
    Help,
}

impl Options {
    fn new(command: Command) -> Self {
        Self {
            command,
            inputs: Vec::new(),
            time_limit: None,
            threads: 1,
            seed: None,
            cuts: false,
            gap: 0.0,
            log: false,
            quiet: false,
            format: OutputFormat::Text,
            backend: Backend::Milp,
            presolve: false,
            export: None,
            out: None,
        }
    }
}

/// Parse the arguments after the program name.
pub fn parse(args: &[String]) -> Result<Parsed, String> {
    let (command, rest) = match args.first().map(String::as_str) {
        Some("help" | "-h" | "--help") => return Ok(Parsed::Help),
        Some("solve") => (Command::Solve, &args[1..]),
        Some("info") => (Command::Info, &args[1..]),
        Some("classify") => (Command::Classify, &args[1..]),
        Some("convert") => (Command::Convert, &args[1..]),
        Some("iis") => (Command::Iis, &args[1..]),
        Some("presolve") => (Command::Presolve, &args[1..]),
        _ => (Command::Solve, args),
    };

    let mut o = Options::new(command);
    let mut it = rest.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(Parsed::Help),
            "-q" | "--quiet" => o.quiet = true,
            "--cuts" => o.cuts = true,
            "--log" => o.log = true,
            "--presolve" => o.presolve = true,
            "--time-limit" => {
                let t: f64 = number(&mut it, arg)?;
                if !(t > 0.0 && t.is_finite()) {
                    return Err("--time-limit must be a positive number".into());
                }
                o.time_limit = Some(t);
            }
            "--threads" => {
                o.threads = number(&mut it, arg)?;
                if o.threads == 0 {
                    return Err("--threads must be >= 1".into());
                }
            }
            "--seed" => o.seed = Some(number(&mut it, arg)?),
            "--gap" => {
                o.gap = number(&mut it, arg)?;
                if !(o.gap >= 0.0 && o.gap.is_finite()) {
                    return Err("--gap must be a non-negative number".into());
                }
            }
            "--format" => {
                o.format = match value(&mut it, arg)? {
                    "text" => OutputFormat::Text,
                    "json" => OutputFormat::Json,
                    other => return Err(format!("unknown --format `{other}` (text or json)")),
                };
            }
            "--solver" => {
                o.backend = match value(&mut it, arg)? {
                    "milp" => Backend::Milp,
                    "highs" if cfg!(feature = "highs") => Backend::Highs,
                    "highs" => {
                        return Err(
                            "this build has no HiGHS support (rebuild with --features highs)"
                                .into(),
                        )
                    }
                    other => return Err(format!("unknown --solver `{other}` (milp or highs)")),
                };
            }
            "--export" => o.export = Some(value(&mut it, arg)?.to_string()),
            "--out" => o.out = Some(value(&mut it, arg)?.to_string()),
            other if other.starts_with('-') => return Err(format!("unknown option `{other}`")),
            other => o.inputs.push(other.to_string()),
        }
    }

    let expected = if command == Command::Convert { 2 } else { 1 };
    match o.inputs.len() {
        0 => Err(match command {
            Command::Convert => "convert needs an input and an output file".into(),
            _ => "missing model file".into(),
        }),
        n if n == expected => Ok(Parsed::Run(o)),
        _ if command == Command::Convert => Err("convert takes exactly two files".into()),
        _ => Err(format!("unexpected extra argument `{}`", o.inputs[1])),
    }
}

fn value<'a>(it: &mut std::slice::Iter<'a, String>, flag: &str) -> Result<&'a str, String> {
    it.next().map(String::as_str).ok_or_else(|| format!("{flag} needs a value"))
}

fn number<T: std::str::FromStr>(
    it: &mut std::slice::Iter<'_, String>,
    flag: &str,
) -> Result<T, String> {
    let v = value(it, flag)?;
    v.parse::<T>().map_err(|_| format!("invalid {flag} `{v}`"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn run(v: &[&str]) -> Options {
        match parse(&args(v)).unwrap() {
            Parsed::Run(o) => o,
            Parsed::Help => panic!("expected a run"),
        }
    }

    #[test]
    fn bare_model_path_means_solve() {
        let o = run(&["m.mps", "--threads", "4", "--cuts"]);
        assert_eq!(o.command, Command::Solve);
        assert_eq!(o.inputs, vec!["m.mps"]);
        assert_eq!(o.threads, 4);
        assert!(o.cuts);
    }

    #[test]
    fn subcommands_are_recognised() {
        assert_eq!(run(&["info", "m.lp"]).command, Command::Info);
        assert_eq!(run(&["classify", "m.lp"]).command, Command::Classify);
        assert_eq!(run(&["iis", "m.lp"]).command, Command::Iis);
        assert_eq!(run(&["presolve", "m.lp"]).command, Command::Presolve);
        let c = run(&["convert", "a.mps", "b.lp"]);
        assert_eq!(c.command, Command::Convert);
        assert_eq!(c.inputs, vec!["a.mps", "b.lp"]);
    }

    #[test]
    fn json_format_and_gap_parse() {
        let o = run(&["info", "m.mps", "--format", "json", "--gap", "0.01"]);
        assert_eq!(o.format, OutputFormat::Json);
        assert!((o.gap - 0.01).abs() < 1e-12);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(parse(&args(&[])).is_err());
        assert!(parse(&args(&["--bogus", "m.mps"])).is_err());
        assert!(parse(&args(&["m.mps", "--time-limit", "x"])).is_err());
        assert!(parse(&args(&["m.mps", "--time-limit", "0"])).is_err());
        assert!(parse(&args(&["m.mps", "extra.mps"])).is_err());
        assert!(parse(&args(&["m.mps", "--format", "xml"])).is_err());
        assert!(parse(&args(&["m.mps", "--threads", "0"])).is_err());
        assert!(parse(&args(&["convert", "only.mps"])).is_err());
        assert!(parse(&args(&["convert", "a", "b", "c"])).is_err());
    }

    #[test]
    fn help_is_not_an_error() {
        assert_eq!(parse(&args(&["--help"])).unwrap(), Parsed::Help);
        assert_eq!(parse(&args(&["help"])).unwrap(), Parsed::Help);
        assert_eq!(parse(&args(&["m.mps", "-h"])).unwrap(), Parsed::Help);
    }
}
