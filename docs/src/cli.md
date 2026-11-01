# Command-line tool

`tpt-opt-cli` reads MPS, CPLEX-LP, and OPB (pseudo-Boolean) files and works on
them from the shell. Format is chosen by extension, falling back to content
sniffing.

```text
tpt-opt-cli model.mps                       # solve (the default subcommand)
tpt-opt-cli solve model.lp --time-limit 30 --threads 4 --seed 7 --cuts
tpt-opt-cli info model.mps --format json
tpt-opt-cli classify model.lp
tpt-opt-cli convert model.mps model.lp
tpt-opt-cli iis infeasible.lp
tpt-opt-cli presolve model.mps --export reduced.lp
```

Use `--format json` on `solve`, `info`, `classify`, `iis`, and `presolve` to get
machine-readable output. `--out <path>` writes the solution values to a file.

Exit codes: `0` the command completed (including infeasible or unbounded
results), `1` I/O, parse, or solver failure, and `2` a usage error.

The full option list is in the crate README at
`crates/tpt-opt-cli/README.md`.
