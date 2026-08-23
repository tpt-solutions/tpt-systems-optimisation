# tpt-opt-cli

A small command-line front end for [`tpt-opt-milp`](../tpt-opt-milp): read a
free-format **MPS** or **CPLEX-LP** model file, solve it with the bundled
branch-and-cut solver, and print status, objective, and variable values.

## Usage

```text
tpt-opt-cli <model.(mps|lp)> [options]

options:
  --time-limit <secs>  wall-clock limit; reports the best incumbent on expiry
  --threads <n>        deterministic parallel tree search (n > 1)
  --seed <u64>         fix the heuristic/branching seed for reproducibility
  --cuts               enable the root cut suite (Gomory/L&P/clique/cover/MIR)
  --export <path>      convert the parsed model to `.mps`/`.lp` and exit
  -q, --quiet          print only status + objective
  -h, --help           this help
```

Examples:

```sh
tpt-opt-cli model.mps
tpt-opt-cli model.lp --time-limit 30 --threads 4 --seed 7 --cuts
tpt-opt-cli model.mps --export model.lp   # format conversion only
```

Format detection is by extension (`.mps` / `.lp`, case-insensitive), falling
back to content sniffing (`Minimize`/`Maximize` ⇒ LP, otherwise MPS).

Exit codes: `0` run completed (any terminal status counts as a completed run,
including infeasible/unbounded) · `1` I/O or parse failure · `2` usage error.

## Building

From the workspace root:

```sh
cargo build -p tpt-opt-cli
target/debug/tpt-opt-cli --help
```

## License

MIT OR Apache-2.0 (see LICENSE-MIT / LICENSE-APACHE).