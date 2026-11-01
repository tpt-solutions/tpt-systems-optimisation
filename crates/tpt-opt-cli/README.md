# tpt-opt-cli

A command-line front end for [`tpt-opt-milp`](../tpt-opt-milp). Read a
free-format **MPS** or **CPLEX-LP** model and inspect it, convert it, presolve
it, find an infeasibility certificate, or solve it.

## Usage

```text
tpt-opt-cli [solve] <model> [options]        solve (the default when no subcommand is given)
tpt-opt-cli info <model> [--format json]     dimensions, variable kinds, class
tpt-opt-cli classify <model>                 linear program or mixed-integer program
tpt-opt-cli convert <in> <out>               convert between .mps, .lp and .opb
tpt-opt-cli iis <model> [--format json]      irreducible infeasible subsystem
tpt-opt-cli presolve <model> [--export <p>]  presolve report, optionally write the reduced model
```

`solve` options:

```text
--time-limit <secs>     wall-clock limit; reports the best incumbent on expiry
--threads <n>           deterministic parallel tree search (n > 1)
--seed <u64>            fix the heuristic/branching seed for reproducibility
--cuts                  enable the root cut suite (MILP backend)
--gap <abs>             absolute optimality gap
--presolve              presolve before solving
--solver <milp|highs>   backend; `highs` needs a build with `--features highs`
--log                   print progress events to stderr
--export <path>         write the parsed model (.mps or .lp) and exit
--out <path>            also write the solution values to <path>
```

Common options: `--format <text|json>` (for `solve`, `info`, `classify`, `iis`,
and `presolve`), `-q`/`--quiet` (status and objective only), `-h`/`--help`.

Examples:

```sh
tpt-opt-cli model.mps
tpt-opt-cli solve model.lp --time-limit 30 --threads 4 --seed 7 --cuts
tpt-opt-cli solve model.lp --format json --out values.txt
tpt-opt-cli info model.mps --format json
tpt-opt-cli convert model.mps model.lp
tpt-opt-cli iis infeasible.lp
```

Format detection is by extension (`.mps`, `.lp`, `.opb`, case-insensitive), falling
back to content sniffing (`Minimize`/`Maximize` means LP, `min:`/`max:` means OPB,
otherwise MPS).
Output files for `convert` and `--export` also need one of those extensions.

Exit codes: `0` the command completed (any terminal solve status counts as a
completed run, including infeasible or unbounded) · `1` I/O, parse, or solver
failure · `2` usage error.

## HiGHS backend

`--solver highs` requires building with the `highs` feature:

```sh
cargo build -p tpt-opt-cli --features highs
```

That feature builds HiGHS from source, so it needs a C++ toolchain (cmake plus
MSVC, gcc, or clang).

## Building

From the workspace root:

```sh
cargo build -p tpt-opt-cli
target/debug/tpt-opt-cli --help
```

## License

MIT OR Apache-2.0 (see LICENSE-MIT / LICENSE-APACHE).
