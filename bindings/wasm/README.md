# tpt-opt-wasm

WebAssembly build of the bundled MILP solver, with a static demo page. The
model is solved in the browser, and no data leaves the page.

## Build

Requires the `wasm32-unknown-unknown` target and `wasm-pack`:

```text
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
wasm-pack build --target web --release --out-dir www/pkg
```

## Run the demo

The page loads an ES module, so serve the folder over HTTP rather than opening
the file directly:

```text
cd www
python -m http.server 8000     # then open http://localhost:8000
```

## API

```js
import init, { solve, version } from "./pkg/tpt_opt_wasm.js";
await init();
const json = solve(modelText, "lp", 5 /* seconds, optional */);
const { status, objective, primal } = JSON.parse(json);
```

`format` is `"lp"` or `"mps"`. Parse and solver failures are thrown as strings.

## Smoke test

After building into `www/pkg`, from this directory:

```text
node tests/smoke.mjs
```

## Notes

- Timing uses the `web-time` crate, so the solver's time limit works in the
  browser through `performance.now()`.
- `wasm-opt` is disabled in `Cargo.toml`, because the optimiser download needs
  network access.

## License

MIT OR Apache-2.0.
