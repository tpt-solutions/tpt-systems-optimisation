// Smoke test for the WebAssembly build, run with Node 18+:
//   node tests/smoke.mjs     (from bindings/wasm, after `wasm-pack build --target web --out-dir www/pkg`)
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";
import init, { solve, version } from "../www/pkg/tpt_opt_wasm.js";

const wasm = await readFile(fileURLToPath(new URL("../www/pkg/tpt_opt_wasm_bg.wasm", import.meta.url)));
await init({ module_or_path: wasm });

const knapsack = `Maximize
 obj: 5 x + 4 y + 3 z
Subject To
 weight: 4 x + 3 y + 2 z <= 7
Bounds
 0 <= x <= 1
 0 <= y <= 1
 0 <= z <= 1
General
 x y z
End
`;

const result = JSON.parse(solve(knapsack, "lp", 5));
assert.equal(result.status, "optimal");
assert.ok(Math.abs(result.objective - 9) < 1e-6, `objective ${result.objective}`);
assert.equal(result.variables, 3);

// A time limit exercises the wasm clock (web-time) inside the solver.
const timed = JSON.parse(solve(knapsack, "lp", 0.5));
assert.equal(timed.status, "optimal");

const infeasible = `Minimize
 obj: x
Subject To
 r: x >= 3
Bounds
 x <= 1
End
`;
assert.equal(JSON.parse(solve(infeasible, "lp", undefined)).status, "infeasible");

assert.throws(() => solve("not a model", "lp", undefined), /LP:/);
assert.throws(() => solve(knapsack, "xml", undefined), /unknown format/);

assert.match(version(), /^\d+\.\d+\.\d+$/);
console.log("wasm smoke test passed:", result);
