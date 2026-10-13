// Headless Pyodide run of a QR case, booted like library/zoomy_gui/pyodide-worker.js.
//
//   node run.mjs --wheels WHEELS_DIR --case case.json --out report.json
//   node run.mjs --wheels WHEELS_DIR --script c1_bump.py --out report.json
//
// WHEELS_DIR is what docs/scripts/build_gui_wheels.py writes (index.json + the zoomy-core
// and zoomy-plotting wheels), i.e. what the deployed worker installs from gui/wheels/.
// Boot = the worker's: micropip, h5py, then micropip ["param", <zoomy-core wheel>],
// param_extract.py, the catalog param pre-extract, engine.py, then its tier-2 installs
// (<zoomy-plotting wheel>, matplotlib; jedi is autocomplete only and skipped). The lazy
// installs (plotly, meshio+rich) follow the worker's regexes, applied to all cells up
// front. Node has no XMLHttpRequest, which zoomy_core.misc.download uses in the browser's
// workers: a synchronous one is shimmed below. `QR_PYODIDE_DIR` points at an unpacked npm
// `pyodide` package, `QR_PYODIDE_CACHE` at a package cache directory.
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const QR = path.dirname(HERE);
const GUI = path.resolve(QR, "../../library/zoomy_gui");

const args = {};
for (let i = 2; i < process.argv.length; i += 2) args[process.argv[i].replace(/^--/, "")] = process.argv[i + 1];

const mod = process.env.QR_PYODIDE_DIR
    ? await import(pathToFileURL(path.join(process.env.QR_PYODIDE_DIR, "pyodide.mjs")).href)
    : await import("pyodide");
const log = (s) => process.stderr.write(s + "\n");

// Synchronous GET, as a browser Web Worker's XMLHttpRequest (no CORS in node).
const FETCH_JS = "const r = await fetch(process.argv[1]); process.stderr.write(String(r.status));"
    + " process.stdout.write(Buffer.from(await r.arrayBuffer()));";
globalThis.XMLHttpRequest = class {
    open(method, url) { this.url = url; }
    send() {
        const p = spawnSync(process.execPath, ["--input-type=module", "-e", FETCH_JS, this.url],
                            { maxBuffer: 1 << 30 });
        this.status = Number(String(p.stderr).trim().split(/\s+/).pop()) || 0;
        const b = p.stdout;
        this.response = this.responseType === "arraybuffer"
            ? b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) : b.toString("utf8");
    }
};

const t0 = Date.now();
const stamp = (msg) => log(`[${((Date.now() - t0) / 1000).toFixed(1)}s] ${msg}`);
const opts = { stdout: log, stderr: log };
if (process.env.QR_PYODIDE_CACHE) opts.packageCacheDir = process.env.QR_PYODIDE_CACHE;
const py = await mod.loadPyodide(opts);
stamp(`pyodide ${py.version}`);

await py.loadPackage("micropip");
try { await py.loadPackage(["h5py"]); } catch (e) { log("h5py failed: " + (e.message || e)); }
const mp = py.pyimport("micropip");
const index = JSON.parse(fs.readFileSync(path.join(args.wheels, "index.json"), "utf8"));
py.FS.mkdirTree("/tmp/wheels");
const wheel = (name) => {
    const whl = index[name].wheel;
    py.FS.writeFile("/tmp/wheels/" + whl, fs.readFileSync(path.join(args.wheels, whl)));
    return "emfs:/tmp/wheels/" + whl;
};
await mp.install(["param", wheel("zoomy-core")]);
stamp("zoomy-core " + index["zoomy-core"].wheel);
await py.runPythonAsync(fs.readFileSync(path.join(GUI, "param_extract.py"), "utf8"));
for (const tab of ["models", "solvers"]) {          // the worker's boot pre-extract
    for (const c of JSON.parse(fs.readFileSync(path.join(GUI, "cards", tab, "default.json"), "utf8"))) {
        if (!c.class) continue;
        try { py.globals.get("extract_param_schema")(c.class, py.toPy(c.init || {})); } catch (e) { /* as the worker */ }
    }
}
await py.runPythonAsync(fs.readFileSync(path.join(GUI, "engine.py"), "utf8"));
await mp.install([wheel("zoomy-plotting")]);
await py.loadPackage(["matplotlib"]);
stamp("engine + tier 2 ready");

py.FS.mkdirTree("/home/pyodide/qr");
for (const f of ["session_runner.py", "c1_bump.py"]) {
    py.FS.writeFile("/home/pyodide/qr/" + f, fs.readFileSync(path.join(QR, f), "utf8"));
}
await py.runPythonAsync("import sys; sys.path.insert(0, '/home/pyodide/qr')");

let report;
if (args.case) {
    const caseSpec = JSON.parse(fs.readFileSync(args.case, "utf8"));
    const code = caseSpec.cells.map((c) => c[1]).join("\n");
    if (/\b(import\s+plotly|from\s+plotly)/.test(code)) await mp.install(["plotly"]);
    if (/\b(import\s+meshio|from\s+meshio)/.test(code)) await mp.install(["meshio", "rich"]);
    // Diagnosis only (never in the tests): --stage f1,f2 puts host files into the case
    // directory, e.g. a mesh the case cannot fetch.
    py.FS.mkdirTree("/home/pyodide/qr_case");
    for (const f of (args.stage ? args.stage.split(",") : [])) {
        py.FS.writeFile("/home/pyodide/qr_case/" + path.basename(f), fs.readFileSync(f));
    }
    py.globals.set("_qr_case_json", JSON.stringify(caseSpec));
    report = await py.runPythonAsync(`
import json, sys
import session_runner as _R
_rep = _R.run_case(json.loads(_qr_case_json), process_code, "/home/pyodide/qr_case")
_rep["modules_loaded"] = sorted(m for m in ("pyamg", "mpi4py", "h5py", "matplotlib", "zoomy_plotting")
                                if m in sys.modules)
_rep["packages"] = _R.zoomy_packages()
json.dumps(_rep)
`);
} else {
    report = await py.runPythonAsync("import json, c1_bump; json.dumps(c1_bump.main())");
}
const rep = JSON.parse(report);
rep.pyodide = py.version;
rep.wall_seconds = (Date.now() - t0) / 1000;
fs.writeFileSync(args.out, JSON.stringify(rep, null, 1));
stamp("done");
process.exit(0);
