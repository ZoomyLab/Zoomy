/* Source for the Pyodide Web Worker, embedded as a string and started from a
 * Blob URL (no separate asset to serve). Runs Python OFF the main thread so the
 * Theia UI stays responsive, and reuses the Zoomy GUI's proven tricks:
 *   - tiered installs fired at boot (zoomy-core blocking; jedi + zoomy-plotting
 *     + matplotlib in the background) so the kernel is warm before first use;
 *   - a parso AST cache on IDBFS so jedi cold-start (15-25 s parsing zoomy_core)
 *     becomes <1 s on the 2nd+ visit;
 *   - engine.py's `complete_code` (jedi + signature supplement) verbatim.
 */

// Python helpers: notebook-cell exec (last-expression display + matplotlib PNG +
// rich repr) and jedi autocomplete (copied from zoomy_gui/engine.py).
const PY_HELPERS = `
import sys, io, os, base64, json, ast
os.environ.setdefault("MPLBACKEND", "AGG")
__zoomy_ns__ = {"__name__": "__main__"}

def __zoomy_exec__(src):
    outs = []
    tree = ast.parse(src, mode="exec")
    val = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body.pop()
        exec(compile(tree, "<cell>", "exec"), __zoomy_ns__)
        val = eval(compile(ast.Expression(last.value), "<cell>", "eval"), __zoomy_ns__)
    else:
        exec(compile(tree, "<cell>", "exec"), __zoomy_ns__)
    if val is not None:
        rich = None
        for meth, mime in (("_repr_html_", "text/html"),
                           ("_repr_markdown_", "text/markdown"),
                           ("_repr_latex_", "text/latex")):
            fn = getattr(val, meth, None)
            if fn:
                try:
                    r = fn()
                    if r:
                        rich = {"mime": mime, "data": r}
                        break
                except Exception:
                    pass
        outs.append(rich if rich else {"mime": "text/plain", "data": repr(val)})
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None:
        for num in plt.get_fignums():
            buf = io.BytesIO()
            plt.figure(num).savefig(buf, format="png", dpi=110, bbox_inches="tight")
            outs.append({"mime": "image/png",
                         "data": base64.b64encode(buf.getvalue()).decode()})
        plt.close("all")
    return json.dumps(outs)

def complete_code(code, row, col, limit=80):
    try:
        import jedi
    except ImportError:
        return {"completions": [], "error": "jedi unavailable"}
    # Interpreter uses the live kernel namespace, so after a cell runs a name
    # completes from the actual object (jedi cannot statically infer members of
    # factory constructors like SME). Static Script is the fallback.
    try:
        try:
            script = jedi.Interpreter(code, [__zoomy_ns__])
        except Exception:
            script = jedi.Script(code)
        completions = script.complete(row, col)
    except Exception as e:
        return {"completions": [], "error": str(e)}
    out = []
    seen_names = set()
    def _key(name):
        return (name or "").rstrip("=")
    for c in completions[:limit]:
        k = _key(c.name)
        if k in seen_names:
            continue
        seen_names.add(k)
        sig_str = ""
        try:
            sigs = c.get_signatures()
            if sigs:
                sig_str = sigs[0].to_string()
        except Exception:
            pass
        doc = ""
        try:
            doc = c.docstring(raw=True) or ""
            if len(doc) > 2000:
                doc = doc[:2000] + " ..."
        except Exception:
            pass
        out.append({"name": c.name, "type": c.type, "signature": sig_str,
                    "docstring": doc, "module": getattr(c, "module_name", "") or ""})
    try:
        signatures = script.get_signatures(row, col)
    except Exception:
        signatures = []
    for sig in signatures:
        sig_str = sig.to_string()
        doc = ""
        try:
            doc = sig.docstring(raw=True) or ""
            if len(doc) > 2000:
                doc = doc[:2000] + " ..."
        except Exception:
            pass
        for param in sig.params:
            pname = (param.name or "").rstrip("=")
            if not pname or pname in seen_names:
                continue
            seen_names.add(pname)
            out.append({"name": pname + "=", "type": "param", "signature": sig_str,
                        "docstring": doc, "module": ""})
    return {"completions": out}
`;

export const PYODIDE_VERSION = 'v0.29.3';

// The worker JS. String.raw keeps regex backslashes (\b, \s) intact; the only
// substitutions are the pyodide URL and the JSON-encoded Python helpers.
export const WORKER_SOURCE = String.raw`
"use strict";
var py = null;
var _bootPromise = null, _jediPromise = null, _mplPromise = null, _zpPromise = null;
importScripts("https://cdn.jsdelivr.net/pyodide/${PYODIDE_VERSION}/full/pyodide.js");
var PY_HELPERS = ${JSON.stringify(PY_HELPERS)};
function log(msg) { postMessage({ type: "log", msg: msg }); }

async function boot() {
    if (_bootPromise) return _bootPromise;
    _bootPromise = (async function () {
        log("Booting Pyodide…");
        py = await loadPyodide({ stdout: function () {}, stderr: function () {} });
        await py.loadPackage("micropip");
        log("Installing zoomy-core…");
        try { await py.loadPackage(["h5py"]); } catch (e) {}
        var mp = py.pyimport("micropip");
        await mp.install(["zoomy-core"]);
        await py.runPythonAsync(PY_HELPERS);
        log("Kernel ready.");
    })();
    return _bootPromise;
}

var _parsoMounted = false;
async function mountParsoCache() {
    if (_parsoMounted) return;
    try {
        var path = "/home/pyodide/.cache/parso";
        py.FS.mkdirTree(path);
        py.FS.mount(py.FS.filesystems.IDBFS, {}, path);
        await new Promise(function (res, rej) { py.FS.syncfs(true, function (e) { e ? rej(e) : res(); }); });
        _parsoMounted = true;
    } catch (e) { log("parso cache unavailable: " + (e.message || e)); }
}
async function persistParsoCache() {
    if (!_parsoMounted) return;
    try { await new Promise(function (res, rej) { py.FS.syncfs(false, function (e) { e ? rej(e) : res(); }); }); }
    catch (e) {}
}

function installJedi() {
    if (_jediPromise) return _jediPromise;
    _jediPromise = (async function () {
        await boot();
        log("installing autocomplete…");
        try { var mp = py.pyimport("micropip"); await mp.install(["jedi"]); }
        catch (e) { log("jedi failed: " + (e.message || e)); return; }
        await mountParsoCache();
        try {
            var prime = ["from zoomy_core.model.models import SME", "model = SME(level=0)", "model."].join("\n");
            var res = py.globals.get("complete_code")(prime, 3, 6);
            if (res && res.destroy) res.destroy();
        } catch (e) { log("jedi prime failed: " + (e.message || e)); }
        await persistParsoCache();
        log("autocomplete ready");
    })();
    return _jediPromise;
}
function installMpl() {
    if (_mplPromise) return _mplPromise;
    _mplPromise = (async function () {
        await boot();
        try { await py.loadPackage(["matplotlib"]); await py.runPythonAsync("import matplotlib; matplotlib.use('AGG')"); log("matplotlib ready"); }
        catch (e) { log("matplotlib failed: " + (e.message || e)); }
    })();
    return _mplPromise;
}
function installZp() {
    if (_zpPromise) return _zpPromise;
    _zpPromise = (async function () {
        await boot();
        try { var mp = py.pyimport("micropip"); await mp.install(["zoomy-plotting"]); log("plotting ready"); }
        catch (e) { log("zoomy-plotting failed: " + (e.message || e)); }
    })();
    return _zpPromise;
}

var _MPL_RE = /\b(import\s+matplotlib|from\s+matplotlib|matplotlib\.)/;
var _ZP_RE  = /\b(zoomy_plotting|open_hdf5)\b/;
async function ensureDeps(code) {
    var needs = [];
    if (_MPL_RE.test(code)) needs.push(installMpl());
    if (_ZP_RE.test(code))  needs.push(installZp());
    if (needs.length) await Promise.all(needs);
}

async function runCell(code) {
    await boot();
    await ensureDeps(code);
    var outs = [];
    py.setStdout({ batched: function (s) { outs.push({ type: "stream", text: s + "\n" }); } });
    py.setStderr({ batched: function (s) { outs.push({ type: "stream", text: s + "\n" }); } });
    try {
        py.globals.set("__zoomy_src__", code);
        var j = await py.runPythonAsync("__zoomy_exec__(__zoomy_src__)");
        var rich = JSON.parse(j || "[]");
        for (var i = 0; i < rich.length; i++) { outs.push({ type: "data", mime: rich[i].mime, value: rich[i].data }); }
    } catch (e) {
        outs.push({ type: "error", ename: "PythonError", evalue: (e && e.message) || String(e) });
    }
    return outs;
}

async function complete(code, row, col) {
    await installJedi();
    var r = py.globals.get("complete_code")(code, row, col);
    var conv = r.toJs ? r.toJs({ dict_converter: Object.fromEntries }) : r;
    if (r.destroy) r.destroy();
    return conv;
}

function fireBackground() {
    Promise.all([installJedi(), installZp(), installMpl()]).then(function () { postMessage({ type: "background_ready" }); });
}

onmessage = async function (e) {
    var m = e.data;
    try {
        if (m.cmd === "warm") {
            await boot(); postMessage({ type: "ready", id: m.id }); fireBackground();
        } else if (m.cmd === "run") {
            var o = await runCell(m.code); postMessage({ type: "result", id: m.id, outputs: o });
        } else if (m.cmd === "complete") {
            var c = await complete(m.code, m.row, m.col); postMessage({ type: "result", id: m.id, data: c });
        }
    } catch (err) {
        postMessage({ type: "error", id: m.id, error: (err && err.message) || String(err) });
    }
};

// Auto-warm the moment the worker is created.
boot().then(function () { postMessage({ type: "ready" }); fireBackground(); })
      .catch(function (err) { postMessage({ type: "log", msg: "boot failed: " + (err && err.message || err) }); });
`;
