/* Pyodide Web Worker — runs Python in a background thread, never blocks UI */

var py = null;

importScripts("https://cdn.jsdelivr.net/pyodide/v0.29.3/full/pyodide.js");

/* Silence the chatty "Loading <pkg> from CDN" stdout that Pyodide emits
   during loadPackage. We still capture unexpected stderr. */
var _origConsoleLog = self.console.log;
self.console.log = function () { /* drop pyodide package-download chatter */ };

/* SharedArrayBuffer-backed interrupt wiring. The main thread posts a
   SAB via the "set_interrupt_buffer" command; Pyodide's setInterruptBuffer
   polls it between bytecodes and raises KeyboardInterrupt when the main
   thread writes 2. If the buffer arrives before Pyodide has finished
   loading we stash it and wire it up at the end of initPyodide. */
var _pendingInterruptBuffer = null;

async function initPyodide() {
    if (py) return py;
    postMessage({ type: "log", level: "info", msg: "Booting Pyodide…" });
    py = await loadPyodide({ stdout: function () {}, stderr: function () {} });
    await py.loadPackage("micropip");
    if (_pendingInterruptBuffer) {
        try {
            py.setInterruptBuffer(new Uint8Array(_pendingInterruptBuffer));
            postMessage({ type: "log", level: "info", msg: "Cooperative interrupt enabled (SAB)" });
        } catch (e) {
            postMessage({ type: "log", level: "warn", msg: "setInterruptBuffer failed: " + (e.message || e) });
        }
        _pendingInterruptBuffer = null;
    }
    return py;
}

/* Each install phase caches its Promise so concurrent callers share the same
   in-flight work instead of racing to install the same packages twice. */
var _paramPromise = null;
var _execPromise = null;
var _mplPromise = null;
var _plotlyPromise = null;

function installParam() {
    if (_paramPromise) return _paramPromise;
    _paramPromise = (async function () {
        await initPyodide();
        /* h5py must load BEFORE zoomy_core is imported. zoomy_core.mesh.base_mesh
           does a try/except import of h5py at module scope and caches
           _HAVE_H5PY; if h5py isn't available at that first import, later
           write_to_hdf5 / from_hdf5 calls will RuntimeError even after the
           package loads. The pre-extract loop imports zoomy_core, so h5py
           must land inside installParam, not later. */
        try {
            await py.loadPackage(["h5py"]);
        } catch (e) {
            postMessage({ type: "log", level: "warn", msg: "h5py failed: " + (e.message || e) });
        }
        var mp = py.pyimport("micropip");
        await mp.install(["param", "zoomy-core"]);
        var code = await fetch("param_extract.py").then(function (r) { return r.text(); });
        await py.runPythonAsync(code);
    })();
    return _paramPromise;
}

function installExec() {
    if (_execPromise) return _execPromise;
    _execPromise = (async function () {
        await installParam();
        /* zoomy-plotting is no longer loaded here — it moved to tier 2
           (background). The engine.py function open_hdf5() imports it
           lazily, and any run_code that touches the HDF5 path waits on
           installZoomyPlotting() through ensureVizDeps(). */
        var code = await fetch("engine.py").then(function (r) { return r.text(); });
        await py.runPythonAsync(code);

        /* Register display callback: funnels rich output to main thread */
        self._zoomyDisplayBridge = function (cellJson) {
            postMessage({ type: "display", cell: cellJson });
        };
        await py.runPythonAsync([
            "import sys, json as _json",
            "from js import _zoomyDisplayBridge",
            "def _zoomy_display_cb(cell):",
            "    _zoomyDisplayBridge(_json.dumps(cell))",
            "sys._zoomy_display_callback = _zoomy_display_cb"
        ].join("\n"));
    })();
    return _execPromise;
}

/* Lazy per-library installers. Triggered by run_code when the user's snippet
   actually references the library. Each is promise-guarded so concurrent
   viz refreshes don't try to install twice. */
function installMatplotlib() {
    if (_mplPromise) return _mplPromise;
    _mplPromise = (async function () {
        postMessage({ type: "log", level: "info", msg: "installing matplotlib" });
        try {
            await py.loadPackage(["matplotlib"]);
            postMessage({ type: "log", level: "info", msg: "matplotlib ready" });
        } catch (e) {
            postMessage({ type: "log", level: "warn", msg: "matplotlib failed: " + (e.message || e) });
        }
    })();
    return _mplPromise;
}

function installPlotly() {
    if (_plotlyPromise) return _plotlyPromise;
    _plotlyPromise = (async function () {
        postMessage({ type: "log", level: "info", msg: "installing plotly" });
        try {
            var mp = py.pyimport("micropip");
            await mp.install(["plotly"]);
            postMessage({ type: "log", level: "info", msg: "plotly ready" });
        } catch (e) {
            postMessage({ type: "log", level: "warn", msg: "plotly failed: " + (e.message || e) });
        }
    })();
    return _plotlyPromise;
}

var _meshioPromise = null;
/* Lazy install for meshio — only needed when a user uploads a .msh
   file and the generated mesh-card snippet runs. Pure-Python wheel (~1 MB)
   so we don't pay it unless someone actually touches the path. */
function installMeshio() {
    if (_meshioPromise) return _meshioPromise;
    _meshioPromise = (async function () {
        postMessage({ type: "log", level: "info", msg: "installing meshio" });
        try {
            var mp = py.pyimport("micropip");
            /* meshio's submodules import `rich` unconditionally (for
               pretty CLI output), so it has to come along for the ride
               even when we only care about `meshio.read`. */
            await mp.install(["meshio", "rich"]);
            postMessage({ type: "log", level: "info", msg: "meshio ready" });
        } catch (e) {
            postMessage({ type: "log", level: "warn", msg: "meshio failed: " + (e.message || e) });
            throw e;
        }
    })();
    return _meshioPromise;
}

var _zpPromise = null;
function installZoomyPlotting() {
    if (_zpPromise) return _zpPromise;
    _zpPromise = (async function () {
        postMessage({ type: "log", level: "info", msg: "installing plotting" });
        try {
            var mp = py.pyimport("micropip");
            await mp.install(["zoomy-plotting"]);
            postMessage({ type: "log", level: "info", msg: "plotting ready" });
        } catch (e) {
            postMessage({ type: "log", level: "warn", msg: "zoomy-plotting failed: " + (e.message || e) });
        }
    })();
    return _zpPromise;
}

/* --- Parso cache on IndexedDB -------------------------------------
   Jedi's per-call cost is dominated by parso parsing every .py file
   it touches. Parso pickles ASTs to $HOME/.cache/parso/ by default; in
   Pyodide that path is ephemeral per tab. Mounting IDBFS at that path
   persists the cache across page loads so the second+ visit skips the
   ~20 s zoomy_core parse and cold-starts autocomplete in <1 s.

   First visit: cache populated during priming → syncfs(false) flushes
   to IDB. Subsequent visits: syncfs(true) populates FS from IDB →
   priming becomes a pure cache hit.

   Failure is silent and non-fatal — worst case we fall back to the
   cold-priming path. */
var _parsoCacheMounted = false;
async function mountParsoCache() {
    if (_parsoCacheMounted) return;
    postMessage({ type: "log", level: "info", msg: "updating cache" });
    try {
        var path = "/home/pyodide/.cache/parso";
        py.FS.mkdirTree(path);
        py.FS.mount(py.FS.filesystems.IDBFS, {}, path);
        await new Promise(function (resolve, reject) {
            py.FS.syncfs(true, function (err) { err ? reject(err) : resolve(); });
        });
        _parsoCacheMounted = true;
        postMessage({ type: "log", level: "info", msg: "cache ready" });
    } catch (e) {
        postMessage({ type: "log", level: "warn", msg: "cache unavailable (in-memory): " + (e.message || e) });
    }
}
async function persistParsoCache() {
    if (!_parsoCacheMounted) return;
    try {
        await new Promise(function (resolve, reject) {
            py.FS.syncfs(false, function (err) { err ? reject(err) : resolve(); });
        });
    } catch (e) {
        postMessage({ type: "log", level: "warn", msg: "cache update failed: " + (e.message || e) });
    }
}

/* The named-results shelf (/tmp/zoomy_results) is backed by IDBFS so a
   locally-saved run survives a page reload (true cross-session access).
   Mount once, syncfs(true) to load persisted stores, syncfs(false) after
   each write. Failure is silent — worst case the shelf is in-memory only
   (works within the session). */
var _resultsShelfMounted = false;
async function mountResultsShelf() {
    if (_resultsShelfMounted) return;
    try {
        var path = "/tmp/zoomy_results";
        py.FS.mkdirTree(path);
        py.FS.mount(py.FS.filesystems.IDBFS, {}, path);
        await new Promise(function (resolve, reject) {
            py.FS.syncfs(true, function (err) { err ? reject(err) : resolve(); });
        });
        _resultsShelfMounted = true;
    } catch (e) {
        postMessage({ type: "log", level: "warn", msg: "results shelf in-memory only: " + (e.message || e) });
    }
}
async function persistResultsShelf() {
    if (!_resultsShelfMounted) return;
    try {
        await new Promise(function (resolve, reject) {
            py.FS.syncfs(false, function (err) { err ? reject(err) : resolve(); });
        });
    } catch (e) {
        postMessage({ type: "log", level: "warn", msg: "results shelf persist failed: " + (e.message || e) });
    }
}

var _jediPromise = null;
function installJedi() {
    if (_jediPromise) return _jediPromise;
    _jediPromise = (async function () {
        postMessage({ type: "log", level: "info", msg: "installing autocomplete" });
        try {
            var mp = py.pyimport("micropip");
            await mp.install(["jedi"]);
        } catch (e) {
            postMessage({ type: "log", level: "warn", msg: "jedi failed: " + (e.message || e) });
            return;
        }
        /* Mount the persistent parso cache BEFORE priming so jedi's
           first parse of any zoomy_core module finds its pickled AST
           (on 2nd+ visits) instead of re-parsing from source. */
        await mountParsoCache();
        /* Prime jedi's parser cache with a representative zoomy_core
           completion. On a cold-cache run this takes 15-25 s (walking
           + parsing every transitive module). On a warm-cache run
           (IDBFS populated from a prior visit) it's <1 s. Either way,
           subsequent user completions are ~50 ms. */
        try {
            await installExec();   // pulls engine.py in so complete_code is available
            var priming = [
                "from zoomy_core.model.models import SME",
                "model = SME(level=0)",
                "model.",
            ].join("\n");
            /* 3rd line, column after "model." */
            var res = py.globals.get("complete_code")(priming, 3, 6);
            if (res && res.destroy) res.destroy();
        } catch (e) {
            /* Priming failure is non-fatal — autocomplete still works,
               just with a slow first call. Log it so we notice in CI. */
            postMessage({ type: "log", level: "warn", msg: "jedi prime failed: " + (e.message || e) });
        }
        /* Persist whatever parso just parsed so the NEXT visit starts warm. */
        await persistParsoCache();
        postMessage({ type: "log", level: "info", msg: "autocomplete ready" });
    })();
    return _jediPromise;
}

/* Regex sniffing of user code to decide which background / optional
   installs must be awaited before run_code proceeds. Each regex maps
   to a promise-guarded installer; callers hook onto the in-flight
   promise so a snippet that arrives mid-install just waits its turn. */
var _MPL_RE    = /\b(import\s+matplotlib|from\s+matplotlib|matplotlib\.)/;
var _PLOTLY_RE = /\b(import\s+plotly|from\s+plotly)/;
var _MESHIO_RE = /\b(import\s+meshio|from\s+meshio)/;
/* zoomy-plotting is used via engine.open_hdf5 — every solver-template
   snippet ends with `open_hdf5(path)`, which lazy-imports zp inside
   Python. Run_code must block on the zp install if the snippet needs it. */
var _ZP_RE     = /\b(open_hdf5|open_result|open_results|zoomy_plotting)\b/;

async function ensureVizDeps(code) {
    var needs = [];
    if (_ZP_RE.test(code))     needs.push(installZoomyPlotting());
    if (_MPL_RE.test(code))    needs.push(installMatplotlib());
    if (_PLOTLY_RE.test(code)) needs.push(installPlotly());
    if (_MESHIO_RE.test(code)) needs.push(installMeshio());
    if (needs.length) await Promise.all(needs);
}

var paramCache = {};

onmessage = async function (e) {
    var msg = e.data;
    /* Only log user-visible commands (run_code, describe_model); cache hits
       and param extraction are invisible plumbing. */
    if (msg.cmd === "run_code" || msg.cmd === "describe_model") {
        postMessage({ type: "log", level: "info", msg: msg.cmd + " (id=" + msg.id + ")" });
    }
    try {
        if (msg.cmd === "set_interrupt_buffer") {
            /* Wire it in now if Pyodide is already up; otherwise stash it
               for initPyodide to install as soon as the runtime is ready. */
            if (py) {
                try {
                    py.setInterruptBuffer(new Uint8Array(msg.buffer));
                    postMessage({ type: "log", level: "info", msg: "Cooperative interrupt enabled (SAB)" });
                } catch (e) {
                    postMessage({ type: "log", level: "warn", msg: "setInterruptBuffer failed: " + (e.message || e) });
                }
            } else {
                _pendingInterruptBuffer = msg.buffer;
            }
            return;

        } else if (msg.cmd === "init") {
            await initPyodide();
            postMessage({ type: "ready", id: msg.id });

        } else if (msg.cmd === "extract_params") {
            var cacheKey = msg.class_path + "|" + JSON.stringify(msg.init || {});
            if (paramCache[cacheKey]) {
                postMessage({ type: "log", level: "info", msg: "Param cache hit for " + msg.class_path.split(".").pop() });
                postMessage({ type: "result", id: msg.id, data: paramCache[cacheKey] });
                return;
            }
            var wt0 = performance.now();
            await installParam();
            var wt1 = performance.now();
            var result = py.globals.get("extract_param_schema")(msg.class_path, py.toPy(msg.init || {}));
            var wt2 = performance.now();
            paramCache[cacheKey] = result;
            postMessage({ type: "log", level: "info", msg: "Worker timing: installParam=" + (wt1-wt0).toFixed(0) + "ms, extract=" + (wt2-wt1).toFixed(0) + "ms" });
            postMessage({ type: "result", id: msg.id, data: result });

        } else if (msg.cmd === "preload_params") {
            await installParam();
            for (var i = 0; i < msg.cards.length; i++) {
                var c = msg.cards[i];
                var key = c.class_path + "|" + JSON.stringify(c.init || {});
                if (!paramCache[key]) {
                    try {
                        paramCache[key] = py.globals.get("extract_param_schema")(c.class_path, py.toPy(c.init || {}));
                    } catch (err) {}
                }
            }
            postMessage({ type: "log", level: "info", msg: "Pre-extracted params for " + msg.cards.length + " models" });

        } else if (msg.cmd === "run_code") {
            await installExec();
            /* A viz card that references open_result(...) needs the
               persisted results shelf mounted before it runs. */
            if (_ZP_RE.test(msg.code)) await mountResultsShelf();
            await ensureVizDeps(msg.code);
            var result = py.globals.get("process_code")(msg.code);
            postMessage({ type: "result", id: msg.id, data: result });

        } else if (msg.cmd === "complete_code") {
            /* Autocomplete via jedi. First call micropip-installs jedi
               (~2 MB; 3-5 s on a warm Pyodide); subsequent calls are
               cache hits and resolve in 30-100 ms. */
            await installExec();
            await installJedi();
            var completions = py.globals.get("complete_code")(msg.code, msg.row, msg.col);
            /* Pyodide proxies Python dicts as PyProxy objects; convert
               to a plain JS value before posting. */
            var converted = completions.toJs ? completions.toJs({ dict_converter: Object.fromEntries }) : completions;
            if (completions.destroy) completions.destroy();
            postMessage({ type: "result", id: msg.id, data: converted });

        } else if (msg.cmd === "open_hdf5") {
            /* Point the store at an HDF5 file already on Pyodide's VFS
               (written by the solver template) or at one we just wrote
               via write_hdf5_bytes. engine.open_hdf5 lazy-imports
               zoomy_plotting, so the install must finish first. */
            await installExec();
            await installZoomyPlotting();
            py.globals.get("open_hdf5")(msg.path);
            postMessage({ type: "result", id: msg.id, data: "ok" });

        } else if (msg.cmd === "write_hdf5_bytes") {
            /* Stream an HDF5 binary (e.g. downloaded from the server's
               /jobs/{id}/results/hdf5 endpoint) into Pyodide's VFS, then
               hand the path to engine.open_hdf5 (which requires zp). */
            await installExec();
            await installZoomyPlotting();
            var dir = msg.path.replace(/\/[^\/]*$/, "");
            if (dir) py.FS.mkdirTree(dir);
            py.FS.writeFile(msg.path, new Uint8Array(msg.bytes));
            py.globals.get("open_hdf5")(msg.path);
            postMessage({ type: "result", id: msg.id, data: "ok" });

        } else if (msg.cmd === "write_result_bytes") {
            /* Stage an HDF5 store into the local results shelf
               (/tmp/zoomy_results/<slug>.h5) WITHOUT opening it as the
               active store — engine.result_path computes the slugged path
               (single source of truth shared with open_result). Writing
               needs no zp; only reading (open_result) does. */
            await installExec();
            await mountResultsShelf();
            var rpath = py.globals.get("result_path")(msg.name);
            var rdir = rpath.replace(/\/[^\/]*$/, "");
            if (rdir) py.FS.mkdirTree(rdir);
            py.FS.writeFile(rpath, new Uint8Array(msg.bytes));
            await persistResultsShelf();
            postMessage({ type: "result", id: msg.id, data: rpath });

        } else if (msg.cmd === "save_result_local") {
            /* Copy the current run's open store into the local results
               shelf under msg.name (local "Save result as…"). */
            await installExec();
            await mountResultsShelf();
            var slug = py.globals.get("save_result_local")(msg.name);
            await persistResultsShelf();
            postMessage({ type: "result", id: msg.id, data: slug });

        } else if (msg.cmd === "list_results_local") {
            /* Names present in the local (VFS) results shelf. */
            await installExec();
            await mountResultsShelf();
            var names = py.globals.get("list_results")();
            var arr = names.toJs ? names.toJs() : names;
            if (names.destroy) names.destroy();
            postMessage({ type: "result", id: msg.id, data: arr });

        } else if (msg.cmd === "read_store_bytes") {
            /* Raw HDF5 bytes of the current run's open store — the source for
               routing the post-processing chain to a backend after a local
               (Pyodide) run. engine.store_source_path() returns the VFS path
               the solver template wrote (simulation.h5); we read it back. */
            await installExec();
            var srcPath = py.globals.get("store_source_path")();
            if (!srcPath) throw new Error("no open store to post-process (run a simulation first)");
            var storeBytes = py.FS.readFile(srcPath);   // Uint8Array
            postMessage({ type: "result", id: msg.id, data: storeBytes }, [storeBytes.buffer]);

        } else if (msg.cmd === "write_user_mesh") {
            /* Materialise a user-uploaded gmsh .msh into the VFS at
               `msg.path`. The card's snippet then calls
               `meshio.read(msg.path)`; ensure meshio is installed first. */
            await installMeshio();
            var mdir = msg.path.replace(/\/[^\/]*$/, "");
            if (mdir) py.FS.mkdirTree(mdir);
            py.FS.writeFile(msg.path, new Uint8Array(msg.bytes));
            postMessage({ type: "result", id: msg.id, data: "ok" });

        } else if (msg.cmd === "describe_model") {
            postMessage({ type: "log", level: "info", msg: "describe_model for " + msg.class_path.split(".").pop() + " (may take 30-60s)" });
            await installExec();
            /* Use runPythonAsync so the event loop can breathe */
            var descCode = [
                "import sys as _sys",
                "def _describe_model(class_path, init_kwargs):",
                "    _sys.stdout.write('describe: importing ' + class_path + '\\n')",
                "    mod_path, cls_name = class_path.rsplit('.', 1)",
                "    mod = __import__(mod_path, fromlist=[cls_name])",
                "    cls = getattr(mod, cls_name)",
                "    try:",
                "        _sys.stdout.write('describe: instantiating ' + cls_name + '...\\n')",
                "        m = cls(**init_kwargs) if init_kwargs else cls()",
                "        _sys.stdout.write('describe: calling describe()\\n')",
                "        if hasattr(m, 'describe'):",
                "            return str(m.describe())",
                "        return cls.__doc__ or cls.__name__",
                "    except Exception as e:",
                "        import traceback; return traceback.format_exc()",
            ].join("\n");
            await py.runPythonAsync(descCode);
            /* (calling Python — silent) */
            var desc = await py.runPythonAsync(
                "_describe_model('" + msg.class_path + "', " + JSON.stringify(msg.init || {}) + ")"
            );
            postMessage({ type: "log", level: "info", msg: "describe_model done (" + (desc ? desc.length : 0) + " chars)" });
            postMessage({ type: "result", id: msg.id, data: desc });
        }
    } catch (err) {
        postMessage({ type: "error", id: msg.id, error: err.message || String(err) });
    }
};

/* Start loading everything immediately when worker is created */
(async function () {
    await initPyodide();
    await installParam();

    /* Pre-extract params for all cards with a class — cold imports happen here, not on gear click */
    try {
        /* Load cards from the authored registry (single default.json/tab). */
        var allCards = [];
        var dirs = ["cards/models/default.json", "cards/solvers/default.json"];
        for (var di = 0; di < dirs.length; di++) {
            try {
                var arr = await fetch(dirs[di]).then(function (r) { return r.ok ? r.json() : []; });
                allCards = allCards.concat(arr);
            } catch (e) {}
        }
        var count = 0;
        for (var j = 0; j < allCards.length; j++) {
            var c = allCards[j];
            if (c["class"]) {
                var key = c["class"] + "|" + JSON.stringify(c.init || {});
                if (!paramCache[key]) {
                    try {
                        paramCache[key] = py.globals.get("extract_param_schema")(c["class"], py.toPy(c.init || {}));
                        count++;
                    } catch (e) {}
                }
            }
        }
    } catch (e) {}

    await installExec();
    postMessage({ type: "log", level: "info", msg: "runtime ready" });
    postMessage({ type: "fully_ready" });

    /* --- Install tiers ---
     *   1. CORE (boot-blocking, above): zoomy-core + param + h5py +
     *      engine.py + display hook. h5py stays here because
     *      zoomy_core.mesh.base_mesh caches _HAVE_H5PY at module
     *      import time; the boot pre-extract loop imports zoomy_core,
     *      so h5py must already be present when that happens.
     *
     *   2. BACKGROUND (eager but non-blocking, here):
     *        jedi                — autocomplete; first click is
     *                              usually a gear or editor; kick off
     *                              FIRST so it has the most wall-
     *                              clock time to finish before use.
     *        zoomy-plotting      — needed at end of every Pyodide
     *                              solver run (open_hdf5) and for
     *                              every viz refresh. ensureVizDeps
     *                              blocks run_code on this if the
     *                              snippet touches open_hdf5 / zp.
     *        matplotlib          — needed only by mpl viz cards; last
     *                              in the priority line but still
     *                              pre-warmed so the first viz
     *                              refresh is instant.
     *
     *   3. OPTIONAL (fully lazy, in run_code): plotly. Many snippets
     *      never touch it, so we don't spend the install budget until
     *      a snippet actually imports it.
     *
     * All four installers are promise-guarded, so concurrent callers
     * attach to the in-flight install instead of starting a duplicate. */
    /* Fire tier-2 installs, then post background_ready once they all
       resolve so the main thread can hide the "Installing…" toast.
       Individual "… ready" log lines keep showing up in the debug
       pane as each install completes. */
    Promise.all([
        installJedi(),              // highest-priority tier 2
        installZoomyPlotting(),
        installMatplotlib(),
    ]).then(function () {
        postMessage({ type: "log", level: "info", msg: "all ready" });
        postMessage({ type: "background_ready" });
    });
})();
