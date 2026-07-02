/**
 * ZoomyCLI — the single façade the GUI talks to.
 *
 * The CLI owns:
 *  - a Storage (fs-backed on Node, fetch-backed in the browser) for card
 *    manifests, snippet files, and static assets.
 *  - one PyodideAdapter (always present in the browser — Pyodide is the
 *    "local Python" runtime) and zero-or-more HttpAdapters (remote
 *    simulation backends).
 *
 * Design points called out in the plan:
 *  - The server is a job queue; runCode / extractParams / describeModel
 *    go through Pyodide regardless of which HTTP adapter is connected.
 *  - submitCase picks the HTTP adapter when one is connected for the
 *    requested tag; falls back to synthesising a Pyodide run locally.
 *  - Every method returns a Promise so call-sites look the same whether
 *    the adapter is sync or async.
 */

import { NotSupportedError } from "./adapters/pyodide_adapter.mjs";

export class ZoomyCLI {
    /**
     * @param {object} options
     * @param {object} options.storage           Storage instance.
     * @param {object} options.pyodide           PyodideAdapter (required).
     * @param {Map<string, object>} [options.httpAdapters]
     *          Optional map of tag -> HttpAdapter. Callers can also
     *          register/unregister via connectHttp / disconnectHttp.
     */
    constructor(options) {
        options = options || {};
        if (!options.pyodide) throw new Error("ZoomyCLI: options.pyodide is required");
        if (!options.storage) throw new Error("ZoomyCLI: options.storage is required");
        this.storage = options.storage;
        /* Mutable — app.js swaps this to a per-session PyodideAdapter
           when a session becomes active. At construction it points at
           the boot-time adapter; the first session to run code
           "claims" that adapter (no cold-boot tax for the first
           session); later sessions spawn their own. */
        this.pyodide = options.pyodide;
        this.http = options.httpAdapters instanceof Map
            ? options.httpAdapters
            : new Map();
    }

    // ------------------------------------------------------------------
    // Adapter registry — lets callers register HTTP backends at runtime.
    // ------------------------------------------------------------------

    registerHttp(adapter) {
        if (!adapter || !adapter.tag) throw new Error("HttpAdapter without a tag");
        this.http.set(adapter.tag, adapter);
        this._emitConnectionsChange();
        return adapter;
    }

    unregisterHttp(tag) {
        const a = this.http.get(tag);
        if (a) { try { a.disconnect(); } catch (e) {} this.http.delete(tag); }
        this._emitConnectionsChange();
    }

    httpFor(tag) {
        if (!tag) return null;
        return this.http.get(tag) || null;
    }

    isHttpConnected(tag) {
        const a = this.httpFor(tag);
        return !!(a && a.isConnected && a.isConnected());
    }

    /**
     * List every tag we can execute against right now: `numpy` for the
     * always-available Pyodide runtime, plus one entry per connected
     * HttpAdapter. Returns pretty labels (tag + source) so the navbar
     * can display them verbatim.
     */
    availableTags() {
        const out = ["numpy (pyodide)"];
        for (const [tag, a] of this.http) {
            if (a.isConnected()) out.push(tag);
        }
        return out;
    }

    /**
     * Unified "is this tag executable?" test. `numpy` (Pyodide) is
     * always connected; other tags must have a live HttpAdapter.
     */
    isTagConnected(tag) {
        if (tag === "numpy") return true;
        return this.isHttpConnected(tag);
    }

    getUrlForTag(tag) {
        const a = this.httpFor(tag);
        return a ? a.url : null;
    }

    /**
     * Register a listener that fires whenever the HTTP adapter set
     * changes (register, unregister, or a heartbeat marks an adapter
     * as lost). Returns an unsubscribe function.
     */
    onConnectionsChange(listener) {
        if (!this._connectionListeners) this._connectionListeners = new Set();
        this._connectionListeners.add(listener);
        return () => this._connectionListeners.delete(listener);
    }

    _emitConnectionsChange() {
        if (!this._connectionListeners) return;
        for (const fn of this._connectionListeners) {
            try { fn(); } catch (e) {}
        }
    }

    /**
     * High-level backend discovery: probe the default localhost URL
     * and, on success, register an HttpAdapter for it. Used by the
     * GUI's "Discover" button.
     */
    async discover(defaultUrl) {
        try { return await this.connect(defaultUrl || "http://localhost:8080"); }
        catch (e) { return null; }
    }

    /**
     * Probe a URL's /health. On 200+ok, register an HttpAdapter for it
     * and start a heartbeat that unregisters the adapter if /health
     * stops responding. Returns the adapter on success, null on
     * failure. The HttpAdapter class is resolved on first call via
     * dynamic import to avoid a hard dep cycle.
     */
    async connect(url) {
        if (!this._HttpAdapter) {
            const mod = await import("./adapters/http_adapter.mjs");
            this._HttpAdapter = mod.HttpAdapter;
        }
        const adapter = new this._HttpAdapter({ url });
        try {
            await adapter.connect();
        } catch (e) {
            return null;
        }
        this.registerHttp(adapter);
        this._startHeartbeat(adapter);
        return adapter;
    }

    disconnect(tag) {
        const a = this.httpFor(tag);
        if (!a) return;
        this._stopHeartbeat(tag);
        try { a.disconnect(); } catch (e) {}
        this.http.delete(tag);
        this._emitConnectionsChange();
    }

    _startHeartbeat(adapter) {
        if (!this._heartbeats) this._heartbeats = new Map();
        const tag = adapter.tag;
        if (this._heartbeats.has(tag)) clearInterval(this._heartbeats.get(tag));
        const handle = setInterval(async () => {
            try {
                const h = await adapter.health();
                if (h.status !== "ok") throw new Error("bad status");
            } catch (e) {
                this._stopHeartbeat(tag);
                this.http.delete(tag);
                this._emitConnectionsChange();
            }
        }, 5000);
        this._heartbeats.set(tag, handle);
    }

    _stopHeartbeat(tag) {
        if (!this._heartbeats) return;
        const h = this._heartbeats.get(tag);
        if (h) { clearInterval(h); this._heartbeats.delete(tag); }
    }

    // ------------------------------------------------------------------
    // Card / tab / snippet loading.
    // ------------------------------------------------------------------

    async listTabs() {
        return await this.storage.readJson("cards/tabs.json");
    }

    /**
     * Load the cards for a given type dir (models / meshes / solvers /
     * visualizations), merging legacy tier files and, when a session is
     * provided, per-session user-authored cards.
     *
     * Merge order (later wins on id collision):
     *   1. cards/<dir>/default.json    (read-only, shipped)
     *   2. cards/<dir>/generated.json  (read-only, shipped)
     *   3. cards/<dir>/user.json       (legacy — kept for back-compat)
     *   4. cards/sessions/<session>/<dir>/user/<id>/card.json  (new)
     *
     * Cards without an `id` are preserved in place (no dedup), which
     * keeps existing test fixtures working.
     *
     * @param {string} dir
     * @param {object} [opts]
     * @param {string} [opts.session]  Active session id; when absent,
     *    only the legacy tiers are loaded (pre-session behaviour).
     */
    async listCards(dir, opts) {
        opts = opts || {};
        const sources = ["default.json", "generated.json", "user.json"];
        const buckets = [];
        for (const src of sources) {
            const list = await this.storage.tryReadJson("cards/" + dir + "/" + src);
            if (Array.isArray(list)) buckets.push(list);
        }
        if (opts.session) {
            const { listUserCards } = await import("./user_cards.mjs");
            const userCards = await listUserCards(this.storage, opts.session, dir);
            if (userCards.length) buckets.push(userCards);
        }
        // Insertion-order merge with last-wins id dedup.
        const idxById = new Map();
        const result = [];
        for (const bucket of buckets) {
            for (const card of bucket) {
                if (!card) continue;
                if (card.id && idxById.has(card.id)) {
                    result[idxById.get(card.id)] = card;
                } else {
                    if (card.id) idxById.set(card.id, result.length);
                    result.push(card);
                }
            }
        }
        return result;
    }

    async fetchSnippet(path) {
        return await this.storage.tryReadText(path);
    }

    /**
     * Registry: prefer any connected HTTP backend's /api/v1/registry,
     * else return null. Callers can merge with static listings.
     */
    async listRegistry(tag) {
        if (tag) {
            const a = this.httpFor(tag);
            if (a && a.isConnected()) return await a.listRegistry();
            return null;
        }
        // No specific tag: return the first registry from any connected
        // adapter, or null.
        for (const [, a] of this.http) {
            if (a.isConnected()) {
                try { return await a.listRegistry(); } catch (e) {}
            }
        }
        return null;
    }

    // ------------------------------------------------------------------
    // Interactive Python primitives — always through Pyodide. If a
    // future backend gains the surface we can delegate, but for now
    // this avoids duplicating execution on the server.
    // ------------------------------------------------------------------

    async runCode(code) {
        return await this.pyodide.runCode(code);
    }

    async extractParams(classPath, init) {
        return await this.pyodide.extractParams(classPath, init);
    }

    async describeModel(classPath, init) {
        return await this.pyodide.describeModel(classPath, init);
    }

    async openHdf5(path) {
        return await this.pyodide.openHdf5(path);
    }

    async writeHdf5Bytes(path, bytes) {
        return await this.pyodide.writeHdf5Bytes(path, bytes);
    }

    async preloadParams(cards) {
        return await this.pyodide.preloadParams(cards);
    }

    /**
     * Autocomplete via jedi (Pyodide). Forwards the full source buffer
     * + cursor to the worker; returns jedi's proposed completions.
     * (row 1-indexed, col 0-indexed — matches jedi's API.)
     */
    async complete(code, row, col) {
        return await this.pyodide.complete(code, row, col);
    }

    // ------------------------------------------------------------------
    // Simulation submission — HTTP first, Pyodide as local fallback.
    // ------------------------------------------------------------------

    /**
     * @param {object} options
     * @param {string} [options.tag]   Backend tag; falls back to Pyodide
     *                                 (local) when absent or not connected.
     * @param {object} [options.case]  Server-style case payload (used with
     *                                 HTTP adapter).
     * @param {string} [options.code]  Local Python code (used with Pyodide).
     * @param {function} [options.onStatus]   Cb(statusJson) for remote jobs.
     * @param {AbortSignal} [options.signal]  Aborts both modes.
     *
     * Resolves to { mode, result } where result is a runCode result for
     * local mode or { job_id, hdf5 } for remote mode (after download).
     */
    async submitCase(options) {
        options = options || {};
        const http = options.tag ? this.httpFor(options.tag) : null;

        if (http && http.isConnected() && (options.casePy || options.case)) {
            // Prefer the composed canonical case .py (POST /cases); fall back to
            // a raw case object (legacy /jobs) if only that was given.
            const payload = options.casePy
                ? { case_py: options.casePy,
                    mesh_b64: options.meshB64 || null,
                    mesh_name: options.meshName || null }
                : options.case;
            const res = await http.submitCase(payload, {
                onStatus: options.onStatus,
                signal: options.signal,
            });
            // Pipe the HDF5 into Pyodide's VFS so viz cards can read it.
            if (res.hdf5) {
                const path = "/tmp/zoomy_sim/" + res.job_id + ".h5";
                await this.pyodide.writeHdf5Bytes(path, new Uint8Array(res.hdf5));
            }
            return { mode: "http", result: res };
        }

        // Local: caller must pass `code`. We don't synthesise case->code
        // here because the card-assembly logic lives in app.js; keeping
        // the CLI transport-agnostic means it doesn't import that.
        if (!options.code) {
            throw new Error("submitCase: local mode requires options.code");
        }
        const result = await this.pyodide.runCode(options.code);
        return { mode: "pyodide", result };
    }

    // ----- case interchange (the real work; the GUI is a thin frontend) ------
    //
    // A case is a single jupytext "percent" .py where each section is a cell
    // tagged `# %% zoomy={...}` — runnable, a notebook (.py<->.ipynb), and
    // losslessly mapped to/from the GUI cards. Mirrors zoomy_prepost.case so the
    // server (to_folder) and the browser agree on one format. The MODEL cell
    // carries the fully-specified model INCLUDING its IC + BC (else the PDE is
    // not well-posed and cannot run).

    /** Ordered case cells from resolved card data (shared by .py and .ipynb).
     *  v2: markdown HEADINGS (## Model, ## Mesh, ...) are the primary
     *  structure — they survive jupytext and hand editing; the zoomy metadata
     *  stays as a machine hint. ## Visualization is emitted only when the
     *  spec carries viz code (optional, keeps the figure reproducible). */
    _caseCells(spec) {
        spec = spec || {};
        const meta = spec.meta || {}, model = spec.model || {}, mesh = spec.mesh || {};
        const settings = spec.settings || {}, solver = spec.solver || {};
        const viz = spec.visualization || {};
        const trim = (s) => String(s || "").replace(/\s+$/, "");
        let head = "# " + (meta.title || "Zoomy case") + (meta.description ? "\n\n" + meta.description : "");
        if (meta.gui_url) head += "\n\n[← Back to the Zoomy GUI](" + meta.gui_url + ")";
        const H = (section, title) => ({ type: "markdown", meta: { role: "heading", section }, source: "## " + title });
        const cells = [
            { type: "markdown",
              meta: { role: "meta", title: meta.title || null, description: meta.description || null },
              source: head },
            H("model", "Model"),
            { type: "code",
              meta: { role: "model", class_path: model.class_path || null, init: model.init || {} },
              source: trim(model.code) },
            H("mesh", "Mesh"),
            { type: "code",
              meta: { role: "mesh", spec: mesh.spec || null },
              source: trim(mesh.code) },
            H("settings", "Settings"),
            { type: "code",
              meta: { role: "settings", settings },
              source: "settings = " + JSON.stringify(settings, null, 2) },
            H("solver", "Solver"),
            { type: "code",
              meta: { role: "solver", tag: solver.tag || "numpy", params: solver.params || {} },
              source: "solver_tag = " + JSON.stringify(solver.tag || "numpy") },
        ];
        if (viz.code) {
            cells.push(H("visualization", "Visualization"));
            cells.push({ type: "code", meta: { role: "visualization" }, source: trim(viz.code) });
        }
        return cells;
    }

    /** Resolved card data -> canonical case .py (jupytext percent + zoomy meta). */
    composeCase(spec) {
        return this._caseCells(spec).map((c) => {
            const marker = "# %%" + (c.type === "markdown" ? " [markdown]" : "") +
                           " zoomy=" + JSON.stringify(c.meta);
            const src = c.type === "markdown"
                ? c.source.split("\n").map((l) => "# " + l).join("\n")
                : c.source;
            return marker + "\n" + src + "\n";
        }).join("\n");
    }

    /** spec -> a downloadable artifact string; fmt "py" (default) or "ipynb". */
    exportCase(spec, fmt) {
        if (fmt === "ipynb") {
            const cells = this._caseCells(spec).map((c) => {
                const lines = c.source.split("\n");
                const source = lines.map((l, i) => (i < lines.length - 1 ? l + "\n" : l));
                const cell = { cell_type: c.type, metadata: { zoomy: c.meta }, source };
                if (c.type === "code") { cell.outputs = []; cell.execution_count = null; }
                return cell;
            });
            return JSON.stringify({
                cells,
                metadata: { kernelspec: { name: "python3", display_name: "Python 3" },
                            language_info: { name: "python" } },
                nbformat: 4, nbformat_minor: 5,
            }, null, 1);
        }
        return this.composeCase(spec);
    }

    /** Canonical case .py -> spec (importing a case back to cards).
     *  v2, HEADINGS FIRST: markdown cells whose heading opens a known section
     *  (## Model / Mesh / Settings / Solver / Visualization) claim the code
     *  cells that follow — robust to hand editing; zoomy metadata is the
     *  fallback/hint (class_path, init, settings, tag). */
    parseCase(pyText) {
        // split the percent .py into cells
        const cells = [];
        let cur = null;
        for (const line of String(pyText || "").split("\n")) {
            const m = line.match(/^# %%( \[markdown\])?(?: zoomy=(.*))?\s*$/);
            if (m) {
                if (cur) cells.push(cur);
                let meta = {};
                try { meta = m[2] ? JSON.parse(m[2]) : {}; } catch (e) { meta = {}; }
                cur = { markdown: !!m[1], meta, source: [] };
            } else if (cur) {
                cur.source.push(line);
            }
        }
        if (cur) cells.push(cur);

        const sectionOf = (mdSource) => {
            const m = mdSource.match(/^\s*#{1,3}\s*(model|mesh|settings|solver|visuali[sz]ation|numerics)\b/im);
            if (!m) return null;
            const s = m[1].toLowerCase();
            return s.startsWith("visuali") ? "visualization" : s;
        };

        const sources = {}, hints = {}, spec = {};
        let current = null;
        for (const c of cells) {
            const src = c.source.join("\n").replace(/^\n+|\n+$/g, "");
            if (c.markdown) {
                const md = src.replace(/^# ?/gm, "");   // strip the comment prefix
                const sec = sectionOf(md);
                if (sec) { current = sec; continue; }
                if (c.meta.role === "meta") {
                    spec.meta = { title: c.meta.title, description: c.meta.description };
                } else if (!spec.meta && current === null) {
                    const first = (md.trim().split("\n")[0] || "");
                    if (first.startsWith("#")) spec.meta = { title: first.replace(/^#+\s*/, "") };
                }
                continue;
            }
            const sec = current || ({ model: "model", mesh: "mesh", settings: "settings",
                                      solver: "solver", visualization: "visualization",
                                      numerics: "numerics" })[c.meta.role];
            if (!sec) continue;
            sources[sec] = sources[sec] ? sources[sec] + "\n\n" + src : src;
            if (c.meta && Object.keys(c.meta).length) {
                hints[sec] = Object.assign(hints[sec] || {}, c.meta);
            }
        }

        if (sources.model !== undefined) {
            const h = hints.model || {};
            spec.model = { code: sources.model, class_path: h.class_path, init: h.init || {} };
        }
        if (sources.mesh !== undefined) {
            const h = hints.mesh || {};
            spec.mesh = { code: sources.mesh, spec: h.spec };
        }
        if (sources.settings !== undefined || (hints.settings && hints.settings.settings)) {
            let s = hints.settings && hints.settings.settings;
            if (!s) {
                const m = (sources.settings || "").match(/settings\s*=\s*(\{[\s\S]*\})/);
                if (m) { try { s = JSON.parse(m[1]); } catch (e) { s = null; } }
            }
            spec.settings = s || {};
        }
        if (sources.solver !== undefined || (hints.solver && hints.solver.tag)) {
            let tag = hints.solver && hints.solver.tag;
            if (!tag) {
                const m = (sources.solver || "").match(/solver_tag\s*=\s*["']([\w-]+)["']/);
                tag = m ? m[1] : "numpy";
            }
            spec.solver = { tag, params: (hints.solver && hints.solver.params) || {} };
        }
        if (sources.visualization !== undefined) spec.visualization = { code: sources.visualization };
        if (sources.numerics !== undefined) spec.numerics = { code: sources.numerics };
        return spec;
    }

    // ----- open a case in Jupyter (Lite in-browser, or a backend's JupyterLab) --

    /**
     * Open the composed case as a notebook in Jupyter. Routing mirrors Run:
     *  - a backend with a reachable JupyterLab (container `jupyter` mode on
     *    :8888) -> stage the notebook via the Jupyter REST contents API and
     *    open that lab (the backend's REAL kernel: zoomy_jax, firedrake, ...);
     *  - otherwise -> stage into the deployed JupyterLite's IndexedDB store
     *    and open Lite (in-browser pyodide kernel; no server at all).
     *
     * @param {object} spec       Resolved card spec (same as composeCase).
     * @param {object} [options]
     * @param {string} [options.jupyterUrl] Backend JupyterLab base (e.g.
     *          "http://localhost:8888"); tried first when given.
     * @param {string} [options.liteUrl]    JupyterLite deployment base
     *          (default: "../jupyter-lite/_output/" next to the GUI).
     * @param {string} [options.filename]   Notebook name (default zoomy_case.ipynb).
     * @returns {Promise<{mode: "server"|"lite", url: string}>}
     */
    async openInJupyter(spec, options) {
        options = options || {};
        const filename = options.filename || "zoomy_case.ipynb";
        const nb = JSON.parse(this.exportCase(spec, "ipynb"));
        /* Stage the FULL project alongside the notebook — the case folder the
           server would materialize, plus a README that links back to the GUI —
           so the Jupyter file browser shows the whole case, not a lone .ipynb. */
        const files = this._caseProjectFiles(spec);

        if (options.jupyterUrl) {
            try {
                const url = await this._stageInServer(nb, files, options.jupyterUrl, filename);
                window.open(url, "_blank");
                return { mode: "server", url };
            } catch (e) {
                /* fall through to Lite — backend jupyter not reachable */
            }
        }
        /* JupyterLite's only kernel is `python` (Pyodide); any other kernelspec
           name makes Lab prompt instead of auto-attaching. */
        nb.metadata = nb.metadata || {};
        nb.metadata.kernelspec = { name: "python", display_name: "Python (Pyodide)", language: "python" };
        const liteBase = new URL(options.liteUrl || "../jupyter-lite/_output/", window.location.href).href;
        await this._stageInLite(nb, files, liteBase, filename);
        const url = liteBase + "lab/index.html?path=" + encodeURIComponent(filename);
        window.open(url, "_blank");
        return { mode: "lite", url };
    }

    /** The case-folder view of the spec: the same files the server materializes
     *  (model.py / mesh.py / settings.json) + a README linking back to the GUI. */
    _caseProjectFiles(spec) {
        const cells = this._caseCells(spec);
        const by = {};
        for (const c of cells) by[c.meta.role] = c;
        const guiUrl = (typeof window !== "undefined" && window.location) ? window.location.href.split("#")[0] : "";
        const title = (spec.meta && spec.meta.title) || "Zoomy case";
        const files = {};
        if (by.model && by.model.source) files["model.py"] = by.model.source + "\n";
        if (by.mesh && by.mesh.source) files["mesh.py"] = by.mesh.source + "\n";
        files["settings.json"] = JSON.stringify((spec.settings || {}), null, 2) + "\n";
        files["README.md"] =
            "# " + title + "\n\n" +
            "Composed by the Zoomy GUI — the case folder next to `zoomy_case.ipynb`:\n" +
            "`model.py` (model incl. IC + BC), `mesh.py`, `settings.json`.\n\n" +
            (guiUrl ? "[← Back to the Zoomy GUI](" + guiUrl + ")\n" : "");
        return files;
    }

    /** PUT the notebook + project files via the Jupyter Server REST contents
     *  API. Needs the server to allow the GUI origin (zoomy-jupyter sets CORS
     *  + token-less localhost by default). Returns the lab URL for the file. */
    async _stageInServer(nb, files, jupyterUrl, filename) {
        const base = jupyterUrl.replace(/\/+$/, "");
        const put = async (path, body) => {
            const resp = await fetch(base + "/api/contents/" + encodeURIComponent(path), {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            if (!resp.ok) throw new Error("contents PUT " + path + " failed: HTTP " + resp.status);
        };
        await put(filename, { type: "notebook", format: "json", content: nb });
        for (const [name, text] of Object.entries(files || {})) {
            await put(name, { type: "file", format: "text", content: text });
        }
        return base + "/lab/tree/" + encodeURIComponent(filename);
    }

    /** Candidate IndexedDB names for the Lite contents store. Ground truth
     *  (probed against a real Lite 0.8 boot): the drive uses
     *  `contentsStorageName` VERBATIM when set (our build pins
     *  "zoomy-jupyterlite"), else `JupyterLite Storage - ${resolved absolute
     *  PATHNAME of the deployment, with trailing slash}` (never the raw "./",
     *  never the full origin URL). Existing JupyterLite DBs are included too. */
    async _liteDbNames(liteBase) {
        const names = new Set();
        try {
            const cfg = await (await fetch(liteBase + "jupyter-lite.json")).json();
            const jcd = (cfg && cfg["jupyter-config-data"]) || {};
            if (jcd.contentsStorageName) {
                names.add(jcd.contentsStorageName);
                names.add("JupyterLite Storage - " + jcd.contentsStorageName);
            }
        } catch (e) { /* config fetch is best-effort */ }
        names.add("JupyterLite Storage - " + new URL("./", liteBase).pathname);
        try {
            const dbs = (await indexedDB.databases()) || [];
            for (const d of dbs) {
                if (d.name && (d.name.startsWith("JupyterLite Storage") || d.name === "zoomy-jupyterlite")) names.add(d.name);
            }
        } catch (e) { /* indexedDB.databases() unsupported -> candidates only */ }
        return [...names];
    }

    /** Write the notebook + project files into JupyterLite's IndexedDB
     *  contents store (localforage layout: store "files", key = path, value =
     *  a full Jupyter contents model). Static build contents merge with these
     *  at listing time, so the file browser shows both. Same-origin only. */
    async _stageInLite(nb, files, liteBase, filename) {
        const now = new Date().toISOString();
        const models = [];
        models.push([filename, {
            name: filename, path: filename, type: "notebook",
            format: "json", mimetype: "application/json",
            content: nb, size: JSON.stringify(nb).length,
            created: now, last_modified: now, writable: true,
        }]);
        for (const [name, text] of Object.entries(files || {})) {
            models.push([name, {
                name, path: name, type: "file",
                format: "text",
                mimetype: name.endsWith(".json") ? "application/json" : "text/plain",
                content: text, size: text.length,
                created: now, last_modified: now, writable: true,
            }]);
        }
        const put = (dbName) => new Promise((resolve, reject) => {
            const open = indexedDB.open(dbName);
            open.onupgradeneeded = () => {
                /* fresh DB (Lite not booted yet): create the localforage-style
                   "files" store; Lite upgrade-adds its other stores on boot. */
                const db = open.result;
                if (!db.objectStoreNames.contains("files")) db.createObjectStore("files");
            };
            open.onerror = () => reject(open.error);
            open.onsuccess = () => {
                const db = open.result;
                if (!db.objectStoreNames.contains("files")) { db.close(); resolve(false); return; }
                const tx = db.transaction("files", "readwrite");
                for (const [key, model] of models) tx.objectStore("files").put(model, key);
                tx.oncomplete = () => { db.close(); resolve(true); };
                tx.onerror = () => { db.close(); reject(tx.error); };
            };
        });
        const names = await this._liteDbNames(liteBase);
        const results = await Promise.allSettled(names.map(put));
        if (!results.some((r) => r.status === "fulfilled" && r.value)) {
            throw new Error("could not stage the case in any JupyterLite store");
        }
    }

    /** Read a staged case back from JupyterLite's store (the shared browser
     *  filesystem): returns the parsed spec from zoomy_case.ipynb, or null.
     *  This is the GUI's "import back what I edited in Jupyter" hook. */
    async readCaseFromLite(options) {
        options = options || {};
        const filename = options.filename || "zoomy_case.ipynb";
        const liteBase = new URL(options.liteUrl || "../jupyter-lite/_output/", window.location.href).href;
        const names = await this._liteDbNames(liteBase);
        for (const dbName of names) {
            const model = await new Promise((resolve) => {
                const open = indexedDB.open(dbName);
                open.onupgradeneeded = () => { /* don't create stores on read */ };
                open.onerror = () => resolve(null);
                open.onsuccess = () => {
                    const db = open.result;
                    if (!db.objectStoreNames.contains("files")) { db.close(); resolve(null); return; }
                    const req = db.transaction("files", "readonly").objectStore("files").get(filename);
                    req.onsuccess = () => { db.close(); resolve(req.result || null); };
                    req.onerror = () => { db.close(); resolve(null); };
                };
            });
            if (model && model.content) {
                const py = this._notebookToPercentPy(model.content);
                return this.parseCase(py);
            }
        }
        return null;
    }

    /** Minimal .ipynb -> percent-py (cells with zoomy metadata) so parseCase
     *  can consume a notebook read back from a Jupyter store. */
    _notebookToPercentPy(nb) {
        const parts = [];
        for (const cell of (nb.cells || [])) {
            const z = cell.metadata && cell.metadata.zoomy;
            if (!z) continue;
            const src = Array.isArray(cell.source) ? cell.source.join("") : String(cell.source || "");
            const marker = "# %%" + (cell.cell_type === "markdown" ? " [markdown]" : "") + " zoomy=" + JSON.stringify(z);
            parts.push(marker + "\n" + src + "\n");
        }
        return parts.join("\n");
    }

    /** Cancel a remote job (tag) or interrupt Pyodide. */
    async cancel(options) {
        options = options || {};
        if (options.tag && options.jobId) {
            const a = this.httpFor(options.tag);
            if (a) return await a.cancelJob(options.jobId);
        }
        // Default = interrupt the local Pyodide.
        return this.pyodide.interrupt();
    }

    /** Static helper so callers can test for the error type. */
    static NotSupportedError = NotSupportedError;
}
