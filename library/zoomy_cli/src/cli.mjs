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

    /** Ordered case cells from resolved card data (shared by .py and .ipynb). */
    _caseCells(spec) {
        spec = spec || {};
        const meta = spec.meta || {}, model = spec.model || {}, mesh = spec.mesh || {};
        const settings = spec.settings || {}, solver = spec.solver || {};
        const trim = (s) => String(s || "").replace(/\s+$/, "");
        return [
            { type: "markdown",
              meta: { role: "meta", title: meta.title || null, description: meta.description || null },
              source: "# " + (meta.title || "Zoomy case") + (meta.description ? "\n\n" + meta.description : "") },
            { type: "code",
              meta: { role: "model", class_path: model.class_path || null, init: model.init || {} },
              source: trim(model.code) },
            { type: "code",
              meta: { role: "mesh", spec: mesh.spec || null },
              source: trim(mesh.code) },
            { type: "code",
              meta: { role: "settings", settings },
              source: "settings = " + JSON.stringify(settings, null, 2) },
            { type: "code",
              meta: { role: "solver", tag: solver.tag || "numpy", params: solver.params || {} },
              source: "solver_tag = " + JSON.stringify(solver.tag || "numpy") },
        ];
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

    /** Canonical case .py -> spec (for importing a downloaded case back to cards). */
    parseCase(pyText) {
        const spec = {};
        const cells = [];
        let cur = null;
        for (const line of String(pyText || "").split("\n")) {
            const m = line.match(/^# %%(?: \[markdown\])? zoomy=(.*)$/);
            if (m) {
                if (cur) cells.push(cur);
                let meta = {};
                try { meta = JSON.parse(m[1]); } catch (e) { meta = {}; }
                cur = { meta, source: [] };
            } else if (cur) {
                cur.source.push(line);
            }
        }
        if (cur) cells.push(cur);
        for (const c of cells) {
            const src = c.source.join("\n").replace(/^\n+|\n+$/g, "");
            switch (c.meta.role) {
                case "model": spec.model = { code: src, class_path: c.meta.class_path, init: c.meta.init || {} }; break;
                case "mesh": spec.mesh = { code: src, spec: c.meta.spec }; break;
                case "settings": spec.settings = c.meta.settings || {}; break;
                case "solver": spec.solver = { tag: c.meta.tag || "numpy", params: c.meta.params || {} }; break;
                case "meta": spec.meta = { title: c.meta.title, description: c.meta.description }; break;
            }
        }
        return spec;
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
