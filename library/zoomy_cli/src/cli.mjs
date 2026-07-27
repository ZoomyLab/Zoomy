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
        // TEMPORARY foam≡OpenFOAM alias — remove once the zoomy_openfoam container
        // is rebuilt to report OpenFOAM. An OpenFOAM-tagged request binds to a
        // still-foam-reporting adapter (and vice versa).
        return this.http.get(tag)
            || (tag === "OpenFOAM" && this.http.get("foam"))
            || (tag === "foam" && this.http.get("OpenFOAM"))
            || null;
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
     * visualizations): the single authored registry plus, when a session
     * is provided, per-session runtime user-authored cards.
     *
     * Merge order (later wins on id collision):
     *   1. cards/<dir>/default.json                            (authored)
     *   2. cards/sessions/<session>/<dir>/user/<id>/card.json  (runtime)
     *
     * There is exactly ONE authored tier: the generated.json / user.json
     * merge tiers were removed (catalog is now hand-curated into
     * default.json). The per-session user-card tier stays — users still
     * create their own cards at runtime.
     *
     * Cards without an `id` are preserved in place (no dedup), which
     * keeps existing test fixtures working.
     *
     * @param {string} dir
     * @param {object} [opts]
     * @param {string} [opts.session]  Active session id; when absent,
     *    only the authored default.json is loaded.
     */
    async listCards(dir, opts) {
        opts = opts || {};
        const buckets = [];
        // Tier 1: the effective catalog = shipped default.json overlaid with
        // the user's catalog overlay {removed, added} (see readCatalogOverlay).
        // With no overlay this returns the shipped default.json unchanged.
        const effective = await this.effectiveCatalog(dir);
        if (Array.isArray(effective) && effective.length) buckets.push(effective);
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

    // ------------------------------------------------------------------
    // Catalog overlay — the single card-curation mechanism.
    //
    // The card list a user sees is the shipped cards/<dir>/default.json
    // OVERLAID with a per-dir overlay {removed:[ids], added:[cards]} kept
    // in the writable storage (IndexedDB in the browser, disk on Node):
    //
    //   effective = shipped.filter(not in removed).concat(added)
    //
    // Removing a shipped card records its id in `removed`; removing a
    // catalog-added card drops it from `added`; "restore defaults" clears
    // the overlay. Save/Load project export/adopt the effective catalog as
    // cards/<dir>/default.json. This is distinct from the per-session
    // runtime user cards (listCards tier 2) — those are session scratch,
    // never part of the shipped catalog.
    // ------------------------------------------------------------------

    _catalogOverlayPath(dir) {
        return "catalog/" + dir + ".json";
    }

    /** Read the {removed, added} overlay for a dir (empty when none). */
    async readCatalogOverlay(dir) {
        const ov = await this.storage.tryReadJson(this._catalogOverlayPath(dir));
        return {
            removed: (ov && Array.isArray(ov.removed)) ? ov.removed.slice() : [],
            added:   (ov && Array.isArray(ov.added))   ? ov.added.slice()   : [],
        };
    }

    /** Persist the {removed, added} overlay for a dir. */
    async writeCatalogOverlay(dir, overlay) {
        const clean = {
            removed: (overlay && Array.isArray(overlay.removed)) ? overlay.removed : [],
            added:   (overlay && Array.isArray(overlay.added))   ? overlay.added   : [],
        };
        await this.storage.writeJson(this._catalogOverlayPath(dir), clean);
        return clean;
    }

    /** Drop the overlay for a dir entirely ("restore defaults"). */
    async clearCatalogOverlay(dir) {
        try {
            if (this.storage.deletePath) {
                await this.storage.deletePath(this._catalogOverlayPath(dir));
                return;
            }
        } catch (e) { /* fall through to empty write */ }
        await this.writeCatalogOverlay(dir, { removed: [], added: [] });
    }

    /**
     * The effective catalog for a dir: shipped default.json minus the
     * overlay's `removed` ids, plus the overlay's `added` cards (added
     * cards whose id already survives in the kept set are skipped so a
     * stale add can't shadow a shipped card). Order = shipped order first,
     * then added.
     */
    async effectiveCatalog(dir) {
        const shipped = await this.storage.tryReadJson("cards/" + dir + "/default.json");
        const base = Array.isArray(shipped) ? shipped : [];
        const ov = await this.readCatalogOverlay(dir);
        const removed = new Set(ov.removed);
        const kept = base.filter((c) => !(c && c.id && removed.has(c.id)));
        const keptIds = new Set(kept.map((c) => c && c.id).filter(Boolean));
        const added = ov.added.filter((c) => c && c.id && !keptIds.has(c.id));
        return kept.concat(added);
    }

    /**
     * Adopt an externally-supplied catalog array (e.g. a project zip's
     * cards/<dir>/default.json) as this dir's catalog, by rewriting the
     * overlay so effectiveCatalog(dir) reproduces it against the *current*
     * shipped default.json:
     *   removed = shipped ids absent from the incoming catalog
     *   added   = incoming cards whose id is not a shipped id
     * Returns the overlay written.
     */
    async adoptCatalog(dir, catalog) {
        const incoming = Array.isArray(catalog) ? catalog : [];
        const shipped = await this.storage.tryReadJson("cards/" + dir + "/default.json");
        const base = Array.isArray(shipped) ? shipped : [];
        const shippedIds = new Set(base.map((c) => c && c.id).filter(Boolean));
        const incomingIds = new Set(incoming.map((c) => c && c.id).filter(Boolean));
        const removed = [...shippedIds].filter((id) => !incomingIds.has(id));
        const added = incoming.filter((c) => c && c.id && !shippedIds.has(c.id));
        return await this.writeCatalogOverlay(dir, { removed, added });
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
        // engine.process_code returns json.dumps({status, output, store_meta, ...});
        // parse it so callers get an OBJECT (res.output / res.status / res.store_meta),
        // not a JSON string. (Callers that already tolerate a string still work.)
        const r = await this.pyodide.runCode(code);
        if (typeof r === "string") { try { return JSON.parse(r); } catch { return r; } }
        return r;
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

    /** Run a coupling on a foam backend: POST the participants to /couple, where
     *  build_coupled_bundle expands the participant OF-cases + the shared
     *  precice-config and launches both (they share the coupling folder = the
     *  exchange-directory). Returns the child job ids. */
    async submitCoupling(options) {
        options = options || {};
        const a = this.httpFor(options.tag || "OpenFOAM");
        if (!a || !a.isConnected()) {
            throw new Error("submitCoupling: backend '" + (options.tag || "OpenFOAM") + "' not connected");
        }
        return await a.runCoupling({
            coupling_id: options.coupling_id,
            scheme: options.scheme || "parallel-explicit",
            participants: options.participants || [],
            onStatus: options.onStatus,
        });
    }

    /**
     * Route the enabled post-processing chain to a connected `postprocess`
     * backend (the chain's steps — lift3d / to_vtk — live in zoomy_prepost,
     * which the browser worker does NOT ship, so they can only execute on a
     * backend). Uploads the just-finished run's store + steps, waits, and
     * returns the transformed artifacts; the caller stages them into the VFS.
     *
     * @param {object} options
     * @param {string} [options.tag]        Postprocess backend tag (default "postprocess").
     * @param {Uint8Array|ArrayBuffer} options.storeBytes   The run's simulation.h5.
     * @param {string[]} options.steps      Enabled steps.
     * @param {number}  [options.nz]        Vertical layers for lift3d.
     * @param {string}  [options.modelPy]   Model cell (needed by lift3d).
     * @param {function}[options.onStatus]  cb(statusJson) per poll.
     * @returns {Promise<{job_id, artifacts:[{name,bytes}]}>}
     */
    async runPostprocChain(options) {
        options = options || {};
        const tag = options.tag || "postprocess";
        const a = this.httpFor(tag);
        if (!a || !a.isConnected()) {
            throw new Error("runPostprocChain: backend '" + tag + "' not connected");
        }
        const steps = options.steps || [];
        if (!steps.length) return { job_id: null, artifacts: [] };
        return await a.runPostprocChain({
            storeBytes: options.storeBytes,
            steps,
            nz: options.nz || 10,
            modelPy: options.modelPy || null,
            onStatus: options.onStatus,
        });
    }

    /** Read the CURRENT local (Pyodide) run's open store as raw HDF5 bytes —
     *  the source for routing the chain after a local run. */
    async readStoreBytes() {
        return await this.pyodide.readStoreBytes();
    }

    // ------------------------------------------------------------------
    // Named result stores — the RESULTS SHELF.
    //
    // A run's HDF5 store can be SAVED UNDER A NAME and any visualization
    // can OPEN other results by name, across sessions and runs:
    //   - remote runs   -> the connected backend's server-side registry
    //                      (POST/GET/DELETE /api/v1/results);
    //   - local runs     -> the Pyodide VFS shelf /tmp/zoomy_results/*.h5.
    // ------------------------------------------------------------------

    /** Save a completed remote job's HDF5 into the backend `tag`'s shelf. */
    async saveResult(tag, jobId, name) {
        const a = this.httpFor(tag);
        if (!a || !a.isConnected()) throw new Error("saveResult: backend '" + tag + "' not connected");
        return await a.saveResult(jobId, name);
    }

    /** List named results from the backend `tag`'s shelf. */
    async listResults(tag) {
        const a = this.httpFor(tag);
        if (!a || !a.isConnected()) throw new Error("listResults: backend '" + tag + "' not connected");
        return await a.listResults(tag);
    }

    /** Fetch a named result's HDF5 bytes (ArrayBuffer) from backend `tag`. */
    async fetchResult(tag, name) {
        const a = this.httpFor(tag);
        if (!a || !a.isConnected()) throw new Error("fetchResult: backend '" + tag + "' not connected");
        return await a.fetchResult(name);
    }

    /** Delete a named result from backend `tag`'s shelf. */
    async deleteResult(tag, name) {
        const a = this.httpFor(tag);
        if (!a || !a.isConnected()) throw new Error("deleteResult: backend '" + tag + "' not connected");
        return await a.deleteResult(name);
    }

    /**
     * Make a named result available to viz cards: fetch its bytes from
     * backend `tag` and stage them into the local Pyodide results shelf
     * (/tmp/zoomy_results/<name>.h5) so `open_result(name)` works. When
     * `bytes` is passed it is staged directly (used for local runs — no
     * server round-trip).
     */
    async stageResult(name, options) {
        options = options || {};
        let bytes = options.bytes || null;
        if (!bytes) {
            if (!options.tag) throw new Error("stageResult: need bytes or a backend tag");
            const buf = await this.fetchResult(options.tag, name);
            bytes = new Uint8Array(buf);
        }
        await this.pyodide.writeResultBytes(name, bytes);
        return { name, size: bytes.byteLength };
    }

    /** Names present in the local (Pyodide-VFS) results shelf. */
    async listResultsLocal() {
        return await this.pyodide.listResultsLocal();
    }

    /** Save the current LOCAL run's open store into the local shelf. */
    async saveResultLocal(name) {
        return await this.pyodide.saveResultLocal(name);
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
        /* Option A: one merged "Solver settings" section — the backend tag is
           just an informational "backend" entry inside the settings. */
        const settingsOut = Object.assign({}, settings);
        if (solver.tag && settingsOut.backend === undefined) settingsOut.backend = solver.tag;
        // Carry the solver card id too: several solvers can share one backend tag
        // (e.g. the coupled zoomyFoam + incompressibleVOF both run on "OpenFOAM"), so
        // the tag alone can't restore WHICH one a case selected.
        if (solver.id && settingsOut.solver_id === undefined) settingsOut.solver_id = solver.id;
        let head = "# " + (meta.title || "Zoomy case") + (meta.description ? "\n\n" + meta.description : "");
        if (meta.gui_url) head += "\n\n[← Back to the Zoomy GUI](" + meta.gui_url + ")";
        const H = (section, title) => ({ type: "markdown", meta: { role: "heading", section }, source: "## " + title });
        const cells = [
            { type: "markdown",
              meta: { role: "meta", title: meta.title || null, description: meta.description || null },
              source: head },
            H("model", "Model"),
            { type: "code",
              meta: { role: "model", class_path: model.class_path || null, init: model.init || {}, card: model.card || null },
              source: trim(model.code) },
            H("mesh", "Mesh"),
            { type: "code",
              meta: { role: "mesh", spec: mesh.spec || null },
              source: trim(mesh.code) },
            H("settings", "Solver settings"),
            { type: "code",
              meta: { role: "settings", settings: settingsOut },
              source: "settings = " + JSON.stringify(settingsOut, null, 2) },
            /* Run: the exported file is FULLY runnable — `python case.py` or
               Run-All in Jupyter solves in-process wherever zoomy_core exists
               (JupyterLite, container JupyterLab, local env). */
            H("run", "Run"),
            { type: "code", meta: { role: "run" },
              source: (spec.run && spec.run.code) || this._runCode() },
        ];
        /* Visualization is ALWAYS attached: the selected viz card's code
           (with the notebook prelude) when provided, else the generated
           default plot — case AND figure reproduce. The post-processing
           chain lives INSIDE the viz code (prepended by gatherCaseSpec via
           chainCode — survives jupytext); spec.postproc rides along as a
           zoomy metadata hint so parseCase can round-trip the enabled
           steps and the GUI strip can restore from an imported case. */
        // Post-processing: its OWN section (the zoomy_prepost chain) BEFORE the
        // visualization, so the exported .py/.ipynb reproduces the pipeline that
        // ran and the viz reads the post-processed result. spec.postproc rides on
        // the cell meta so parseCase round-trips the enabled steps + Nz.
        if (Array.isArray(spec.postproc) && spec.postproc.length) {
            cells.push(H("postproc", "Post-processing"));
            cells.push({ type: "code", meta: { role: "postproc", steps: spec.postproc, nz: spec.postproc_nz || 10 },
                         source: this.chainCode(spec.postproc, spec.postproc_nz) });
        }
        cells.push(H("visualization", "Visualization"));
        cells.push({ type: "code", meta: { role: "visualization" },
                     source: trim(viz.code) || this._vizCode() });
        return cells;
    }

    /** Notebook prelude for viz-card code: provides the globals the GUI
     *  worker injects (store/time_step/field_name/display), reading the
     *  simulation.h5 the Run section produced. */
    vizPrelude() {
        return [
            "import matplotlib",
            "import matplotlib.pyplot as plt",
            "import zoomy_plotting as zp",
            "",
            "store = zp.read_hdf5(\"simulation.h5\")",
            "time_step = store.n_snapshots - 1        # last snapshot",
            "field_name = None                        # None -> first field",
            "try:",
            "    display                              # provided by Jupyter",
            "except NameError:",
            "    display = lambda *a: None            # plain-script fallback",
            "",
        ].join("\n");
    }

    /** Post-processing chain code (zoomy_prepost.steps) for the enabled
     *  steps — a subset of ["lift3d", "to_h5"]. The GUI prepends this to
     *  the visualization code (gatherCaseSpec) so exports/notebooks
     *  reproduce the pipeline; the chain runs BEFORE the plot. Blocks are
     *  emitted in EXECUTION order regardless of input order: to_h5 first
     *  (it produces the simulation.h5 that lift3d and the viz read).
     *  Every block is guarded so the case still runs where a step doesn't
     *  apply (no VTK output, zoomy_prepost not installed). */
    chainCode(steps, nz) {
        steps = steps || [];
        nz = nz || 10;
        const blocks = [];
        if (steps.indexOf("to_h5") !== -1) {
            blocks.push([
                "# post-processing: VTK -> H5 (zoomy_prepost.vtk_to_hdf5)",
                "try:",
                "    import os, glob",
                "    pvds = sorted(glob.glob(\"*.pvd\"))    # (no VTK present -> skip)",
                "    if pvds and not os.path.exists(\"simulation.h5\"):",
                "        from zoomy_prepost import vtk_to_hdf5",
                "        vtk_to_hdf5(pvds[0], \"simulation.h5\")",
                "except ImportError:",
                "    pass  # zoomy_prepost/meshio missing -> skip",
            ].join("\n"));
        }
        if (steps.indexOf("lift3d") !== -1) {
            blocks.push([
                "# post-processing: 2D -> 3D lift (zoomy_prepost.lift3d; needs the case model)",
                "try:",
                "    from zoomy_prepost import lift3d",
                "    lift3d(\".\", \"simulation.h5\", \"simulation_3d\", Nz=" + nz + ")",
                "except ImportError:",
                "    pass  # needs zoomy_prepost (available in containers; Lite lockfile TBD)",
            ].join("\n"));
        }
        return blocks.length ? blocks.join("\n\n") + "\n" : "";
    }

    /** Generated default visualization (mirrors zoomy_prepost.case._viz_code). */
    _vizCode() {
        return this.vizPrelude() + [
            "field = next(iter(store.field.keys()))",
            "with zp.apply_style():",
            "    fig, ax = plt.subplots()",
            "    zp.MatplotlibPlotter(store).plot(ax, time_step=time_step, field=field,",
            "                                     **({} if store.dim == 1 else {\"cmap\": \"viridis\", \"colorbar\": True}))",
            "    if store.times is not None and len(store.times):",
            "        ax.set_title(f\"{field} — t = {float(store.times[time_step]):.3f}\")",
            "fig.savefig(\"simulation.png\", dpi=150, bbox_inches=\"tight\")",
            "display(fig)",
            "print(\"figure -> simulation.png\")",
        ].join("\n");
    }

    /** Generated in-process runner (mirrors zoomy_prepost.case._run_code). */
    _runCode() {
        return [
            "# Runs IN-PROCESS with the numpy solver (no server needed) — works in",
            "# JupyterLite, a backend container's JupyterLab, any env with zoomy_core,",
            "# or standalone as `python run.py` in a materialized case folder (the",
            "# zoomy-server generic runner executes exactly this file).",
            "# For other backends, submit this file via the Zoomy GUI / zoomy-server.",
            "import json",
            "import os",
            "",
            "# Headless shim: the model/mesh cards end in display(model.describe()),",
            "# a Jupyter builtin. Define a no-op so `python run.py` runs on any",
            "# backend / plain env (the GUI worker & Jupyter provide the real one).",
            "try:",
            "    display",
            "except NameError:",
            "    def display(*_a, **_k):",
            "        pass",
            "",
            "# Folder form: the notebook defines model/mesh/settings in the cells above;",
            "# a standalone run.py loads them from the sibling case-folder files.",
            "if \"model\" not in globals():",
            "    exec(open(\"model.py\").read())",
            "if \"mesh\" not in globals():",
            "    exec(open(\"mesh.py\").read())",
            "if \"settings\" not in globals():",
            "    settings = json.load(open(\"settings.json\"))",
            "",
            "from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec",
            "from zoomy_core.fvm.solver_numpy import FreeSurfaceFlowSolver",
            "import zoomy_core.fvm.timestepping as ts",
            "from zoomy_core.misc.misc import Zstruct",
            "",
            "# effective settings = general keys + this backend's branch",
            "_eff = {k: v for k, v in settings.items() if not isinstance(v, dict)}",
            "_eff.update(settings.get(\"numpy\", {}))",
            "",
            "nsm = NumericalSystemModel.from_system_model(",
            "    model,",
            "    reconstruction=ReconstructionSpec(",
            "        order=_eff.get(\"reconstruction_order\", 1),",
            "        limiter=_eff.get(\"limiter\", \"venkatakrishnan\")))",
            "solver = FreeSurfaceFlowSolver(",
            "    time_end=_eff.get(\"time_end\", 0.1),",
            "    compute_dt=ts.adaptive(CFL=_eff.get(\"cfl\", 0.45)),",
            "    settings=Zstruct(output=Zstruct(",
            "        directory=os.getcwd(), filename=\"simulation\",",
            "        snapshots=_eff.get(\"output_snapshots\", 10), clean_directory=False)))",
            "",
            "mesh.write_to_hdf5(\"simulation.h5\")   # mesh first; the solver appends /fields",
            "Q, Qaux = solver.solve(mesh, nsm, write_output=True)",
            "print(\"done -> simulation.h5\")",
        ].join("\n");
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
            /* "solver settings" must match before bare "solver" (legacy). */
            const m = mdSource.match(/^\s*#{1,3}\s*(model|mesh|solver\s+settings|settings|solver|visuali[sz]ation|numerics|run)\b/im);
            if (!m) return null;
            const s = m[1].toLowerCase().replace(/\s+/g, " ");
            if (s.startsWith("visuali")) return "visualization";
            if (s === "solver settings") return "settings";
            return s;
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
            spec.model = { code: sources.model, class_path: h.class_path, init: h.init || {}, card: h.card || null };
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
        /* backend tag (option A): settings.backend; legacy fallback = a bare
           "## Solver" section / solver_tag line / solver metadata. */
        let tag = spec.settings && spec.settings.backend;
        if (!tag && (sources.solver !== undefined || (hints.solver && hints.solver.tag))) {
            tag = hints.solver && hints.solver.tag;
            if (!tag) {
                const m = (sources.solver || "").match(/solver_tag\s*=\s*["']([\w-]+)["']/);
                tag = m ? m[1] : null;
            }
        }
        if (tag) spec.solver = { tag, id: (spec.settings && spec.settings.solver_id) || (hints.solver && hints.solver.id) || null, params: (hints.solver && hints.solver.params) || {} };
        if (sources.visualization !== undefined) spec.visualization = { code: sources.visualization };
        /* post-processing chain hint (round-trips the GUI strip toggles) */
        const vh = hints.visualization;
        if (vh && Array.isArray(vh.postproc) && vh.postproc.length) spec.postproc = vh.postproc;
        if (sources.numerics !== undefined) spec.numerics = { code: sources.numerics };
        if (sources.run !== undefined) spec.run = { code: sources.run };   // re-export fidelity
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
