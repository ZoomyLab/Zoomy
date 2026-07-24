/**
 * PyodideAdapter — wraps a Web Worker running pyodide-worker.js.
 *
 * The adapter is the single integration point between the CLI and the
 * worker's message protocol. Public methods mirror the worker's `cmd`
 * surface (runCode, extractParams, describeModel, openHdf5, write-
 * Hdf5Bytes) so the CLI's own method list is a direct pass-through.
 *
 * Ownership & lifecycle:
 *  - The adapter can be constructed with an existing `Worker` (the GUI
 *    creates one at boot and passes it in) or asked to create its own
 *    via `options.workerUrl`.
 *  - Every outgoing message gets a monotonic id so concurrent calls
 *    don't collide. The listener filters by id.
 *  - `interrupt()` writes SIGINT into the shared interrupt buffer (if
 *    the host has one); otherwise falls back to `terminate()` +
 *    `createWorker()`, mirroring the stop-simulation path in app.js.
 *  - `destroy()` terminates the worker and rejects any in-flight calls.
 *
 * The adapter is intentionally quiet on the log channel — callers can
 * subscribe to `onLog(msg)` to receive the worker's `{type:"log"}`
 * messages. `onDisplay(cell)` receives `{type:"display"}` messages so
 * the GUI can route them to the right card.
 */

export class NotSupportedError extends Error {
    constructor(method) {
        super("PyodideAdapter: method '" + method + "' not implemented");
        this.name = "NotSupportedError";
    }
}

export class PyodideAdapter {
    /**
     * @param {object} options
     * @param {Worker} [options.worker]       Existing worker (preferred).
     * @param {string} [options.workerUrl]    Path to pyodide-worker.js
     *                                        (used only when no worker given).
     * @param {function} [options.onLog]      cb({level, msg}) — optional.
     * @param {function} [options.onDisplay]  cb(cell) — optional.
     * @param {function} [options.onReady]    cb() when worker posts fully_ready.
     * @param {function} [options.onBackgroundReady]
     *          cb() when all tier-2 (matplotlib / jedi / zoomy_plotting)
     *          installs have resolved and the worker is fully warm.
     * @param {SharedArrayBuffer} [options.interruptBuffer]
     *          When cross-origin isolated, this is used for cooperative cancel.
     */
    constructor(options) {
        options = options || {};
        this.tag = "pyodide";
        this.workerUrl = options.workerUrl || "pyodide-worker.js";
        this.onLog = options.onLog || function () {};
        this.onDisplay = options.onDisplay || function () {};
        this.onReady = options.onReady || function () {};
        this.onBackgroundReady = options.onBackgroundReady || function () {};
        this._interruptBuffer = options.interruptBuffer || null;
        this._interruptView = this._interruptBuffer ? new Uint8Array(this._interruptBuffer) : null;

        this._pending = new Map();
        this._msgId = 0;
        this._worker = options.worker || null;
        if (this._worker) this._wire(this._worker);
    }

    _createWorker() {
        const w = new Worker(this.workerUrl);
        this._wire(w);
        if (this._interruptBuffer) {
            if (this._interruptView) this._interruptView[0] = 0;
            w.postMessage({ cmd: "set_interrupt_buffer", buffer: this._interruptBuffer });
        }
        return w;
    }

    _wire(w) {
        w.onmessage = (e) => {
            const msg = e.data;
            if (msg.type === "fully_ready")       { this.onReady(); return; }
            if (msg.type === "background_ready")  { this.onBackgroundReady(); return; }
            if (msg.type === "log")     { this.onLog(msg); return; }
            if (msg.type === "display") { this.onDisplay(msg.cell); return; }
            const cb = this._pending.get(msg.id);
            if (!cb) return;
            this._pending.delete(msg.id);
            if (msg.type === "error") cb.reject(new Error(msg.error));
            else cb.resolve(msg.data);
        };
    }

    /** Ensure we have a worker; returns it. */
    _ensureWorker() {
        if (!this._worker) this._worker = this._createWorker();
        return this._worker;
    }

    /** Ensure worker exists AND is booted. Resolves when fully_ready. */
    async connect() {
        this._ensureWorker();
        if (this._ready) return;
        await new Promise((resolve) => {
            const prev = this.onReady;
            this.onReady = () => { prev && prev(); this._ready = true; resolve(); };
            /* If the worker is already ready (because it was handed to us
               post-boot) we'll never get another event. Use a tiny probe
               to detect this case. */
            this._postCmd({ cmd: "init" }).then(() => { this.onReady = prev; this._ready = true; resolve(); }, () => {});
        });
    }

    disconnect() {
        if (!this._worker) return;
        try { this._worker.terminate(); } catch (e) {}
        for (const [, cb] of this._pending) { try { cb.reject(new Error("disconnected")); } catch (e) {} }
        this._pending.clear();
        this._worker = null;
        this._ready = false;
    }

    /** Low-level message roundtrip — used by every public method. */
    _postCmd(msg) {
        const w = this._ensureWorker();
        const id = ++this._msgId;
        msg.id = id;
        return new Promise((resolve, reject) => {
            this._pending.set(id, { resolve, reject });
            w.postMessage(msg);
        });
    }

    // -------------------------------------------------------------------
    // Adapter surface — mirrors pyodide-worker.js commands.
    // -------------------------------------------------------------------

    async runCode(code) {
        return await this._postCmd({ cmd: "run_code", code });
    }

    async extractParams(classPath, init) {
        return await this._postCmd({ cmd: "extract_params", class_path: classPath, init: init || {} });
    }

    async describeModel(classPath, init) {
        return await this._postCmd({ cmd: "describe_model", class_path: classPath, init: init || {} });
    }

    async openHdf5(path) {
        return await this._postCmd({ cmd: "open_hdf5", path });
    }

    async writeHdf5Bytes(path, bytes) {
        return await this._postCmd({ cmd: "write_hdf5_bytes", path, bytes });
    }

    /**
     * Write HDF5 bytes into the local results shelf (Pyodide VFS
     * /tmp/zoomy_results/<name>.h5) WITHOUT installing them as the active
     * `store`. Lets a viz card `open_result(name)` a saved run without
     * clobbering the store the current run produced.
     */
    async writeResultBytes(name, bytes) {
        return await this._postCmd({ cmd: "write_result_bytes", name, bytes });
    }

    /** List the names saved in the local (Pyodide-VFS) results shelf. */
    async listResultsLocal() {
        return await this._postCmd({ cmd: "list_results_local" });
    }

    /** Raw HDF5 bytes of the current run's open store (its ``source_path``
     *  in the VFS). Used to route the post-processing chain to a backend
     *  after a local run. Returns a Uint8Array. */
    async readStoreBytes() {
        return await this._postCmd({ cmd: "read_store_bytes" });
    }

    /** Save the current run's open store into the local shelf under `name`. */
    async saveResultLocal(name) {
        return await this._postCmd({ cmd: "save_result_local", name });
    }

    /**
     * Materialise a user-uploaded mesh file (gmsh .msh bytes) into the
     * Pyodide VFS at `path` so a mesh snippet can `meshio.read(path)`.
     * Ensures meshio is installed in the worker on first call.
     */
    async writeUserMesh(path, bytes) {
        return await this._postCmd({ cmd: "write_user_mesh", path, bytes });
    }

    async preloadParams(cards) {
        return await this._postCmd({ cmd: "preload_params", cards });
    }

    /**
     * Jedi-backed autocomplete. (row is 1-indexed for jedi; col is 0-indexed.)
     * Returns { completions: [{name, type, signature, docstring, module}, ...] }.
     */
    async complete(code, row, col) {
        return await this._postCmd({ cmd: "complete_code", code, row, col });
    }

    /**
     * Cooperative or forceful cancel. Prefers SIGINT on the shared
     * buffer (keeps the worker alive, no re-boot); falls back to a
     * full terminate + re-create if no buffer is configured.
     */
    interrupt() {
        if (this._interruptView) {
            this._interruptView[0] = 2;     // SIGINT
            return { mode: "cooperative" };
        }
        if (!this._worker) return { mode: "noop" };
        try { this._worker.terminate(); } catch (e) {}
        for (const [, cb] of this._pending) { try { cb.reject(new Error("interrupted")); } catch (e) {} }
        this._pending.clear();
        this._ready = false;
        this._worker = null;
        this._ensureWorker();                // fresh worker boots lazily
        return { mode: "terminate+recreate" };
    }

    /** After a cooperative interrupt completes we clear the SIGINT flag. */
    resetInterrupt() {
        if (this._interruptView) this._interruptView[0] = 0;
    }

    // -------------------------------------------------------------------
    // HTTP-side methods that don't make sense here. Keep a consistent
    // surface with HttpAdapter by declaring them; they simply throw so
    // the CLI can decide to route these calls to the right adapter.
    // -------------------------------------------------------------------
    submitCase() { throw new NotSupportedError("submitCase"); }
    cancelJob()  { throw new NotSupportedError("cancelJob"); }
    listRegistry() { throw new NotSupportedError("listRegistry"); }
    health()       { throw new NotSupportedError("health"); }
}
