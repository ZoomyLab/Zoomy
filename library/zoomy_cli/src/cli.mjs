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
        return adapter;
    }

    unregisterHttp(tag) {
        const a = this.http.get(tag);
        if (a) { try { a.disconnect(); } catch (e) {} this.http.delete(tag); }
    }

    httpFor(tag) {
        if (!tag) return null;
        return this.http.get(tag) || null;
    }

    isHttpConnected(tag) {
        const a = this.httpFor(tag);
        return !!(a && a.isConnected && a.isConnected());
    }

    // ------------------------------------------------------------------
    // Card / tab / snippet loading.
    // ------------------------------------------------------------------

    async listTabs() {
        return await this.storage.readJson("cards/tabs.json");
    }

    /**
     * Load the cards for a given type dir (models / meshes / solvers /
     * visualizations), merging `default.json`, `generated.json`, and
     * `user.json` in that order. Missing files are silently skipped.
     */
    async listCards(dir) {
        const sources = ["default.json", "generated.json", "user.json"];
        const out = [];
        for (const src of sources) {
            const list = await this.storage.tryReadJson("cards/" + dir + "/" + src);
            if (Array.isArray(list)) out.push.apply(out, list);
        }
        return out;
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

        if (http && http.isConnected() && options.case) {
            const res = await http.submitCase(options.case, {
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

    /** Connect-to-HTTP helper: creates an adapter, probes /health,
     *  registers on success. */
    async connectHttp(url, AdapterClass) {
        const adapter = new AdapterClass({ url });
        await adapter.connect();
        return this.registerHttp(adapter);
    }

    /** Static helper so callers can test for the error type. */
    static NotSupportedError = NotSupportedError;
}
