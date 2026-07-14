/**
 * HttpAdapter — wraps zoomy_server's /api/v1 routes for browser use.
 *
 * The server's API is a job queue (POST /jobs + poll + HDF5 download),
 * so this adapter's primary job is `submitCase` which bundles that
 * sequence. Interactive primitives (runCode, extractParams,
 * describeModel) are NOT supported over HTTP by design — the CLI
 * transparently falls back to Pyodide when those are requested.
 *
 * Symmetry with PyodideAdapter is maintained in method name only:
 * methods that the server can't provide throw NotSupportedError so the
 * CLI's dispatcher can catch and re-route without polymorphism hacks.
 */

import { NotSupportedError } from "./pyodide_adapter.mjs";

const DEFAULT_POLL_INTERVAL_MS = 2000;

export class HttpAdapter {
    /**
     * @param {object} options
     * @param {string}  options.url       Base URL (e.g. http://localhost:8080).
     * @param {string}  [options.tag]     Label for the adapter (defaults to the
     *                                    server's /health.tag at connect time).
     * @param {number}  [options.pollMs]  Job poll interval (ms).
     */
    constructor(options) {
        options = options || {};
        if (!options.url) throw new Error("HttpAdapter: options.url is required");
        this.url = options.url.replace(/\/$/, "");
        this.tag = options.tag || "http";
        this.pollMs = options.pollMs || DEFAULT_POLL_INTERVAL_MS;
        this._pollTimers = new Map();
        this._connected = false;
        this.backends = [];
    }

    async _fetchJson(path, init) {
        const r = await fetch(this.url + path, init);
        if (!r.ok) throw new Error("HTTP " + r.status + " at " + path);
        return await r.json();
    }

    async connect() {
        const data = await this._fetchJson("/api/v1/health");
        if (data.status !== "ok") throw new Error("unhealthy: " + JSON.stringify(data));
        if (data.tag) this.tag = data.tag;
        this.backends = data.backends || [];
        this._connected = true;
        return { tag: this.tag, backends: this.backends };
    }

    disconnect() {
        for (const [, h] of this._pollTimers) clearInterval(h);
        this._pollTimers.clear();
        this._connected = false;
    }

    isConnected() { return this._connected; }

    async health() {
        return await this._fetchJson("/api/v1/health");
    }

    async listRegistry() {
        return await this._fetchJson("/api/v1/registry");
    }

    /**
     * Submit a case to the server and poll to completion, downloading
     * the HDF5 output when it arrives.
     *
     * @param {object} caseData   Case definition or { case_dir } payload.
     * @param {object} [options]
     * @param {function} [options.onStatus]   cb(statusJson) per poll.
     * @param {AbortSignal} [options.signal]  Aborting rejects the promise
     *                                        AND cancels the remote job.
     * @returns {Promise<{job_id: string, hdf5: ArrayBuffer}>}
     */
    async submitCase(caseData, options) {
        options = options || {};
        // A composed case (has case_py) is uploaded to /cases, where the server
        // materializes the case folder; a legacy raw case object goes to /jobs.
        const endpoint = caseData && caseData.case_py ? "/api/v1/cases" : "/api/v1/jobs";
        const resp = await fetch(this.url + endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(caseData),
        });
        if (!resp.ok) throw new Error("submit failed: HTTP " + resp.status);
        const body = await resp.json();
        const jobId = body.job_id;

        const status = await this._pollUntilTerminal(jobId, options);
        if (status.status === "failed") {
            throw new Error(status.error || "job failed");
        }
        if (status.status === "cancelled") {
            return { job_id: jobId, status: "cancelled", hdf5: null };
        }

        // Pull the binary HDF5 output.
        const r = await fetch(this.url + "/api/v1/jobs/" + jobId + "/results/hdf5");
        if (!r.ok) throw new Error("HDF5 download failed: HTTP " + r.status);
        const hdf5 = await r.arrayBuffer();
        return { job_id: jobId, status: "complete", hdf5 };
    }

    async _pollUntilTerminal(jobId, options) {
        options = options || {};
        const onStatus = options.onStatus || function () {};
        const signal = options.signal || null;
        return await new Promise((resolve, reject) => {
            const tick = async () => {
                if (signal && signal.aborted) {
                    clearInterval(handle);
                    this._pollTimers.delete(jobId);
                    try { await this.cancelJob(jobId); } catch (e) {}
                    reject(new Error("aborted"));
                    return;
                }
                try {
                    const status = await this._fetchJson("/api/v1/jobs/" + jobId);
                    try { onStatus(status); } catch (e) {}
                    const terminal = status.status === "complete" ||
                                     status.status === "failed" ||
                                     status.status === "cancelled";
                    if (terminal) {
                        clearInterval(handle);
                        this._pollTimers.delete(jobId);
                        resolve(status);
                    }
                } catch (e) {
                    // Transient errors: keep polling unless signal fired.
                }
            };
            const handle = setInterval(tick, this.pollMs);
            this._pollTimers.set(jobId, handle);
            tick();
        });
    }

    // ------------------------------------------------------------------
    // Named result stores — the RESULTS SHELF (server-side registry).
    // A completed job's HDF5 can be saved under a name and reopened by
    // name from any later session/run. See zoomy_server/results.py.
    // ------------------------------------------------------------------

    /** Save a completed job's HDF5 into the shelf. Returns {name,size,created}. */
    async saveResult(jobId, name) {
        const r = await fetch(this.url + "/api/v1/results", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ job_id: jobId, name }),
        });
        if (!r.ok) throw new Error("saveResult failed: HTTP " + r.status);
        return await r.json();
    }

    /** List named results: [{name, size, created}, ...] (newest first). */
    async listResults() {
        return await this._fetchJson("/api/v1/results");
    }

    /** Fetch a named result's HDF5 bytes as an ArrayBuffer. */
    async fetchResult(name) {
        const r = await fetch(this.url + "/api/v1/results/" +
                              encodeURIComponent(name) + "/hdf5");
        if (!r.ok) throw new Error("fetchResult failed: HTTP " + r.status);
        return await r.arrayBuffer();
    }

    /** Delete a named result. Returns {status} or {status:"not_found"}. */
    async deleteResult(name) {
        const r = await fetch(this.url + "/api/v1/results/" +
                              encodeURIComponent(name), { method: "DELETE" });
        if (!r.ok && r.status !== 404) throw new Error("deleteResult failed: HTTP " + r.status);
        if (r.status === 404) return { status: "not_found" };
        return await r.json();
    }

    async cancelJob(jobId) {
        const h = this._pollTimers.get(jobId);
        if (h) { clearInterval(h); this._pollTimers.delete(jobId); }
        const r = await fetch(this.url + "/api/v1/jobs/" + jobId, { method: "DELETE" });
        if (!r.ok && r.status !== 404) throw new Error("cancel failed: HTTP " + r.status);
        if (r.status === 404) return { status: "not_found" };
        return await r.json();
    }

    // ------------------------------------------------------------------
    // Not supported over HTTP (interactive primitives; run via Pyodide).
    // ------------------------------------------------------------------
    runCode()         { throw new NotSupportedError("runCode"); }
    extractParams()   { throw new NotSupportedError("extractParams"); }
    describeModel()   { throw new NotSupportedError("describeModel"); }
    openHdf5()        { throw new NotSupportedError("openHdf5"); }
    writeHdf5Bytes()  { throw new NotSupportedError("writeHdf5Bytes"); }
    complete()        { throw new NotSupportedError("complete"); }
}
