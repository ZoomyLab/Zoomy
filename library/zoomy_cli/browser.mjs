/**
 * Browser entry for zoomy_cli.
 *
 * This file is what the GUI imports via `<script type="module">`:
 *     import { createBrowserCLI } from "./zoomy_cli/browser.mjs";
 *
 * It wires up FetchStorage + PyodideAdapter with sensible defaults and
 * returns a ready-to-use ZoomyCLI. HTTP adapters are NOT created here —
 * connect to them on demand via cli.connectHttp(url, HttpAdapter).
 */

export { ZoomyCLI } from "./src/cli.mjs";
export { PyodideAdapter, NotSupportedError } from "./src/adapters/pyodide_adapter.mjs";
export { HttpAdapter } from "./src/adapters/http_adapter.mjs";
export { FetchStorage } from "./src/storage.mjs";

import { ZoomyCLI } from "./src/cli.mjs";
import { PyodideAdapter } from "./src/adapters/pyodide_adapter.mjs";
import { FetchStorage } from "./src/storage.mjs";

/**
 * Convenience factory for the GUI. Either pass an existing Worker (the
 * GUI already creates one) or let the adapter boot its own.
 *
 * @param {object} [options]
 * @param {Worker} [options.worker]
 * @param {string} [options.workerUrl]
 * @param {SharedArrayBuffer} [options.interruptBuffer]
 * @param {function} [options.onLog]
 * @param {function} [options.onDisplay]
 * @param {function} [options.onReady]
 * @returns {ZoomyCLI}
 */
export function createBrowserCLI(options) {
    options = options || {};
    const pyodide = new PyodideAdapter({
        worker: options.worker,
        workerUrl: options.workerUrl,
        interruptBuffer: options.interruptBuffer,
        onLog: options.onLog,
        onDisplay: options.onDisplay,
        onReady: options.onReady,
    });
    const storage = new FetchStorage();
    return new ZoomyCLI({ storage, pyodide });
}
