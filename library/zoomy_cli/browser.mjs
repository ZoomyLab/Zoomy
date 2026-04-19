/**
 * Browser entry for zoomy_cli.
 *
 * This file is what the GUI imports via `<script type="module">`:
 *     import { createBrowserCLI } from "./zoomy_cli/browser.mjs";
 *
 * It wires up FetchStorage (+ IdbStorage overlay for user-authored
 * cards) with PyodideAdapter and returns a ready-to-use ZoomyCLI. HTTP
 * adapters are NOT created here — connect to them on demand via
 * cli.connect(url) or cli.registerHttp(adapter).
 */

export { ZoomyCLI } from "./src/cli.mjs";
export { PyodideAdapter, NotSupportedError } from "./src/adapters/pyodide_adapter.mjs";
export { HttpAdapter } from "./src/adapters/http_adapter.mjs";
export { FetchStorage } from "./src/storage.mjs";
export { IdbStorage } from "./src/adapters/idb_storage.mjs";
export * as userCards from "./src/user_cards.mjs";

import { ZoomyCLI } from "./src/cli.mjs";
import { PyodideAdapter } from "./src/adapters/pyodide_adapter.mjs";
import { FetchStorage } from "./src/storage.mjs";
import { IdbStorage } from "./src/adapters/idb_storage.mjs";

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
    // IndexedDB should be available in every target browser, but guard
    // the construction so a Safari-style "no IDB in private mode" fail
    // doesn't take down the entire GUI at boot. Writes will surface the
    // missing overlay as a clear error at use-site instead.
    let overlay = null;
    try { overlay = new IdbStorage(); }
    catch (e) { /* no-op — writes will throw a clearer error later */ }
    const storage = new FetchStorage({ overlay });
    return new ZoomyCLI({ storage, pyodide });
}
