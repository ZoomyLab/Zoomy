/* Loads the vendored Zoomy GUI "brain" (the zoomy_cli ESM) at runtime and builds
 * a single ZoomyCLI wired to the vendored Pyodide worker + card catalog under
 * `gui/`. The dynamic import is hidden from the bundler (via `new Function`) so
 * esbuild/webpack leaves the served ESM — and its relative `./src/*.mjs` imports
 * and the worker's relative asset fetches — intact.
 *
 * Reusing zoomy_cli gives the real GUI everything for free: card catalog,
 * param extraction, run/describe/complete, case compose/parse/export, remote
 * backends by URL, and the results shelf. The Theia side only renders. */

export interface DisplayCell { mime: string; content: string; }

// The active per-run collector for streamed display() output. runCode routes
// each display cell to whoever is currently running.
let displaySink: ((cell: DisplayCell) => void) | undefined;
export function setDisplaySink(fn: ((cell: DisplayCell) => void) | undefined): void { displaySink = fn; }

let logSink: ((level: string, msg: string) => void) | undefined;
export function setLogSink(fn: ((level: string, msg: string) => void) | undefined): void { logSink = fn; }

let cliPromise: Promise<any> | undefined;

function loadScript(src: string): Promise<void> {
    return new Promise((res, rej) => {
        const s = document.createElement('script'); s.src = src;
        s.onload = () => res(); s.onerror = () => rej(new Error('load ' + src));
        document.head.appendChild(s);
    });
}

/** The single shared ZoomyCLI. First call boots the vendored brain + Pyodide worker. */
export function getZoomyCli(): Promise<any> {
    if (!cliPromise) {
        cliPromise = (async () => {
            const base = new URL('gui/', document.baseURI).href;
            // core.js publishes window.ZoomyCore (CardState/Project/…) for later
            // state/session work; harmless to load now.
            try { await loadScript(base + 'core.js'); } catch (e) { /* non-fatal for card render/run */ }
            // Hidden dynamic import: the bundler must NOT try to resolve this.
            const dynImport = new Function('u', 'return import(u)') as (u: string) => Promise<any>;
            const mod = await dynImport(base + 'zoomy_cli/browser.mjs');
            const { ZoomyCLI, PyodideAdapter, FetchStorage, IdbStorage } = mod;
            const pyodide = new PyodideAdapter({
                workerUrl: base + 'pyodide-worker.js',
                onLog: (level: string, msg: string) => logSink && logSink(level, msg),
                onDisplay: (cell: DisplayCell) => displaySink && displaySink(cell),
            });
            let overlay: any = null;
            try { overlay = new IdbStorage(); } catch (e) { /* private mode: writes error later */ }
            const storage = new FetchStorage({ baseUrl: base, overlay });
            return new ZoomyCLI({ storage, pyodide });
        })();
    }
    return cliPromise;
}
