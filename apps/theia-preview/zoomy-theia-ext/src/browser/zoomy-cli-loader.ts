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

let libsPromise: Promise<void> | undefined;
/** Load KaTeX (+ auto-render) and marked so card descriptions and describe()
 *  output render markdown + math — the same CDN libs the standalone GUI uses. */
export function ensureRenderLibs(): Promise<void> {
    if (!libsPromise) {
        libsPromise = (async () => {
            const link = document.createElement('link');
            link.rel = 'stylesheet'; link.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css';
            document.head.appendChild(link);
            await loadScript('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js');
            await loadScript('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js');
            await loadScript('https://cdn.jsdelivr.net/npm/marked@12/marked.min.js');
        })().catch(() => { /* offline / blocked — fall back to plain text */ });
    }
    return libsPromise;
}

// Tiny shared event so the activity-bar view can refresh its case list when the
// config widget creates/loads/switches cases.
const caseListeners = new Set<() => void>();
export function onCasesChanged(fn: () => void): () => void { caseListeners.add(fn); return () => caseListeners.delete(fn); }
export function emitCasesChanged(): void { caseListeners.forEach(f => { try { f(); } catch { /* ignore */ } }); }

// Shared event: connected backends changed (connect / disconnect). The status
// bar and the left Zoomy view both refresh from it.
const backendListeners = new Set<() => void>();
export function onBackendsChanged(fn: () => void): () => void { backendListeners.add(fn); return () => backendListeners.delete(fn); }
export function emitBackendsChanged(): void { backendListeners.forEach(f => { try { f(); } catch { /* ignore */ } }); }

// Shared event: simulation console output. The bottom "Simulation" panel
// subscribes and streams these; `{kind:'clear'}` resets it.
export interface SimOutputEvent { kind: 'clear' | 'line'; level?: 'info' | 'stdout' | 'error' | 'ok'; text?: string; }
const simListeners = new Set<(e: SimOutputEvent) => void>();
export function onSimOutput(fn: (e: SimOutputEvent) => void): () => void { simListeners.add(fn); return () => simListeners.delete(fn); }
export function emitSimOutput(e: SimOutputEvent): void { simListeners.forEach(f => { try { f(e); } catch { /* ignore */ } }); }

let jszipPromise: Promise<void> | undefined;
/** Load JSZip from a CDN (for shipping a set of cases as a ZIP artefact by URL,
 *  like the old GUI). Exposes window.JSZip. */
export function ensureJSZip(): Promise<void> {
    if (!jszipPromise) {
        jszipPromise = loadScript('https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js')
            .catch(() => { /* offline — project-from-URL just won't work */ });
    }
    return jszipPromise;
}

let gitPromise: Promise<any> | undefined;
/** Load isomorphic-git + a browser FS (lightning-fs, IndexedDB-backed) + the web
 *  http client from a CDN. Returns {git, http, fs} for in-browser clone/commit/push.
 *  Hidden dynamic import so the bundler leaves the ESM alone. */
export function ensureGit(): Promise<any> {
    if (!gitPromise) {
        gitPromise = (async () => {
            const imp = new Function('u', 'return import(u)') as (u: string) => Promise<any>;
            const [gitMod, httpMod, fsMod] = await Promise.all([
                imp('https://esm.sh/isomorphic-git@1.27.1'),
                imp('https://esm.sh/isomorphic-git@1.27.1/http/web'),
                imp('https://esm.sh/@isomorphic-git/lightning-fs@4.6.0'),
            ]);
            const FS = fsMod.default || fsMod;
            const fs = new FS('zoomy-git');
            return { git: gitMod.default || gitMod, http: httpMod.default || httpMod, fs };
        })();
    }
    return gitPromise;
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
                // The adapter calls onLog with the whole {level,msg} message object.
                onLog: (m: any) => { logSink && logSink(m?.level || 'info', m?.msg ?? String(m)); },
                // The worker posts display cells as a JSON string (json.dumps(cell)).
                onDisplay: (cell: any) => {
                    let c = cell;
                    if (typeof cell === 'string') { try { c = JSON.parse(cell); } catch { c = { mime: 'text/plain', content: cell }; } }
                    displaySink && displaySink(c);
                },
            });
            let overlay: any = null;
            try { overlay = new IdbStorage(); } catch (e) { /* private mode: writes error later */ }
            const storage = new FetchStorage({ baseUrl: base, overlay });
            return new ZoomyCLI({ storage, pyodide });
        })();
    }
    return cliPromise;
}
