import React from '@theia/core/shared/react';
import { injectable, inject, postConstruct } from '@theia/core/shared/inversify';
import { ReactWidget } from '@theia/core/lib/browser/widgets/react-widget';
import { OpenerService, open } from '@theia/core/lib/browser';
import { URI, Emitter } from '@theia/core';
import { BinaryBuffer } from '@theia/core/lib/common/buffer';
import { FileService } from '@theia/filesystem/lib/browser/file-service';
import { getZoomyCli, setDisplaySink, setLogSink, ensureRenderLibs, ensureJSZip, emitCasesChanged, emitBackendsChanged, emitSimOutput, DisplayCell } from './zoomy-cli-loader';

// The project root in the browser FS. A case is a folder here with a canonical
// `case.py` (zoomy_prepost jupytext) that is the SINGLE SOURCE OF TRUTH — the GUI
// only ever edits an open case, and every edit is written back to case.py, so the
// folder / CLI / GUI can never drift out of sync.
const PROJECT_ROOT = 'file:///zoomy/cases';
const CURRENT_CASE_KEY = 'zoomy-current-case';

declare const window: any;
/** Render markdown via marked when available, else the minimal inline fallback. */
function renderMd(s: string): string {
    try { if (window.marked?.parse) { return window.marked.parse(s || ''); } } catch { /* fall through */ }
    return mdInline(s);
}
/** Render markdown that also contains LaTeX. We pre-render each $$…$$ / $…$ span
 *  to HTML with katex.renderToString and splice it back AFTER marked, so (a) the
 *  markdown parser never mangles the `\\` matrix row separators, and (b) the baked
 *  KaTeX HTML survives React re-renders (no fragile post-render auto-typeset). */
function renderMathMd(md: string): string {
    const math: string[] = [];
    const stash = (raw: string, tex: string, display: boolean) => {
        let out = raw;
        try { if (window.katex) { out = window.katex.renderToString(tex.trim(), { displayMode: display, throwOnError: false }); } } catch { /* keep raw */ }
        math.push(out); return '@@ZMATH' + (math.length - 1) + '@@';
    };
    let s = (md || '')
        .replace(/\$\$([\s\S]*?)\$\$/g, (m, tex) => stash(m, tex, true))
        .replace(/\$([^$\n]+?)\$/g, (m, tex) => stash(m, tex, false));
    let html = renderMd(s);
    html = html.replace(/@@ZMATH(\d+)@@/g, (_m, i) => math[+i] || '');
    return html;
}

interface CardOut { cells: DisplayCell[]; stdout: string; status: string; running: boolean; }
interface TabDef { dir: string; label: string; }
const TABS: TabDef[] = [
    { dir: 'models', label: 'Model' },
    { dir: 'meshes', label: 'Mesh' },
    { dir: 'solvers', label: 'Solver' },
    { dir: 'visualizations', label: 'Visualization' },
];

/** Trigger a browser download of text content. */
function download(name: string, text: string, mime: string): void {
    downloadBlob(name, new Blob([text], { type: mime }));
}
/** Trigger a browser download of a Blob (e.g. a project .zip). */
function downloadBlob(name: string, blob: Blob): void {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
}

/** Inject the Zoomy widget stylesheet ONCE. A single `.zoomy-btn` rule defines
 *  every button's look (so buttons aren't styled inline one-by-one), and the
 *  selected/active state changes ONLY the outline (blue) — never the inner
 *  background, so labels stay readable. Also the theme-aware MBD logo backing. */
export function ensureZoomyStyles(): void {
    if (document.getElementById('zoomy-widget-styles')) { return; }
    const s = document.createElement('style');
    s.id = 'zoomy-widget-styles';
    s.textContent = `
.zoomy-btn { cursor: pointer; display: inline-flex; align-items: center; gap: 5px;
  border: 1px solid var(--theia-panel-border); border-radius: 6px; padding: 4px 11px;
  font-size: 12.5px; line-height: 1.4; background: transparent; color: var(--theia-foreground); }
.zoomy-btn:hover:not(:disabled) { background: var(--theia-list-hoverBackground); }
.zoomy-btn:disabled { opacity: .55; cursor: not-allowed; }
/* Active / selected: change ONLY the outline — inner background is untouched. */
.zoomy-btn.active { border-color: var(--theia-focusBorder, var(--theia-button-background));
  box-shadow: 0 0 0 1px var(--theia-focusBorder, var(--theia-button-background)) inset; font-weight: 600; }
.zoomy-btn.pill { border-radius: 999px; padding: 3px 12px; }
/* Primary action button (Run / Render / Create). */
.zoomy-btn.primary { border: none; background: var(--theia-button-background);
  color: var(--theia-button-foreground); font-weight: 600; }
.zoomy-btn.primary:hover:not(:disabled) { background: var(--theia-button-hoverBackground, var(--theia-button-background)); }
/* MBD + RWTH lockup: no chip on light themes; a subtle light backing only where
   needed so the dark-navy wordmarks stay legible on dark themes. */
.zoomy-mbd-logo { display: block; margin-top: 10px; }
.zoomy-mbd-logo img { width: 100%; max-width: 260px; height: auto; display: block; }
.theia-dark .zoomy-mbd-logo, body[data-theme="dark"] .zoomy-mbd-logo {
  background: #ffffff; border-radius: 6px; padding: 6px 8px; }
`;
    document.head.appendChild(s);
}

/** Derive a param schema from a card's init dict when there's no class/params
 *  schema (builtin mesh cards): infer type from each value. */
function deriveSchema(init: any): any {
    const s: any = {};
    for (const [k, v] of Object.entries(init || {})) {
        const type = typeof v === 'boolean' ? 'Boolean' : typeof v === 'number' ? (Number.isInteger(v) ? 'Integer' : 'Number') : 'String';
        s[k] = { type, default: v };
    }
    return s;
}

/** Minimal inline markdown → HTML (escaped) for card descriptions: **bold** + `code`. */
function mdInline(s: string): string {
    const esc = (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return esc.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/`([^`]+)`/g, '<code>$1</code>');
}

/** Fill a `{key}` template from the card's init dict (mesh cards use this); a
 *  template with no placeholders is returned unchanged (model/solver cards). */
function fillTemplate(tpl: string, init: any): string {
    if (!tpl) { return ''; }
    return tpl.replace(/\{(\w+)\}/g, (m, k) => (init && k in init ? String(init[k]) : m));
}
/** The runnable code for a card given its effective init (card.init + edits):
 *  the card's template ({key}-filled), else an auto import+construct. */
function cardCode(card: any, init: any): string | undefined {
    if (card.template) { return fillTemplate(card.template, init); }
    if (card.class) {
        const dot = card.class.lastIndexOf('.');
        const mod = card.class.slice(0, dot), cls = card.class.slice(dot + 1);
        const kw = Object.entries(init || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ');
        return `from ${mod} import ${cls}\n\nmodel = ${cls}(${kw})\ndisplay(model.describe())`;
    }
    return undefined; // params-only card (remote backend) — not locally runnable
}

/** The "Model configuration" GUI: the real card catalog (models/meshes/solvers/
 *  visualizations) loaded through the vendored ZoomyCLI and run on the in-browser
 *  Pyodide worker. Styled with Theia/Baukasten theme tokens; swapping in the real
 *  baukasten-ui components is later polish. */
@injectable()
export class ZoomyModelConfigWidget extends ReactWidget {
    static readonly ID = 'zoomy-modelconfig';
    protected cli: any;
    protected cardsByTab: Record<string, any[]> = {};
    protected active = 'models';
    /** Active subtab (category) per tab. */
    protected readonly activeSub: Record<string, string> = {};
    /** Card edit mode: reveals add/remove controls so a user can author a set. */
    protected editMode = false;
    protected loaded = false;
    protected error = '';
    protected kernelStatus = '';
    protected kernelReady = false;
    protected readonly outputs = new Map<string, CardOut>();
    // Param editing: the active card whose parameters show in the right-hand
    // "Zoomy Parameters" panel, the loaded schemas, and the edited values.
    protected activeParamCardId: string | undefined;
    protected activeParamDir: string | undefined;
    protected readonly schemas = new Map<string, any>();
    protected readonly edited = new Map<string, any>();
    /** True code<->card map (like the old GUI): the ACTUAL code for a card, when it
     *  differs from the card template+params (a free-form edit, e.g. a `Coupled` BC).
     *  Populated from the parsed case.py on load/editor-edit; emitted VERBATIM by
     *  gatherSpec so free-form code round-trips exactly instead of being regenerated
     *  from `cardCode(template, params)`. Keyed by card id. */
    protected readonly codeByCard = new Map<string, string>();
    /** The case.py content we last wrote (persistCase) — the echo guard so the
     *  active-case watcher only re-absorbs EXTERNAL (editor) edits, not our writes. */
    protected lastWritten: string | undefined;
    protected caseWatchStarted = false;
    // Derived governing equations per model card. We run the card's template
    // (which ends in `display(model.describe())`) and capture the display cells —
    // the same proven path as the card's Run button, so the $$-math renders as
    // KaTeX. Slow (symbolic derivation) + needs the kernel, so cached by card.id.
    protected readonly modelMath = new Map<string, { status: 'waiting' | 'loading' | 'done' | 'error'; cells: DisplayCell[]; stdout: string; key: string }>();
    /** Fired whenever the params target or its values change, so the right-hand
     *  Parameters panel re-renders. */
    protected readonly onParamsChangedEmitter = new Emitter<void>();
    readonly onParamsChanged = this.onParamsChangedEmitter.event;
    /** Set by the frontend module: reconcile the right-hand panel to the current
     *  desired state (open with the active card, or collapsed). Idempotent +
     *  last-write-wins, so rapid open/close/tab-switch can't leave it half-open. */
    paramsPanel: { sync(): void } | undefined;
    /** Set by the frontend module: reveal the bottom "Simulation" output panel. */
    simPanel: { reveal(): void } | undefined;
    /** Whether the Parameters panel should currently be open. */
    hasActiveParams(): boolean { return this.activeParamCardId !== undefined; }
    // Accordion (one expanded card) + selection (one selected card per tab).
    protected expanded: string | undefined;
    protected readonly selected: Record<string, string> = {};
    // Assembled simulation: run selected model→mesh→solver→viz in shared scope.
    protected simRan = false;
    protected simBusy = false;
    protected simStatus = '';
    protected simError: CardOut | undefined;
    protected simStopped = false;
    // Visualization: MULTI-select (checkbox) → which viewers are exported with
    // the case; plus ONE active (expanded) card that drives the right-hand
    // Parameters panel + its own Render + output.
    protected readonly selectedViz = new Set<string>();
    protected vizBusy = false;
    protected storeMeta: any;
    // Post-processing chain: enabled steps routed to a connected `postprocess`
    // backend (zoomy_prepost.steps) — reuses the CLI's runPostprocChain.
    protected readonly postprocSteps = new Set<string>();
    protected postprocNz = 10;   // lift3d vertical layers (editable when lift3d is on)
    protected postprocBusy = false;
    // Case interchange (#3/#5), project persistence (#6), backends (#4).
    protected notice = '';
    backendUrl = 'http://localhost:8080';
    connectedTags: string[] = [];
    // Case-as-source-of-truth. The GUI is only usable with an open case; every
    // edit is written back to the case folder's case.py.
    @inject(FileService) protected readonly fileService: FileService;
    @inject(OpenerService) protected readonly openerService: OpenerService;
    protected caseUri: URI | undefined;
    caseName = '';
    cases: string[] = [];
    /** Coupling folders: a parent (has coupling.yml) with its child cases, whose
     *  names are relative paths "<parent>/<child>" (they appear in `cases` too). */
    couplings: Array<{ name: string; children: string[] }> = [];
    protected static readonly COUPLING_MANIFEST = 'coupling.yml';
    protected newCaseName = '';
    protected persistTimer: any;
    /** External hook so the module can reflect connected backends in the status bar. */
    onBackendsChanged: ((tags: string[]) => void) | undefined;

    @postConstruct()
    protected init(): void {
        this.id = ZoomyModelConfigWidget.ID;
        this.title.label = 'Model configuration';
        this.title.caption = 'Zoomy — model configuration';
        this.title.iconClass = 'codicon codicon-settings-gear';
        this.title.closable = true;
        this.addClass('zoomy-modelconfig-widget');
        ensureZoomyStyles();
        this.node.style.overflow = 'auto';
        // Re-render once the math libs land so renderMathMd bakes KaTeX HTML.
        ensureRenderLibs().then(() => this.update());
        this.load();
        this.update();
    }

    protected async load(): Promise<void> {
        try {
            setLogSink((lvl, msg) => {
                console.log('[zoomy-cli]', lvl, msg);
                // Funnel the Pyodide worker's loguru output into the bottom Log panel.
                emitSimOutput({ kind: 'line', level: /error|warn|crit/i.test(lvl || '') ? 'error' : 'info', text: String(msg) });
                if (/Booting|Installing|Kernel ready|runtime ready|installing|cache|ready/i.test(msg)) {
                    this.kernelStatus = msg;
                    if (/runtime ready|Kernel ready/i.test(msg)) {
                        const wasReady = this.kernelReady; this.kernelReady = true;
                        // Now that the kernel is up, derive equations for an expanded model card.
                        if (!wasReady && this.expanded) { const c = (this.cardsByTab['models'] || []).find(x => x.id === this.expanded); if (c?.class) { this.loadModelMath(c).catch(() => { /* shown as error */ }); } }
                    }
                    this.update();
                }
            });
            this.cli = await getZoomyCli();
            // The built-in numpy (Pyodide) backend is always available — seed the
            // connected list from it so it shows connected/green from the start.
            this.refreshBackends();
            // Warm the Pyodide worker NOW (it auto-boots on creation) so the first
            // Run isn't stuck behind the cold boot + param pre-extract.
            this.cli.runCode('pass').catch(() => { /* background warm-up */ });
            for (const t of TABS) {
                try { this.cardsByTab[t.dir] = await this.cli.listCards(t.dir); }
                catch (e) { this.cardsByTab[t.dir] = []; }
            }
            this.loaded = true;
            if (!this.caseWatchStarted) { this.caseWatchStarted = true; this.startActiveCaseWatch(); }
            await this.listCases();
            // URL-autoload: ?project=<url> ships a SET of cases (a ZIP artefact,
            // like the old GUI); ?case=<url> a single case; else restore last.
            const params = new URLSearchParams(location.search);
            const projectUrl = params.get('project');
            const caseUrl = params.get('case');
            if (projectUrl) {
                await this.loadProjectFromUrl(projectUrl);
            } else if (caseUrl) {
                try { const text = await (await fetch(caseUrl)).text(); await this.newCase('imported', this.cli.parseCase(text)); } catch { /* ignore */ }
            } else {
                // Always open a case (no gate): the last one, else the first
                // existing, else a fresh default "test" case.
                const last = (() => { try { return localStorage.getItem(CURRENT_CASE_KEY); } catch { return null; } })();
                if (last && this.cases.includes(last)) { await this.openCaseByName(last); }
                else if (this.cases.length) { await this.openCaseByName(this.cases[0]); }
                else { await this.newCase('test'); }
            }
            // NO automatic backend scan on load: probing localhost ports makes
            // the browser (Firefox especially) prompt/warn about the page issuing
            // cross-origin requests unprompted, which hurts first-load UX. The
            // user scans on demand via the ↻ refresh button (zoomy.scanBackends).
        } catch (e: any) {
            this.error = e?.message || String(e);
        }
        this.update();
    }

    // === Case as the single source of truth =================================
    protected caseFileUri(name: string): URI { return new URI(PROJECT_ROOT + '/' + name + '/case.py'); }

    /** List case folders under the project root. A leaf case has a case.py; a
     *  coupling folder has coupling.yml and holds child cases one level down
     *  (their names are the relative paths "<parent>/<child>", so caseFileUri
     *  and openCaseByName keep working unchanged). */
    protected async listCases(): Promise<void> {
        try {
            const root = new URI(PROJECT_ROOT);
            if (!(await this.fileService.exists(root))) { this.cases = []; this.couplings = []; return; }
            const stat = await this.fileService.resolve(root);
            const leaves: string[] = [];
            const couplings: Array<{ name: string; children: string[] }> = [];
            for (const child of stat.children || []) {
                if (!child.isDirectory) { continue; }
                const base = child.resource.path.base;
                if (await this.fileService.exists(child.resource.resolve('case.py'))) {
                    leaves.push(base);
                } else if (await this.fileService.exists(child.resource.resolve(ZoomyModelConfigWidget.COUPLING_MANIFEST))) {
                    const sub = await this.fileService.resolve(child.resource);
                    const kids: string[] = [];
                    for (const g of sub.children || []) {
                        if (g.isDirectory && await this.fileService.exists(g.resource.resolve('case.py'))) { kids.push(base + '/' + g.resource.path.base); }
                    }
                    couplings.push({ name: base, children: kids.sort() });
                    leaves.push(...kids);
                }
            }
            // Auto-clean coupling folders left empty (e.g. all children removed one by one).
            for (const e of couplings.filter(c => c.children.length === 0)) {
                try { await this.fileService.delete(root.resolve(e.name), { recursive: true, useTrash: false }); } catch { /* ignore */ }
            }
            this.cases = leaves.sort();
            this.couplings = couplings.filter(c => c.children.length > 0).sort((a, b) => a.name.localeCompare(b.name));
        } catch { this.cases = []; this.couplings = []; }
        emitCasesChanged();
    }

    /** True when a case name is a coupled child (relative path "<parent>/<child>"). */
    isCoupledChild(name: string): boolean { return name.includes('/'); }

    /** Form a coupling from ≥2 top-level cases: create cases/<coupling>/, move each
     *  selected case in as a child, and write the manifest + a placeholder
     *  precice-config.xml one layer above the children (preCICE's expectation). */
    async coupleCases(names: string[]): Promise<void> {
        const kids = [...new Set(names)].filter(n => !this.isCoupledChild(n));
        if (kids.length < 2) { return; }
        const taken = new Set([...this.cases.map(c => c.split('/')[0]), ...this.couplings.map(c => c.name)]);
        let cname = 'coupled'; let i = 2;
        while (taken.has(cname)) { cname = 'coupled_' + i; i++; }
        const root = new URI(PROJECT_ROOT);
        const parent = root.resolve(cname);
        try {
            await this.fileService.createFolder(parent);
            for (const n of kids) { await this.fileService.move(root.resolve(n), parent.resolve(n), { overwrite: false }); }
            await this.fileService.createFile(parent.resolve(ZoomyModelConfigWidget.COUPLING_MANIFEST),
                BinaryBuffer.fromString(this.couplingManifest(cname, kids)), { overwrite: true });
            await this.fileService.createFile(parent.resolve('precice-config.xml'),
                BinaryBuffer.fromString(this.placeholderPreciceConfig(kids)), { overwrite: true });
            await this.listCases();
            const first = cname + '/' + kids[0];
            if (this.cases.includes(first)) { await this.openCaseByName(first); }
            this.update();
        } catch (e: any) { this.setNotice('Couple failed: ' + (e?.message || e)); }
    }

    /** Move a coupled child back to top level; dissolve the parent if <2 remain. */
    async decoupleCase(childRel: string): Promise<void> {
        if (!this.isCoupledChild(childRel)) { return; }
        const parentName = childRel.split('/')[0];
        const root = new URI(PROJECT_ROOT);
        const uniq = (want: string): string => {
            const taken = new Set([...this.cases.map(c => c.split('/')[0]), ...this.couplings.map(c => c.name)]);
            let d = want; let j = 2; while (taken.has(d)) { d = want + '_' + j; j++; } return d;
        };
        try {
            await this.fileService.move(root.resolve(childRel), root.resolve(uniq(childRel.split('/')[1])), { overwrite: false });
            await this.listCases();
            const cp = this.couplings.find(c => c.name === parentName);
            if (cp && cp.children.length < 2) {
                for (const rem of cp.children) { await this.fileService.move(root.resolve(rem), root.resolve(uniq(rem.split('/')[1])), { overwrite: false }); }
                if (await this.fileService.exists(root.resolve(parentName))) { await this.fileService.delete(root.resolve(parentName), { recursive: true, useTrash: false }); }
                await this.listCases();
            }
            this.update();
        } catch (e: any) { this.setNotice('Disconnect failed: ' + (e?.message || e)); }
    }

    protected couplingManifest(name: string, children: string[]): string {
        // Human-readable YAML (hand-written; the GUI only needs a few keys).
        const lines = ['# Zoomy coupling manifest', 'coupling_id: ' + name, 'scheme: parallel-explicit',
            'canonical_output: ' + children[0] + '   # participant whose store the GUI opens', 'participants:'];
        for (const c of children) { lines.push('  - ' + c); }
        return lines.join('\n') + '\n';
    }
    protected placeholderPreciceConfig(children: string[]): string {
        // Placeholder — the generator (Slice 3) fills the real data/mesh/exchange.
        return ['<?xml version="1.0" encoding="UTF-8" ?>', '<precice-configuration>',
            '  <!-- Placeholder coupling contract for participants: ' + children.join(', ') + ' -->',
            '  <!-- The coupled-case generator fills the meshes, exchanged data (b,h,u,v,w,p),', '       mapping and m2n:sockets exchange-directory at run time. -->',
            '</precice-configuration>', ''].join('\n');
    }

    /** Open a coupling's config surface: for now, its precice-config.xml (the
     *  coupling contract) in the editor — the "preCICE card" of the coupling. */
    async openCoupling(name: string): Promise<void> {
        try {
            const uri = new URI(PROJECT_ROOT + '/' + name + '/precice-config.xml');
            if (await this.fileService.exists(uri)) { await open(this.openerService, uri); }
        } catch (e: any) { this.setNotice('Open coupling failed: ' + (e?.message || e)); }
    }

    protected uniqueTopName(want: string): string {
        const taken = new Set([...this.cases.map(c => c.split('/')[0]), ...this.couplings.map(c => c.name)]);
        let d = want; let j = 2; while (taken.has(d)) { d = want + '_' + j; j++; } return d;
    }

    /** Uncouple-all: move every child of a coupling back to top level and delete
     *  the now-empty parent folder (also the fix for a stray empty coupling). */
    async dissolveCoupling(name: string): Promise<void> {
        const cp = this.couplings.find(c => c.name === name);
        const root = new URI(PROJECT_ROOT);
        try {
            if (cp) { for (const ch of cp.children) { await this.fileService.move(root.resolve(ch), root.resolve(this.uniqueTopName(ch.split('/')[1])), { overwrite: false }); } }
            if (await this.fileService.exists(root.resolve(name))) { await this.fileService.delete(root.resolve(name), { recursive: true, useTrash: false }); }
            await this.listCases(); this.update();
        } catch (e: any) { this.setNotice('Uncouple failed: ' + (e?.message || e)); }
    }

    /** Rename a case folder (keeps a coupled child inside its parent). */
    async renameCase(oldRel: string, newName: string): Promise<void> {
        const clean = (newName || '').trim().replace(/[^\w.-]/g, '_');
        if (!clean) { return; }
        const parts = oldRel.split('/');
        const prefix = parts.length > 1 ? parts.slice(0, -1).join('/') + '/' : '';
        const newRel = prefix + clean;
        if (newRel === oldRel) { return; }
        const root = new URI(PROJECT_ROOT);
        try {
            if (this.cases.includes(newRel)) { this.setNotice('A case named "' + clean + '" already exists here.'); return; }
            await this.fileService.move(root.resolve(oldRel), root.resolve(newRel), { overwrite: false });
            try { if (localStorage.getItem(CURRENT_CASE_KEY) === oldRel) { localStorage.setItem(CURRENT_CASE_KEY, newRel); } } catch { /* ignore */ }
            const wasActive = this.caseName === oldRel;
            await this.listCases();
            if (wasActive && this.cases.includes(newRel)) { await this.openCaseByName(newRel); }
            this.update();
        } catch (e: any) { this.setNotice('Rename failed: ' + (e?.message || e)); }
    }

    /** Run a coupling on the foam backend: POST the participants to /couple,
     *  where build_coupled_bundle expands the OF-cases + shared precice-config
     *  and launches both. Gated on a connected "foam" backend — all participants
     *  run there. (Type is inferred from the child name/spec: vof vs sme.) */
    async runCoupling(name: string): Promise<void> {
        const cp = this.couplings.find(c => c.name === name);
        if (!cp || cp.children.length < 2) { this.setNotice('Coupling "' + name + '" needs at least 2 participants.'); return; }
        if (!this.tagMatches('OpenFOAM', this.connectedTags)) {
            this.setNotice('Connect a "OpenFOAM" backend to run the coupling "' + name + '" — all its participants run there (use the ↻ scan / Connect backend).');
            return;
        }
        this.simPanel?.reveal();
        emitSimOutput({ kind: 'line', level: 'info', text: '▶ Running coupling "' + name + '" (' + cp.children.length + ' participants) on the foam backend…' });
        try {
            const participants = cp.children.map(child => {
                const leaf = String(child.split('/').pop());
                return { name: leaf, type: /vof/i.test(leaf) ? 'vof' : 'sme' };
            });
            const res = await this.cli.submitCoupling({ tag: 'OpenFOAM', coupling_id: name, scheme: 'parallel-explicit', participants,
                onStatus: (s: any) => { const m = s?.message || s?.state || (typeof s === 'string' ? s : null); if (m) { emitSimOutput({ kind: 'line', level: 'stdout', text: String(m) }); } } });
            emitSimOutput({ kind: 'line', level: 'ok', text: '✓ Coupling "' + name + '" submitted — ' + ((res?.jobs || []).length) + ' participant job(s) on foam.' });
            this.setNotice('Coupling "' + name + '" running on the foam backend.');
        } catch (e: any) {
            emitSimOutput({ kind: 'line', level: 'error', text: '✗ Coupling run failed: ' + (e?.message || e) });
            this.setNotice('Coupling run failed: ' + (e?.message || e));
        }
    }

    /** Create a new case folder with a case.py, then open it. If a spec is given
     *  (import), use it; otherwise start from the first runnable card in each tab. */
    async newCase(name: string, spec?: any): Promise<void> {
        const clean = (name || 'case').trim().replace(/[^a-zA-Z0-9_-]+/g, '_') || 'case';
        this.selected['models'] = ''; this.selected['meshes'] = ''; this.selected['solvers'] = ''; this.selected['visualizations'] = '';
        this.edited.clear(); this.codeByCard.clear();
        if (spec) { this.applySpec(spec); }
        else {
            for (const dir of ['models', 'meshes', 'solvers']) { const c = this.pickedCard(dir); if (c) { this.selected[dir] = c.id; } }
        }
        // Seed the selected visualization viewer (single-select).
        const firstViz = (this.cardsByTab['visualizations'] || []).find(c => c.snippet);
        if (firstViz) { this.selected['visualizations'] = firstViz.id; this.selectedViz.clear(); this.selectedViz.add(firstViz.id); }
        this.caseUri = this.caseFileUri(clean); this.caseName = clean;
        await this.persistCase();
        await this.listCases();
        try { localStorage.setItem(CURRENT_CASE_KEY, clean); } catch { /* ignore */ }
        this.setNotice('Created case "' + clean + '".'); this.update();
        this.onBackendsChanged?.(this.connectedTags);
    }

    /** Return to the gate (no open case) — e.g. to create another case. */
    closeCase(): void { this.caseUri = undefined; this.caseName = ''; this.newCaseName = ''; this.listCases(); this.update(); }

    /** Re-scan the case folders from the FS (public: the left panel calls this
     *  on activation so Explorer copy/paste of a case shows up). */
    async rescan(): Promise<void> { await this.listCases(); this.update(); }

    /** Duplicate a case folder (case.py + outputs) under a fresh name, then open
     *  it. Exposed as a per-case action in the left panel. */
    async duplicateCase(name: string): Promise<void> {
        if (!name) { return; }
        try {
            const srcDir = new URI(PROJECT_ROOT + '/' + name);
            let dst = name + '_copy', i = 2;
            while (this.cases.includes(dst)) { dst = name + '_copy' + i; i++; }
            await this.fileService.copy(srcDir, new URI(PROJECT_ROOT + '/' + dst), { overwrite: false });
            await this.listCases();
            await this.openCaseByName(dst);
            this.setNotice('Duplicated "' + name + '" → "' + dst + '".'); this.update();
        } catch (e: any) { this.setNotice('Duplicate failed: ' + (e?.message || e)); }
    }

    /** Delete a case folder (the ✕ in the left panel). If it was the active case,
     *  fall back to another existing case, else a fresh default — a case is
     *  always open. */
    async removeCase(name: string): Promise<void> {
        if (!name) { return; }
        try {
            const dir = new URI(PROJECT_ROOT + '/' + name);
            if (await this.fileService.exists(dir)) { await this.fileService.delete(dir, { recursive: true, useTrash: false }); }
            try { if (localStorage.getItem(CURRENT_CASE_KEY) === name) { localStorage.removeItem(CURRENT_CASE_KEY); } } catch { /* ignore */ }
            const wasActive = this.caseName === name;
            await this.listCases();
            if (wasActive) {
                this.caseUri = undefined; this.caseName = '';
                if (this.cases.length) { await this.openCaseByName(this.cases[0]); }
                else { await this.newCase('test'); }
            }
            this.setNotice('Removed case "' + name + '".'); this.update();
        } catch (e: any) { this.setNotice('Remove case failed: ' + (e?.message || e)); }
    }

    /** Ship a SET of cases by URL (like the old GUI's ?project=). Resolves the
     *  URL (zenodo:<id> / direct .zip), unzips, and materializes each .py entry
     *  as a case folder — then opens the first. */
    async loadProjectFromUrl(url: string): Promise<void> {
        this.setNotice('Downloading project artefact…'); this.update();
        try {
            const zipUrl = await this.resolveArtefactUrl(url);
            const buf = await (await fetch(zipUrl)).arrayBuffer();
            await this.loadProjectFromZip(buf);
        } catch (e: any) { this.setNotice('Project load failed: ' + (e?.message || e)); this.update(); }
    }
    /** Resolve an artefact URL: zenodo:<recordId>[/file] → the record's ZIP;
     *  anything else is used as-is (GitHub release asset, raw URL, …). */
    protected async resolveArtefactUrl(url: string): Promise<string> {
        const zen = url.match(/^zenodo:(\d+)(?:\/(.+))?$/);
        if (zen) {
            const rec = await (await fetch('https://zenodo.org/api/records/' + zen[1])).json();
            const files: any[] = rec.files || [];
            const target = (zen[2] && files.find(f => f.key === zen[2])) || files.find(f => /\.zip$/i.test(f.key)) || files[0];
            const link = target?.links?.self || target?.links?.download;
            if (!link) { throw new Error('No file found in Zenodo record ' + zen[1]); }
            return link;
        }
        return url;
    }

    async openCaseByName(name: string): Promise<void> {
        try {
            const uri = this.caseFileUri(name);
            const content = await this.fileService.read(uri);
            this.applySpec(this.cli.parseCase(content.value));
            this.caseUri = uri; this.caseName = name;
            try { localStorage.setItem(CURRENT_CASE_KEY, name); } catch { /* ignore */ }
            // Single source of truth: broadcast so the left Zoomy panel highlights
            // THIS case (keeps left panel ⇄ config selection in sync after a refresh).
            emitCasesChanged();
            this.setNotice('Opened case "' + name + '".'); this.update();
        } catch (e: any) { this.setNotice('Open case failed: ' + (e?.message || e)); }
    }

    /** Recompose the spec from the current selection and write it back to case.py
     *  — keeping the folder the single source of truth. Debounced by schedulePersist. */
    async persistCase(): Promise<void> {
        if (!this.caseUri) { return; }
        try {
            const dir = this.caseUri.parent;
            if (!(await this.fileService.exists(dir))) { await this.fileService.createFolder(dir); }
            const spec = await this.gatherSpec();
            const py = this.cli.exportCase(spec, 'py');
            this.lastWritten = py;   // echo guard: the watcher must ignore our own write
            await this.fileService.write(this.caseUri, py);
        } catch (e: any) { this.setNotice('Save failed: ' + (e?.message || e)); }
    }

    /** Watch the ACTIVE case.py: when it changes on disk from something OTHER than
     *  our own persist (i.e. an editor edit), re-parse it so codeByCard absorbs the
     *  new code — a free-form edit (e.g. adding a `Coupled` BC) then round-trips
     *  through gatherSpec verbatim instead of being regenerated from the template. */
    protected startActiveCaseWatch(): void {
        this.fileService.onDidFilesChange(async event => {
            if (!this.caseUri) { return; }
            const target = this.caseUri.toString();
            if (!event.changes.some(c => c.resource.toString() === target)) { return; }
            let content: string;
            try { content = (await this.fileService.read(this.caseUri)).value; } catch { return; }
            if (content === this.lastWritten) { return; }   // our own persist write
            this.applySpec(this.cli.parseCase(content));     // absorbs the edited code
            this.update();
        });
    }

    /** Capture each selected card's ACTUAL code as an override IN codeByCard when it
     *  differs from the card's template+params output; drop the override when it
     *  matches (a template-driven card whose params still drive it). Keeps free-form
     *  code (the real code<->card map) while letting param forms regenerate. */
    protected absorbCode(spec: any): void {
        const pairs: Array<[string, any]> = [
            ['models', spec?.model?.code], ['meshes', spec?.mesh?.code],
            ['solvers', spec?.run?.code], ['visualizations', spec?.visualization?.code],
        ];
        const norm = (s: any) => String(s || '').replace(/\s+$/, '');
        for (const [dir, code] of pairs) {
            const cardId = this.selected[dir];
            if (!cardId || code == null) { continue; }
            const card = (this.cardsByTab[dir] || []).find(c => c.id === cardId);
            if (!card) { continue; }
            const template = cardCode(card, this.mergedInit(card)) || '';
            if (norm(code) !== norm(template)) { this.codeByCard.set(cardId, String(code)); }
            else { this.codeByCard.delete(cardId); }
        }
    }

    /** The code emitted for a card: a stored free-form override wins; else the
     *  card template filled with its current params. */
    protected cardCodeFor(card: any): string {
        if (card && this.codeByCard.has(card.id)) { return this.codeByCard.get(card.id) as string; }
        return cardCode(card, this.mergedInit(card)) || '';
    }
    /** "Edit" on a model/mesh/solver card: flush pending edits, then open the
     *  case's case.py in the editor (auto-revealed in the Explorer). The case is
     *  the single source of truth, so editing = editing case.py directly. */
    async editCardFile(dir?: string): Promise<void> {
        if (!this.caseUri) { return; }
        try {
            await this.persistCase();
            const options: any = {};
            if (dir) {
                const section = ({ models: 'model', meshes: 'mesh', solvers: 'run', visualizations: 'visualization' } as any)[dir];
                const line = await this.sectionLine(section);
                if (line >= 0) { options.selection = { start: { line, character: 0 }, end: { line, character: 0 } }; options.mode = 'reveal'; }
            }
            await open(this.openerService, this.caseUri, options);
        } catch (e: any) { this.setNotice('Open in editor failed: ' + (e?.message || e)); }
    }
    /** 0-based line of a case.py section (from its `# %% … zoomy={…}` marker). */
    protected async sectionLine(section: string): Promise<number> {
        if (!this.caseUri || !section) { return -1; }
        try {
            const lines = (await this.fileService.read(this.caseUri)).value.split('\n');
            for (let i = 0; i < lines.length; i++) {
                const m = /#\s*%%.*zoomy=(\{.*\})\s*$/.exec(lines[i]);
                if (!m) { continue; }
                let meta: any; try { meta = JSON.parse(m[1]); } catch { continue; }
                if (meta.section === section || meta.role === section) { return i; }
            }
        } catch { /* ignore */ }
        return -1;
    }
    /** "Open in Notebook Mode": export the current case to a .ipynb next to its
     *  case.py and open it in the notebook editor. The case stays the source of
     *  truth (case.py); the notebook is a live, runnable view of the same spec. */
    async openInNotebook(): Promise<void> {
        if (!this.caseUri) { this.setNotice('Open a case first.'); return; }
        try {
            await this.persistCase();
            const spec = await this.gatherSpec();
            const ipynb = this.cli.exportCase(spec, 'ipynb');
            const nbUri = this.caseUri.parent.resolve('case.ipynb');
            await this.fileService.write(nbUri, ipynb);
            await open(this.openerService, nbUri);
        } catch (e: any) { this.setNotice('Open in notebook failed: ' + (e?.message || e)); }
    }
    protected schedulePersist(): void {
        if (!this.caseUri) { return; }
        clearTimeout(this.persistTimer);
        this.persistTimer = setTimeout(() => this.persistCase(), 600);
    }

    /** card.init overlaid with the user's edits from the param form. */
    protected mergedInit(card: any): any { return { ...(card.init || {}), ...(this.edited.get(card.id) || {}) }; }

    /** Open the right-hand Parameters panel for a card (or toggle it closed if it
     *  is already the active one). Also selects the card. The schema loads async;
     *  a preview (derived from the card's init) shows immediately so the panel is
     *  never blank — this is what fixes "Parameters doesn't open for the model". */
    async openParams(card: any, dir: string): Promise<void> {
        if (this.activeParamCardId === card.id) {
            this.activeParamCardId = undefined; this.activeParamDir = undefined;
            this.paramsPanel?.sync(); this.onParamsChangedEmitter.fire(); this.update(); return;
        }
        this.selected[dir] = card.id;
        this.activeParamCardId = card.id; this.activeParamDir = dir;
        // Show an immediate preview schema so the panel has content right away.
        if (dir === 'visualizations') { this.schemas.set(card.id, this.vizParamSchema()); }
        else if (!this.schemas.has(card.id) && !card.params) { this.schemas.set(card.id, deriveSchema(card.init)); }
        this.paramsPanel?.sync(); this.onParamsChangedEmitter.fire(); this.update();
        await this.loadSchema(card, dir);
    }
    // === Card authoring (edit mode) — via the cli catalog overlay ============
    /** Add a card to the active tab: duplicate the selected/first card (a good
     *  starting point to customise) into the active category. Persisted in the
     *  catalog overlay so it survives reloads and can be saved with the project. */
    async addCardToActive(): Promise<void> {
        const dir = this.active;
        try {
            const base = this.pickedCard(dir) || (this.cardsByTab[dir] || [])[0];
            const id = 'user-' + dir.replace(/s$/, '') + '-' + Date.now().toString(36);
            const card = base
                ? { ...JSON.parse(JSON.stringify(base)), id, title: (base.title || 'Card') + ' (copy)', category: this.activeSub[dir] || base.category || 'Custom', source: 'user' }
                : { id, title: 'New card', category: this.activeSub[dir] || 'Custom', source: 'user' };
            const ov = await this.cli.readCatalogOverlay(dir);
            ov.added = (ov.added || []).filter((c: any) => c.id !== id); ov.added.push(card);
            await this.cli.writeCatalogOverlay(dir, ov);
            this.cardsByTab[dir] = await this.cli.listCards(dir);
            this.expanded = id; this.selected[dir] = id;
            this.setNotice('Added card "' + card.title + '". Edit its Parameters, then Save project to keep it.');
            this.update();
        } catch (e: any) { this.setNotice('Add card failed: ' + (e?.message || e)); }
    }
    /** Remove a card from the active tab (a user-added one is dropped; a shipped
     *  one is hidden via the overlay's `removed` list). */
    async removeCard(card: any): Promise<void> {
        const dir = this.active;
        try {
            const ov = await this.cli.readCatalogOverlay(dir);
            if ((ov.added || []).some((c: any) => c.id === card.id)) { ov.added = ov.added.filter((c: any) => c.id !== card.id); }
            else { ov.removed = [...new Set([...(ov.removed || []), card.id])]; }
            await this.cli.writeCatalogOverlay(dir, ov);
            this.cardsByTab[dir] = await this.cli.listCards(dir);
            if (this.expanded === card.id) { this.expanded = undefined; }
            if (this.selected[dir] === card.id) { this.selected[dir] = ''; }
            this.setNotice('Removed card "' + (card.title || card.id) + '".');
            this.update();
        } catch (e: any) { this.setNotice('Remove card failed: ' + (e?.message || e)); }
    }

    /** Rebuild EVERY visualization card's field/time schema from the current
     *  store — the fields + number of snapshots are only known after a run, so
     *  this is called whenever a run/render updates the store (one mechanism for
     *  all viz cards, no per-card wiring). */
    protected refreshVizParams(): void {
        const schema = this.vizParamSchema();
        for (const c of (this.cardsByTab['visualizations'] || [])) { this.schemas.set(c.id, schema); }
        this.onParamsChangedEmitter.fire();
    }
    /** Read the installed store's fields + n_snapshots STRAIGHT from the worker
     *  (the `store` global that open_hdf5 installs for local AND remote runs),
     *  so the field selector + time slider are always accurate — independent of
     *  whatever store_meta a given run returned. Then refresh the viz params. */
    async refreshStoreMeta(): Promise<void> {
        try {
            const code = [
                'import json as _zj',
                '_zs = store',   // always defined in the exec scope (None or a SimulationStore)
                'print("__ZM__" + (_zj.dumps({"fields": list(_zs.field.keys()), "n_snapshots": int(_zs.n_snapshots)}) if _zs is not None else "{}"))',
            ].join('\n');
            const res = await this.cli.runCode(code);
            // Prefer the engine's automatic store_meta; else parse the printed marker.
            if ((res as any)?.store_meta && Array.isArray((res as any).store_meta.fields) && (res as any).store_meta.fields.length) {
                this.storeMeta = (res as any).store_meta;
            } else {
                const m = String(res?.output || '').match(/__ZM__(\{[\s\S]*?\})/);
                if (m) { const meta = JSON.parse(m[1]); if (meta && Array.isArray(meta.fields) && meta.fields.length) { this.storeMeta = meta; } }
            }
        } catch { /* ignore — keep whatever store_meta a run set */ }
        this.refreshVizParams();
    }
    /** Field selector + time slider for a visualization, from the last run's store
     *  (store_meta.fields / n_snapshots). Editable like any card parameter and
     *  passed to the viz snippet — no bespoke visualizer code needed. */
    protected vizParamSchema(): any {
        const fields: string[] = (this.storeMeta?.fields || this.storeMeta?.field_names || []) as string[];
        const nSnap = Math.max(1, Number(this.storeMeta?.n_snapshots || this.storeMeta?.n_steps || 1));
        return {
            field: { type: 'Selector', objects: fields, default: fields[0] ?? null, doc: 'Which stored field to plot (from the last run).' },
            time_step: { type: 'Integer', default: 0, bounds: [0, nSnap - 1], step: 1, widget: 'slider', doc: 'Snapshot index (time) to plot.' },
        };
    }
    /** Load a card's real parameter schema (class introspection via the worker;
     *  inline `params` need no worker; builtin cards expose their init). */
    protected async loadSchema(card: any, dir?: string): Promise<void> {
        let schema: any;
        if (dir === 'visualizations') { schema = this.vizParamSchema(); }
        else if (card.params) { schema = card.params; }
        else if (card.class) {
            // extract_param_schema returns a JSON STRING (json.dumps) — parse it,
            // else res?.params is undefined and we lose every introspected param.
            try {
                const res = await this.cli.extractParams(card.class, card.init || {});
                const parsed = typeof res === 'string' ? JSON.parse(res) : res;
                schema = (parsed && parsed.params) ? parsed.params : deriveSchema(card.init);
            } catch { schema = deriveSchema(card.init); }
        } else { schema = deriveSchema(card.init); }
        this.schemas.set(card.id, schema);
        this.onParamsChangedEmitter.fire(); this.update();
    }
    /** The card currently targeted by the Parameters panel, with its tab dir. */
    activeParamTarget(): { card: any; dir: string } | undefined {
        if (!this.activeParamCardId || !this.activeParamDir) { return undefined; }
        const card = (this.cardsByTab[this.activeParamDir] || []).find(c => c.id === this.activeParamCardId);
        return card ? { card, dir: this.activeParamDir } : undefined;
    }
    /** Close the panel from tab switches / the panel's own close button. */
    closeParams(): void {
        if (this.activeParamCardId === undefined) { return; }
        this.activeParamCardId = undefined; this.activeParamDir = undefined;
        this.paramsPanel?.sync(); this.onParamsChangedEmitter.fire(); this.update();
    }

    protected setParam(card: any, name: string, value: any): void {
        const e = this.edited.get(card.id) || {}; e[name] = value; this.edited.set(card.id, e); this.update();
        this.onParamsChangedEmitter.fire();
        this.schedulePersist();
    }

    protected async runCard(card: any): Promise<void> {
        const code = cardCode(card, this.mergedInit(card));
        if (!code) { return; }
        const out: CardOut = { cells: [], stdout: '', status: 'running', running: true };
        this.outputs.set(card.id, out); this.update();
        setDisplaySink(cell => { out.cells.push(cell); this.update(); });
        try {
            const res = await this.cli.runCode(code);
            out.stdout = res?.output || ''; out.status = res?.status || 'success';
        } catch (e: any) {
            out.status = 'error'; out.stdout = e?.message || String(e);
        } finally {
            setDisplaySink(undefined); out.running = false; this.update();
        }
    }

    protected renderCell(cell: DisplayCell, key: string): React.ReactNode {
        const h = React.createElement;
        const mime = cell.mime || 'text/plain';
        // Markdown/LaTeX describe() output: render markdown; KaTeX typesets the
        // $$…$$ after update (onUpdateRequest → typeset).
        if (mime === 'text/markdown') { return h('div', { key, className: 'zoomy-md', dangerouslySetInnerHTML: { __html: renderMathMd(cell.content) } }); }
        if (mime === 'text/x-latex' || mime === 'text/latex') { return h('div', { key, className: 'zoomy-md', dangerouslySetInnerHTML: { __html: renderMathMd('$$' + cell.content + '$$') } }); }
        if (mime === 'text/html') { return h('div', { key, className: 'zoomy-md', dangerouslySetInnerHTML: { __html: cell.content } }); }
        if (mime === 'image/svg+xml') { return h('div', { key, dangerouslySetInnerHTML: { __html: cell.content } }); }
        if (mime === 'image/png') { return h('img', { key, src: 'data:image/png;base64,' + cell.content, style: { maxWidth: '100%' } }); }
        return h('pre', { key, style: { margin: '2px 0', whiteSpace: 'pre-wrap', fontSize: 12 } }, cell.content);
    }

    protected renderParamForm(card: any): React.ReactNode {
        const h = React.createElement;
        const schema = this.schemas.get(card.id);
        if (!schema) { return h('div', { style: { fontSize: 12, color: 'var(--theia-descriptionForeground)', marginTop: 8 } }, 'Introspecting parameters…'); }
        const names = Object.keys(schema);
        if (!names.length) { return h('div', { style: { fontSize: 12, color: 'var(--theia-descriptionForeground)', marginTop: 8 } }, 'No editable parameters.'); }
        const init = this.mergedInit(card);
        const rowS: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 10, margin: '4px 0' };
        const labelS: React.CSSProperties = { width: 150, fontSize: 12.5, color: 'var(--theia-foreground)' };
        const inputS: React.CSSProperties = { background: 'var(--theia-input-background)', color: 'var(--theia-input-foreground)', border: '1px solid var(--theia-input-border, var(--theia-panel-border))', borderRadius: 4, padding: '3px 6px', fontSize: 12.5 };
        const field = (name: string): React.ReactNode => {
            const p = schema[name] || {};
            const type = p.type || 'Number';
            const cur = init[name] !== undefined ? init[name] : p.default;
            let input: React.ReactNode;
            if (type === 'Boolean') {
                input = h('input', { type: 'checkbox', checked: !!cur, onChange: (e: any) => this.setParam(card, name, e.target.checked) });
            } else if ((type === 'Selector' || type === 'ObjectSelector') && Array.isArray(p.objects)) {
                input = h('select', { style: inputS, value: String(cur), onChange: (e: any) => this.setParam(card, name, e.target.value) },
                    p.objects.map((o: any) => h('option', { key: String(o), value: String(o) }, String(o))));
            } else if (type === 'String') {
                input = h('input', { type: 'text', style: inputS, value: cur == null ? '' : String(cur), onChange: (e: any) => this.setParam(card, name, e.target.value) });
            } else if (p.widget === 'slider' && Array.isArray(p.bounds) && p.bounds[1] != null) {
                const lo = p.bounds[0] ?? 0, hi = p.bounds[1];
                input = h('span', { style: { display: 'inline-flex', alignItems: 'center', gap: 8 } },
                    h('input', { type: 'range', min: lo, max: hi, step: p.step || 1, value: cur == null ? lo : cur, style: { width: 130 }, onChange: (e: any) => this.setParam(card, name, parseInt(e.target.value, 10)) }),
                    h('span', { style: { fontSize: 12, minWidth: 44, color: 'var(--theia-foreground)' } }, (cur == null ? lo : cur) + ' / ' + hi));
            } else if (type === 'Integer' || type === 'Number') {
                const step = type === 'Integer' ? 1 : (p.step || 'any');
                input = h('input', { type: 'number', step, style: inputS, value: cur == null ? '' : cur, onChange: (e: any) => { const v = e.target.value; this.setParam(card, name, v === '' ? null : (type === 'Integer' ? parseInt(v, 10) : parseFloat(v))); } });
            } else {
                // Structural / composed params (Dict, List, ClassSelector, Parameter):
                // shown read-only — they're built in code, not via a simple input.
                const short = cur == null ? '—'
                    : (Array.isArray(cur) ? '[' + cur.length + ' item' + (cur.length === 1 ? '' : 's') + ']'
                    : (typeof cur === 'object' ? '⟨' + (p.type || 'object') + '⟩' : String(cur)));
                input = h('span', { style: { fontSize: 12, color: 'var(--theia-descriptionForeground)', fontStyle: 'italic' }, title: 'Composed in code (' + (p.type || 'object') + ')' }, short);
            }
            const docTip = p.doc ? h('span', { title: p.doc, style: { fontSize: 11, color: 'var(--theia-descriptionForeground)', cursor: 'help' } }, ' ⓘ') : null;
            return h('div', { key: name, style: rowS },
                h('label', { style: labelS, title: p.doc || '' }, name),
                input,
                (p.bounds && Array.isArray(p.bounds)) ? h('span', { style: { fontSize: 11, color: 'var(--theia-descriptionForeground)' } }, '[' + (p.bounds[0] ?? '') + ', ' + (p.bounds[1] ?? '') + ']') : null,
                docTip);
        };
        return h('div', { style: { marginTop: 10, borderTop: '1px dashed var(--theia-panel-border)', paddingTop: 8 } }, names.map(field));
    }

    /** Derive a model card's governing equations via describeModel (Pyodide,
     *  ~30–60s + kernel boot). Cached by card.id; keyed by params so a manual
     *  recompute picks up parameter changes. */
    async loadModelMath(card: any, force = false): Promise<void> {
        if (!card.class) { return; }
        const code = cardCode(card, this.mergedInit(card));
        if (!code) { return; }
        const key = JSON.stringify(this.mergedInit(card));
        const cur = this.modelMath.get(card.id);
        if (!force && cur && (cur.status === 'loading' || cur.status === 'done') && cur.key === key) { return; }
        if (!this.kernelReady) { this.modelMath.set(card.id, { status: 'waiting', cells: [], stdout: '', key }); this.update(); return; }
        const entry = { status: 'loading' as const, cells: [] as DisplayCell[], stdout: '', key };
        this.modelMath.set(card.id, entry); this.update();
        // Capture the template's display(model.describe()) cells (the $$-math).
        setDisplaySink(cell => { entry.cells.push(cell); this.update(); });
        try {
            const res = await this.cli.runCode(code);
            this.modelMath.set(card.id, { status: res?.status === 'error' ? 'error' : 'done', cells: entry.cells, stdout: res?.output || '', key });
        } catch (e: any) {
            this.modelMath.set(card.id, { status: 'error', cells: entry.cells, stdout: e?.message || String(e), key });
        } finally {
            setDisplaySink(undefined); this.update();
        }
    }
    /** The "Governing equations" block shown in an expanded model card. */
    protected renderModelMath(card: any): React.ReactNode {
        const h = React.createElement;
        if (!card.class) { return null; }
        const box: React.CSSProperties = { marginTop: 10, border: '1px solid var(--theia-panel-border)', borderRadius: 6, padding: '8px 10px', background: 'var(--theia-editorWidget-background)' };
        const label: React.CSSProperties = { fontSize: 11, textTransform: 'uppercase', letterSpacing: '.05em', color: 'var(--theia-descriptionForeground)', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 8 };
        const m = this.modelMath.get(card.id);
        const recompute = h('button', { title: 'Recompute with current parameters', onClick: () => this.loadModelMath(card, true), style: { marginLeft: 'auto', cursor: 'pointer', border: 'none', background: 'transparent', color: 'var(--theia-descriptionForeground)', fontSize: 11 } }, h('span', { className: 'codicon codicon-refresh' }));
        const head = (extra?: React.ReactNode) => h('div', { style: label }, 'Governing equations', extra, m?.status === 'done' ? recompute : null);
        if (!m || m.status === 'waiting') {
            return h('div', { style: box }, head(),
                h('div', { style: { fontSize: 12, color: 'var(--theia-descriptionForeground)' } },
                    this.kernelReady ? 'Deriving…' : 'Waiting for the in-browser kernel to boot (first time ~2–3 min), then the equations are derived symbolically.'));
        }
        const spin = m.status === 'loading';
        const cells = m.cells.length ? h('div', { style: { fontSize: 12.5, overflowX: 'auto' } }, m.cells.map((c, i) => this.renderCell(c, 'mm' + i))) : null;
        if (spin && !m.cells.length) { return h('div', { style: box }, head(h('span', { className: 'codicon codicon-loading codicon-modifier-spin', style: { fontSize: 11 } })), h('div', { style: { fontSize: 12, color: 'var(--theia-descriptionForeground)' } }, 'Deriving the equations symbolically (~30–60s)…')); }
        if (m.status === 'error' && !m.cells.length) { return h('div', { style: box }, head(), h('pre', { style: { fontSize: 11, whiteSpace: 'pre-wrap', color: 'var(--theia-errorForeground)' } }, m.stdout)); }
        return h('div', { style: box }, head(spin ? h('span', { className: 'codicon codicon-loading codicon-modifier-spin', style: { fontSize: 11 } }) : undefined), cells);
    }

    /** A preview of a mesh card: the shipped image for curated meshes, else a
     *  lightweight schematic grid drawn from the parametric card's own init
     *  (nx/ny/nz + bounds) — no kernel, no network. */
    protected renderMeshPreview(card: any): React.ReactNode {
        const h = React.createElement;
        const box: React.CSSProperties = { marginTop: 10, border: '1px solid var(--theia-panel-border)', borderRadius: 6, padding: 8, background: 'var(--theia-editorWidget-background)' };
        const label: React.CSSProperties = { fontSize: 11, textTransform: 'uppercase', letterSpacing: '.05em', color: 'var(--theia-descriptionForeground)', marginBottom: 6 };
        // Curated mesh: show the preview image that ships WITH the GUI
        // (gui/previews/…, same-origin) — the images the mesh pipeline
        // (generate_mesh_previews.py) produces, NOT a re-invented schematic.
        if (card.preview) {
            const src = (() => { try { return new URL('gui/' + String(card.preview).replace(/^\/+/, ''), document.baseURI).href; } catch { return 'gui/' + card.preview; } })();
            return h('div', { style: box },
                h('div', { style: label }, 'Mesh preview'),
                h('img', { src, alt: (card.title || 'mesh') + ' preview', style: { maxWidth: '100%', display: 'block', borderRadius: 4 }, onError: (e: any) => { e.currentTarget.style.display = 'none'; } }));
        }
        // A curated mesh (has a .msh) without a preview image: don't fake a grid
        // schematic for it — that only fits the parametric "Create N-D" grids.
        if (card.mesh_file) {
            return h('div', { style: box }, h('div', { style: label }, 'Mesh preview'),
                h('div', { style: { fontSize: 12, color: 'var(--theia-descriptionForeground)' } }, 'No preview image shipped for this mesh yet — run generate_mesh_previews.py.'));
        }
        // Builtin parametric grid → schematic.
        const init = this.mergedInit(card);
        const nx = Math.max(1, Math.round(Number(init.nx ?? init.n_cells ?? 10)) || 10);
        const ny = Math.max(1, Math.round(Number(init.ny ?? 1)) || 1);
        const nz = init.nz != null ? Math.max(1, Math.round(Number(init.nz)) || 1) : 0;
        const W = 280, H = 130, pad = 10;
        const dcx = Math.min(nx, 28), dcy = Math.min(Math.max(ny, 1), 16);
        const stroke = 'var(--theia-descriptionForeground)';
        const lines: React.ReactNode[] = [];
        for (let i = 0; i <= dcx; i++) { const x = pad + i * (W - 2 * pad) / dcx; lines.push(h('line', { key: 'vx' + i, x1: x, y1: pad, x2: x, y2: H - pad, stroke, strokeWidth: 0.6, opacity: 0.45 })); }
        for (let j = 0; j <= dcy; j++) { const y = pad + j * (H - 2 * pad) / dcy; lines.push(h('line', { key: 'hz' + j, x1: pad, y1: y, x2: W - pad, y2: y, stroke, strokeWidth: 0.6, opacity: 0.45 })); }
        const dims = nz ? `${nx}×${ny}×${nz}` : (ny > 1 ? `${nx}×${ny}` : `${nx}`);
        const fmt = (v: any) => (v == null ? '?' : (+v).toString());
        let domain = `x∈[${fmt(init.x_min)}, ${fmt(init.x_max)}]`;
        if (init.y_max != null) { domain += `, y∈[${fmt(init.y_min)}, ${fmt(init.y_max)}]`; }
        if (init.z_max != null) { domain += `, z∈[${fmt(init.z_min)}, ${fmt(init.z_max)}]`; }
        return h('div', { style: box },
            h('div', { style: label }, 'Mesh preview'),
            h('svg', { width: '100%', viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'xMidYMid meet', style: { maxHeight: 150, display: 'block' } },
                h('rect', { x: pad, y: pad, width: W - 2 * pad, height: H - 2 * pad, fill: 'none', stroke: 'var(--theia-focusBorder, var(--theia-button-background))', strokeWidth: 1.2 }),
                lines),
            h('div', { style: { fontSize: 11.5, color: 'var(--theia-descriptionForeground)', marginTop: 5 } },
                dims + ' cells' + (nz ? ' — 3-D (one face shown)' : '') + ' · ' + domain
                + (nx > dcx ? ' · grid subsampled' : '')));
    }

    /** Rendered inside the right-hand "Zoomy Parameters" panel: the active card's
     *  header, description (with math) and its editable parameter form. */
    renderActiveParams(): React.ReactNode {
        const h = React.createElement;
        const wrap = (children: React.ReactNode): React.ReactNode => h('div', { style: { padding: 14, fontFamily: 'var(--theia-font-family)', color: 'var(--theia-foreground)' } }, children);
        const target = this.activeParamTarget();
        if (!target) {
            return wrap(h('div', { style: { color: 'var(--theia-descriptionForeground)', fontSize: 13, lineHeight: 1.7 } },
                'Select a model, mesh, solver or visualization, then click ',
                h('span', { className: 'codicon codicon-settings-gear', style: { verticalAlign: 'middle' } }), ' Parameters on the card to edit its values here.'));
        }
        const { card, dir } = target;
        const label = TABS.find(t => t.dir === dir)?.label || dir;
        return wrap([
            h('div', { key: 'hd', style: { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 } },
                h('span', { style: { fontSize: 11, textTransform: 'uppercase', letterSpacing: '.06em', color: 'var(--theia-descriptionForeground)' } }, label),
                h('div', { style: { fontWeight: 700, fontSize: 15, flex: 1 } }, card.title || card.id),
                h('button', { title: 'Close', onClick: () => this.closeParams(), style: { cursor: 'pointer', border: 'none', background: 'transparent', color: 'var(--theia-descriptionForeground)', padding: 2 } }, h('span', { className: 'codicon codicon-close' }))),
            card.description ? h('div', { key: 'desc', className: 'zoomy-md', style: { color: 'var(--theia-descriptionForeground)', fontSize: 12.5, marginBottom: 4 }, dangerouslySetInnerHTML: { __html: renderMathMd(card.description) } }) : null,
            h('div', { key: 'form' }, this.renderParamForm(card)),
        ]);
    }

    /** Clicking a card selects it in its tab and expands it (accordion: one open).
     *  Selecting a visualization after a sim has run re-renders it (viz changes). */
    protected pick(card: any, dir: string): void {
        const wasExpanded = this.expanded === card.id;
        this.selected[dir] = card.id;
        this.expanded = wasExpanded ? undefined : card.id;
        this.update();
        this.schedulePersist();
        // Derive the governing equations for a newly expanded model card.
        if (dir === 'models' && this.expanded === card.id && card.class) { this.loadModelMath(card).catch(() => { /* shown as error */ }); }
        // Visualization: the expanded card is THE active viewer — open ITS field +
        // time_step params in the right-hand Parameters panel (single active card
        // ⇒ one set of params visible). The checkbox controls export inclusion.
        if (dir === 'visualizations') {
            if (this.expanded === card.id) { this.schemas.set(card.id, this.vizParamSchema()); this.activeParamCardId = card.id; this.activeParamDir = dir; this.paramsPanel?.sync(); this.onParamsChangedEmitter.fire(); }
            else { this.closeParams(); }
        }
    }
    /** Toggle a viewer's membership in the exported (multi-viewer) visualization. */
    protected toggleVizSelect(card: any): void {
        if (this.selectedViz.has(card.id)) { this.selectedViz.delete(card.id); } else { this.selectedViz.add(card.id); }
        this.update(); this.schedulePersist();
    }

    // === Post-processing chain (routes the run's store to a `postprocess`
    // backend for zoomy_prepost.steps — lift3d / VTK→HDF5). Reuses the CLI's
    // runPostprocChain; the artifacts land in the case's outputs/. =============
    protected togglePostprocStep(step: string): void {
        if (this.postprocSteps.has(step)) { this.postprocSteps.delete(step); } else { this.postprocSteps.add(step); }
        this.update();
    }
    /** Which post-processing steps need a `postprocess` backend. `to_h5` does
     *  NOT: its output is just the run's HDF5 store, which we already hold in
     *  the browser (readStoreBytes). `lift3d` / `to_vtk` import zoomy_prepost
     *  (unpublished → not in Pyodide) so they run server-side. */
    protected static readonly POSTPROC_LOCAL = new Set<string>(['to_h5']);
    protected postprocBackendSteps(steps: string[]): string[] {
        return steps.filter(s => !ZoomyModelConfigWidget.POSTPROC_LOCAL.has(s));
    }
    async runPostproc(): Promise<void> {
        if (this.postprocBusy) { return; }
        const steps = [...this.postprocSteps];
        if (!steps.length) { this.setNotice('Tick at least one post-processing step first.'); return; }
        if (!this.simRan) { this.setNotice('Run a simulation first, then post-process its result.'); return; }
        const backendSteps = this.postprocBackendSteps(steps);
        const wantsLocalH5 = steps.length !== backendSteps.length;   // to_h5 ticked
        const connected = !!(this.cli?.isTagConnected && this.cli.isTagConnected('postprocess'));
        if (backendSteps.length && !connected) {
            this.setNotice('“' + backendSteps.join(', ') + '” run zoomy_prepost on a “postprocess” backend — connect one. “VTK → HDF5” needs no backend.');
            return;
        }
        this.postprocBusy = true; this.simPanel?.reveal(); this.update();
        try {
            const outDir = this.caseUri ? this.caseUri.parent.resolve('outputs') : null;
            if (outDir && !(await this.fileService.exists(outDir))) { await this.fileService.createFolder(outDir); }
            const written: string[] = [];
            // Local (no backend): the run's HDF5 store IS the to_h5 artifact.
            if (wantsLocalH5) {
                emitSimOutput({ kind: 'line', level: 'info', text: '▶ Post-processing (VTK → HDF5) locally — writing the run’s HDF5 store…' });
                const bytes: Uint8Array = await this.cli.readStoreBytes();
                if (outDir && bytes && bytes.length) { await this.fileService.createFile(outDir.resolve('simulation.h5'), BinaryBuffer.wrap(bytes), { overwrite: true }); written.push('simulation.h5'); }
            }
            // Backend: lift3d / to_vtk via zoomy_prepost.
            if (backendSteps.length) {
                emitSimOutput({ kind: 'line', level: 'info', text: '▶ Post-processing (' + backendSteps.join(', ') + ') on the "postprocess" backend…' });
                const storeBytes: Uint8Array = await this.cli.readStoreBytes();
                const model = this.pickedCard('models');
                // lift3d execs the model cell (which ends in display(model.describe()),
                // a Jupyter builtin) → prepend a headless no-op so it runs on the
                // backend without a NameError.
                const modelRaw = cardCode(model, this.mergedInit(model));
                const modelPy = modelRaw ? ('try:\n    display\nexcept NameError:\n    def display(*_a, **_k):\n        pass\n' + modelRaw) : null;
                const res = await this.cli.runPostprocChain({ tag: 'postprocess', storeBytes, steps: backendSteps, nz: this.postprocNz, modelPy,
                    onStatus: (s: any) => { const m = s?.message || s?.state || (typeof s === 'string' ? s : null); if (m) { emitSimOutput({ kind: 'line', level: 'stdout', text: String(m) }); } } });
                const artifacts = (res && res.artifacts) || [];
                if (outDir && artifacts.length) { for (const art of artifacts) { await this.fileService.createFile(outDir.resolve(art.name), BinaryBuffer.wrap(art.bytes), { overwrite: true }); written.push(art.name); } }
            }
            if (written.length) { emitCasesChanged(); }
            const names = written.join(', ');
            emitSimOutput({ kind: 'line', level: 'ok', text: '✓ Post-processing complete — ' + written.length + ' artifact(s) in outputs/' + (names ? ': ' + names : '') + '.' });
            this.setNotice('Post-processing complete: ' + (names || 'done') + '.');
        } catch (e: any) {
            emitSimOutput({ kind: 'line', level: 'error', text: '✗ Post-processing failed: ' + (e?.message || e) });
            this.setNotice('Post-processing failed: ' + (e?.message || e));
        } finally { this.postprocBusy = false; this.update(); }
    }
    /** The post-processing strip shown in the Visualization tab: step checkboxes
     *  + Run, gated on a connected `postprocess` backend. */
    protected renderPostproc(): React.ReactNode {
        const h = React.createElement;
        const connected = !!(this.cli?.isTagConnected && this.cli.isTagConnected('postprocess'));
        // [key, label, needsBackend]. to_h5 = the run's HDF5 store, done locally.
        const STEPS: Array<[string, string, boolean]> = [['to_h5', 'VTK → HDF5', false], ['lift3d', 'Lift 2D → 3D', true], ['to_vtk', 'HDF5 → VTK', true]];
        const backendSteps = this.postprocBackendSteps([...this.postprocSteps]);
        const needsBackend = backendSteps.length > 0;
        const ready = this.simRan && !this.postprocBusy && this.postprocSteps.size > 0 && (!needsBackend || connected);
        const chip = (step: string, label: string, back: boolean): React.ReactNode => {
            const on = this.postprocSteps.has(step);
            return h('button', { key: step, className: 'zoomy-btn pill' + (on ? ' active' : ''), title: back ? 'Runs zoomy_prepost on a “postprocess” backend' : 'Runs locally — no backend needed', onClick: () => this.togglePostprocStep(step) },
                h('span', { className: 'codicon codicon-' + (on ? 'check' : 'circle-large-outline'), style: { marginRight: 4, fontSize: 11 } }), label,
                back ? h('span', { className: 'codicon codicon-server', style: { marginLeft: 5, fontSize: 10, opacity: 0.7 } }) : null);
        };
        const inputS: React.CSSProperties = { background: 'var(--theia-input-background)', color: 'var(--theia-input-foreground)', border: '1px solid var(--theia-input-border, var(--theia-panel-border))', borderRadius: 4, padding: '2px 6px', fontSize: 12, width: 56 };
        return h('div', { style: { border: '1px solid var(--theia-panel-border)', borderRadius: 8, padding: '10px 12px', marginBottom: 14, background: 'var(--theia-editorWidget-background)' } },
            h('div', { style: { fontSize: 11, textTransform: 'uppercase', letterSpacing: '.05em', color: 'var(--theia-descriptionForeground)', marginBottom: 8 } }, 'Post-processing'),
            h('div', { style: { display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' } },
                ...STEPS.map(([s, l, b]) => chip(s, l, b)),
                // lift3d's Nz parameter — editable when the step is selected.
                this.postprocSteps.has('lift3d') ? h('span', { style: { display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 12, color: 'var(--theia-descriptionForeground)' } }, 'Nz', h('input', { type: 'number', min: 1, value: this.postprocNz, title: 'Vertical layers for the 2D→3D lift', style: inputS, onChange: (e: any) => { this.postprocNz = Math.max(1, parseInt(e.target.value, 10) || 10); this.update(); this.schedulePersist(); } })) : null,
                h('button', { className: 'zoomy-btn primary', disabled: !ready, style: { marginLeft: 4, opacity: ready ? 1 : 0.55, cursor: ready ? 'pointer' : 'not-allowed' }, onClick: () => this.runPostproc() },
                    h('span', { className: 'codicon codicon-' + (this.postprocBusy ? 'loading codicon-modifier-spin' : 'server-process'), style: { marginRight: 5 } }), this.postprocBusy ? 'Running…' : 'Run post-processing')),
            h('div', { style: { fontSize: 11.5, color: 'var(--theia-descriptionForeground)', marginTop: 8 } },
                '“VTK → HDF5” writes the run’s HDF5 store — no backend. “Lift 2D → 3D” and “HDF5 → VTK” '
                + (needsBackend ? '(' : '')
                + 'run zoomy_prepost on a “postprocess” backend'
                + (needsBackend ? (connected ? ', connected ✓)' : ' — not connected; connect one via the ✕ scan / Connect backend)') : '')
                + '. Artifacts land in the case outputs/; steps stay saved with the case.'));
    }

    /** The selected card for a tab, else the first with runnable code (models/
     *  meshes/solvers) or the first snippet card (visualizations). */
    protected pickedCard(dir: string): any {
        const cards = this.cardsByTab[dir] || [];
        if (dir === 'visualizations') { const sel = cards.find(c => c.id === this.selected[dir] && c.snippet); return sel || cards.find(c => c.snippet); }
        const sel = cards.find(c => c.id === this.selected[dir]);
        if (sel) { return sel; }
        return cards.find(c => !!cardCode(c, this.mergedInit(c)));
    }

    /** Run the selected model → mesh → solver in the shared scope. This COMPUTES
     *  only — it no longer auto-renders a visualization. To visualize, open the
     *  Visualization tab, pick a viewer and click "Render visualization". */
    async runAssembly(): Promise<void> {
        if (this.simBusy) { return; }
        this.simBusy = true; this.simRan = false; this.simError = undefined; this.simStopped = false;
        // Stream the run's console output to the bottom "Simulation" panel.
        emitSimOutput({ kind: 'clear' });
        this.simPanel?.reveal();
        emitSimOutput({ kind: 'line', level: 'info', text: '▶ Running case "' + this.caseName + '"…' });
        try {
            const solver = this.pickedCard('solvers');
            const solverTag = solver?.requires_tag || 'numpy';
            const solverLocal = !!cardCode(solver, this.mergedInit(solver));
            if (!solverLocal) {
                // Remote solver: submit the whole case to its connected backend.
                if (!this.tagMatches(solverTag, this.connectedTags)) {
                    this.simStatus = 'Solver "' + (solver?.title || solverTag) + '" needs a "' + solverTag + '" backend — connect one.';
                    emitSimOutput({ kind: 'line', level: 'error', text: '✗ Needs a "' + solverTag + '" backend (not connected).' }); this.update(); return;
                }
                await this.runOnBackend(solverTag, solver);
            } else {
                await this.runLocalSteps();
            }
            if (this.simStopped) { this.simStatus = 'Stopped.'; emitSimOutput({ kind: 'line', level: 'error', text: '■ Stopped.' }); return; }
            if (this.simError) { return; }
            this.simRan = true;
            await this.refreshStoreMeta(); // populate field selector + time slider from the store
            await this.writeRunOutputs(); // materialize outputs/ inside the case folder
            this.simStatus = 'Simulation complete. Open the Visualization tab, choose a viewer and click Render.';
            emitSimOutput({ kind: 'line', level: 'ok', text: '✓ Simulation complete — open Visualization to render.' });
        } catch (e: any) {
            if (this.simStopped) { this.simStatus = 'Stopped.'; }
            else { this.simStatus = 'Error: ' + (e?.message || String(e)); emitSimOutput({ kind: 'line', level: 'error', text: '✗ ' + (e?.message || String(e)) }); }
        } finally {
            this.simBusy = false; this.update();
        }
    }

    /** Local numpy (Pyodide) run: model → mesh → solver in the shared scope. */
    protected async runLocalSteps(): Promise<void> {
        for (const [dir, label] of [['models', 'model'], ['meshes', 'mesh'], ['solvers', 'solver']] as const) {
            if (this.simStopped) { return; }
            const card = this.pickedCard(dir);
            if (!card) { this.simError = { cells: [], stdout: 'No ' + label + ' selected.', status: 'error', running: false }; emitSimOutput({ kind: 'line', level: 'error', text: 'No ' + label + ' selected.' }); return; }
            const code = cardCode(card, this.mergedInit(card));
            if (!code) { this.simError = { cells: [], stdout: label + ' needs a backend.', status: 'error', running: false }; emitSimOutput({ kind: 'line', level: 'error', text: label + ' "' + (card.title || card.id) + '" needs a backend.' }); return; }
            this.simStatus = 'Running ' + label + ': ' + (card.title || card.id) + '…'; this.update();
            emitSimOutput({ kind: 'line', level: 'info', text: '· ' + label + ': ' + (card.title || card.id) + '…' });
            const res = await this.cli.runCode(code);
            if (res?.output) { emitSimOutput({ kind: 'line', level: 'stdout', text: String(res.output).trimEnd() }); }
            if (res?.status === 'error') { this.simStatus = 'Error in ' + label + ': see below'; this.simError = { cells: [], stdout: res.output || '', status: 'error', running: false }; emitSimOutput({ kind: 'line', level: 'error', text: '✗ Error in ' + label + '.' }); return; }
            if (res?.store_meta) { this.storeMeta = res.store_meta; }
        }
    }

    /** Copy the run's simulation.h5 out of the worker into the case folder's
     *  outputs/ — so the case folder mirrors a CLI run (cases/<name>/outputs/
     *  simulation.h5), for both local and remote runs. Best-effort: a failure
     *  never breaks the run. */
    protected async writeRunOutputs(): Promise<void> {
        if (!this.caseUri) { return; }
        try {
            const bytes: Uint8Array = await this.cli.readStoreBytes();
            if (!bytes || !bytes.length) { return; }
            const outDir = this.caseUri.parent.resolve('outputs');
            if (!(await this.fileService.exists(outDir))) { await this.fileService.createFolder(outDir); }
            await this.fileService.createFile(outDir.resolve('simulation.h5'), BinaryBuffer.wrap(bytes), { overwrite: true });
            emitCasesChanged();
        } catch (e) { /* non-fatal — e.g. no local store bytes for a pure remote run */ }
    }

    /** Remote run: submit the composed case.py to a connected backend, then pull
     *  the returned store into Pyodide's VFS so the viz cards can read it. */
    protected async runOnBackend(tag: string, solver: any): Promise<void> {
        this.simStatus = 'Submitting case to "' + tag + '" backend…'; this.update();
        emitSimOutput({ kind: 'line', level: 'info', text: '· submitting to "' + tag + '" (' + (solver?.title || solver?.id) + ')…' });
        const spec = await this.gatherSpec();
        const casePy = this.cli.exportCase(spec, 'py');
        const r = await this.cli.submitCase({ tag, casePy, onStatus: (s: any) => { const m = s?.message || s?.state || (typeof s === 'string' ? s : null); if (m) { emitSimOutput({ kind: 'line', level: 'stdout', text: String(m) }); } } });
        const res = r?.result || r || {};
        if (res.output || res.log) { emitSimOutput({ kind: 'line', level: 'stdout', text: String(res.output || res.log).trimEnd() }); }
        if (res.status === 'error' || res.error) { this.simError = { cells: [], stdout: res.error || res.output || 'Backend error', status: 'error', running: false }; emitSimOutput({ kind: 'line', level: 'error', text: '✗ Backend error.' }); return; }
        // submitCase already wrote the HDF5 to /tmp/zoomy_sim/<job_id>.h5 — open it.
        if (res.job_id) {
            try { const meta = await this.cli.openHdf5('/tmp/zoomy_sim/' + res.job_id + '.h5'); if (meta?.store_meta) { this.storeMeta = meta.store_meta; } else if (meta) { this.storeMeta = meta; } }
            catch (e: any) { emitSimOutput({ kind: 'line', level: 'error', text: 'Could not open the returned store: ' + (e?.message || e) }); }
        }
    }

    /** Stop a running simulation (cooperative SIGINT if available, else the
     *  worker is terminated + recreated — the kernel then reboots). */
    async stopAssembly(): Promise<void> {
        if (!this.simBusy) { return; }
        this.simStopped = true;
        this.simStatus = 'Stopping…'; emitSimOutput({ kind: 'line', level: 'error', text: '■ Stopping…' }); this.update();
        try {
            const mode = this.cli.interrupt ? this.cli.interrupt() : undefined;
            // A terminate+recreate reboots the kernel — reflect that so Run waits.
            if (!mode || mode.mode === 'terminate+recreate') { this.kernelReady = false; this.kernelStatus = 'restarting…'; }
        } catch (e) { /* ignore */ }
        this.update();
    }

    /** Render the composed visualization: run EACH selected viewer against the
     *  computed store and stack their plots. Called explicitly from the
     *  Visualization tab's viewer (not from Run), reusing the last run's store. */
    /** Render ONE visualization card: run its snippet with the card's edited
     *  field + time_step and stash the plot on the card (shown below it). The
     *  active/expanded card owns its params + output, so they can't drift out of
     *  sync. Re-reads the store metadata so the inline field/time params reflect
     *  what was just rendered. */
    async renderVizCard(card: any): Promise<void> {
        if (this.vizBusy) { return; }
        if (!card?.snippet) { return; }
        if (!this.simRan) { this.setNotice('Run a simulation first (the Simulation bar), then Render.'); return; }
        this.vizBusy = true;
        const out: CardOut = { cells: [], stdout: '', status: 'running', running: true };
        this.outputs.set(card.id, out); this.update();
        setDisplaySink(cell => { out.cells.push(cell); this.update(); });
        try {
            const snippet = await this.cli.fetchSnippet(card.snippet);
            // Pass the card's edited field + time_step (its inline Parameters).
            const ed = this.edited.get(card.id) || {};
            const ts = Number.isFinite(ed.time_step) ? ed.time_step : 0;
            const fld = ed.field != null && ed.field !== '' ? JSON.stringify(String(ed.field)) : 'None';
            const code = 'time_step = ' + ts + '\nfield_name = ' + fld + '\n' + snippet;
            const res = await this.cli.runCode(code);
            out.stdout = res?.output || ''; out.status = res?.status || 'success';
        } catch (e: any) {
            out.status = 'error'; out.stdout = e?.message || String(e);
        } finally {
            setDisplaySink(undefined); out.running = false; this.vizBusy = false;
            this.update();
            // Refresh field/time params straight from the store (robust for local
            // AND remote runs), so the selector + slider reflect the actual run.
            await this.refreshStoreMeta();
        }
    }

    // Notices go to the bottom Log panel, not the config header (which is now
    // just the kernel chip). `notice` is still set for the initial gate screen.
    protected setNotice(msg: string): void { if (msg) { emitSimOutput({ kind: 'line', level: 'info', text: msg }); } this.notice = msg; }

    // --- #3/#5 Case interchange via zoomy_prepost.case (through zoomy_cli). ---
    /** Build the canonical case spec from the current selection + edits. */
    protected async gatherSpec(): Promise<any> {
        const model = this.pickedCard('models'), mesh = this.pickedCard('meshes'), solver = this.pickedCard('solvers'), viz = this.pickedCard('visualizations');
        const spec: any = {
            meta: { title: (model?.title || 'Zoomy case'), description: 'Exported from the Zoomy model-config GUI.' },
            model: { code: this.cardCodeFor(model), class_path: model?.class || null, init: this.mergedInit(model), card: model?.id || null },
            mesh: { code: this.cardCodeFor(mesh), spec: this.mergedInit(mesh) },
            settings: {},
            solver: { tag: solver?.requires_tag || 'numpy', id: solver?.id || null, params: solver?.params ? this.mergedInit(solver) : {} },
        };
        // Run cell = the solver card's stored/template code (a coupling child's coupled
        // run note is stored here as a code override and round-trips verbatim).
        const solverCode = this.cardCodeFor(solver);
        if (solverCode) { spec.run = { code: solverCode }; }
        // Compose EVERY checked viewer into the ## Visualization section (multi-
        // select → several plots of one run in the exported .py / .ipynb).
        const vizCards = (this.cardsByTab['visualizations'] || []).filter(c => this.selectedViz.has(c.id) && c.snippet);
        const chosen = vizCards.length ? vizCards : (viz?.snippet ? [viz] : []);
        if (chosen.length) {
            try {
                const parts: string[] = [this.cli.vizPrelude()];
                for (const vc of chosen) { parts.push('# --- ' + (vc.title || vc.id) + ' ---'); parts.push(await this.cli.fetchSnippet(vc.snippet)); }
                spec.visualization = { code: parts.join('\n') };
            } catch { /* skip viz */ }
        }
        if (this.postprocSteps.size) { spec.postproc = [...this.postprocSteps]; spec.postproc_nz = this.postprocNz; }
        return spec;
    }
    async exportCase(fmt: 'py' | 'ipynb'): Promise<void> {
        try {
            const spec = await this.gatherSpec();
            const text = this.cli.exportCase(spec, fmt);
            download('zoomy_case.' + (fmt === 'ipynb' ? 'ipynb' : 'py'), text, fmt === 'ipynb' ? 'application/json' : 'text/x-python');
            this.setNotice('Exported case as .' + fmt);
        } catch (e: any) { this.setNotice('Export failed: ' + (e?.message || e)); }
    }
    /** Import a case (.py/.ipynb): parse it and re-select the matching cards. */
    importCase(): void {
        const input = document.createElement('input'); input.type = 'file'; input.accept = '.py,.ipynb';
        input.onchange = async () => {
            const file = input.files?.[0]; if (!file) { return; }
            try {
                let text = await file.text();
                if (file.name.endsWith('.ipynb')) { const nb = JSON.parse(text); text = (nb.cells || []).map((c: any) => (Array.isArray(c.source) ? c.source.join('') : c.source)).join('\n\n'); }
                const spec = this.cli.parseCase(text);
                this.applySpec(spec);
                this.setNotice('Imported case: ' + file.name);
            } catch (e: any) { this.setNotice('Import failed: ' + (e?.message || e)); }
        };
        input.click();
    }
    /** Load a case from raw .py/.ipynb text (Explorer "Open in model configurator")
     *  as a new case in the project — keeping the case-as-source-of-truth model. */
    async openCaseText(text: string, isIpynb: boolean, name?: string): Promise<void> {
        try {
            if (isIpynb) { const nb = JSON.parse(text); text = (nb.cells || []).map((c: any) => (Array.isArray(c.source) ? c.source.join('') : c.source)).join('\n\n'); }
            const spec = this.cli.parseCase(text);
            await this.newCase((name || 'case').replace(/\.(py|ipynb)$/, ''), spec);
        } catch (e: any) { this.setNotice('Open case failed: ' + (e?.message || e)); }
    }
    /** Re-select the cards a spec refers to (by class_path / mesh spec / tag). */
    protected applySpec(spec: any): void {
        // Fresh code<->card map for this case: the parsed cell code below repopulates
        // it (a previous case's overrides must not leak into this one).
        this.codeByCard.clear();
        const byClass = (dir: string, cls: string) => (this.cardsByTab[dir] || []).find(c => c.class === cls);
        if (spec?.model) {
            // Match by class_path (the usual case); fall back to an explicit card id
            // for cards whose Python class is null (e.g. the VOF/OpenFOAM participant,
            // which has no zoomy_core model class but must still select vof-openfoam).
            let mc = spec.model.class_path ? byClass('models', spec.model.class_path) : null;
            if (!mc && spec.model.card) { mc = (this.cardsByTab['models'] || []).find(c => c.id === spec.model.card); }
            if (mc) { this.selected['models'] = mc.id; if (spec.model.init) { this.edited.set(mc.id, { ...spec.model.init }); } }
        }
        if (spec?.mesh?.spec) { const meshes = this.cardsByTab['meshes'] || []; const c = meshes[0]; if (c) { this.selected['meshes'] = c.id; this.edited.set(c.id, { ...spec.mesh.spec }); } }
        if (spec?.solver) {
            const solvers = this.cardsByTab['solvers'] || [];
            // Prefer the exact card id (several solvers can share a backend tag,
            // e.g. coupled zoomyFoam + incompressibleVOF both on "OpenFOAM"); fall
            // back to the tag for older cases that only stored the backend.
            const c = (spec.solver.id && solvers.find(s => s.id === spec.solver.id))
                || (spec.solver.tag && solvers.find(s => this.canonTag(s.requires_tag || 'numpy') === this.canonTag(spec.solver.tag)));
            if (c) { this.selected['solvers'] = c.id; }
        }
        // Seed the selected visualization viewer (single-select).
        const firstViz = (this.cardsByTab['visualizations'] || []).find(c => c.snippet);
        if (firstViz) { this.selected['visualizations'] = firstViz.id; this.selectedViz.clear(); this.selectedViz.add(firstViz.id); }
        // Restore enabled post-processing steps + Nz (round-trips via spec.postproc).
        this.postprocSteps.clear();
        if (Array.isArray(spec?.postproc)) { for (const s of spec.postproc) { this.postprocSteps.add(String(s)); } }
        if (spec?.postproc_nz) { this.postprocNz = Math.max(1, parseInt(String(spec.postproc_nz), 10) || 10); }
        this.absorbCode(spec);   // capture each selected card's ACTUAL code (free-form overrides)
        this.jumpSubsToSelectedCategory();   // show each selected card's subtab
        this.update();
    }

    /** On a case switch, jump every tab's 2nd-level subtab to the category of the
     *  case's SELECTED card, so the selection is always visible (e.g. a coupled
     *  solver -> the "Coupling" subtab with zoomyFoam checked, not "Built-in
     *  (NumPy)"). Only here (applySpec, per case-open) — a manual subtab click
     *  within a case is not disturbed. No category on the card -> leave the tab. */
    protected jumpSubsToSelectedCategory(): void {
        for (const dir of ['models', 'meshes', 'solvers', 'visualizations']) {
            const sel = this.selected[dir];
            if (!sel) { continue; }
            const card = (this.cardsByTab[dir] || []).find((c: any) => c.id === sel);
            if (card && card.category) { this.activeSub[dir] = card.category; }
        }
    }

    // --- #6 Project persistence: a project is a ZIP of project.json + every
    // case FOLDER (case.py + its outputs/), matching the CLI/old GUI — not just
    // a JSON. Save downloads the .zip; Load opens one and materializes it. ---
    async saveProject(): Promise<void> {
        try {
            await ensureJSZip();
            const JSZip = (window as any).JSZip;
            if (!JSZip) { throw new Error('JSZip unavailable (offline?)'); }
            const zip = new JSZip();
            const manifest = { version: 1, selected: this.selected, edited: Array.from(this.edited.entries()), active: this.active, cases: this.cases };
            zip.file('project.json', JSON.stringify(manifest, null, 2));
            const casesFolder = zip.folder('cases');
            // Iterate the TOP-LEVEL folders under cases/: flat cases + coupling
            // PARENTS. addFolderToZip recurses, so a coupling parent brings its
            // coupling.yml + precice-config.xml AND its child case folders — the
            // whole nested structure round-trips. (this.cases holds child
            // relative-paths, so iterating it alone would drop the parent files.)
            const topLevel = [...this.cases.filter(n => !n.includes('/')), ...this.couplings.map(c => c.name)];
            for (const name of topLevel) {
                await this.addFolderToZip(casesFolder.folder(name), new URI(PROJECT_ROOT + '/' + name));
            }
            const blob = await zip.generateAsync({ type: 'blob' });
            downloadBlob('zoomy_project.zip', blob);
            await this.cli.storage.writeJson('projects/current.json', manifest);  // quick in-browser restore
            this.setNotice('Saved zoomy_project.zip — ' + this.cases.length + ' case' + (this.cases.length === 1 ? '' : 's') + ' (incl. outputs).');
        } catch (e: any) { this.setNotice('Save failed: ' + (e?.message || e)); }
    }
    /** Recursively add a case folder's files (case.py + outputs/**) to a JSZip folder. */
    protected async addFolderToZip(zf: any, dir: URI): Promise<void> {
        if (!(await this.fileService.exists(dir))) { return; }
        const stat = await this.fileService.resolve(dir);
        for (const child of stat.children || []) {
            if (child.isDirectory) { await this.addFolderToZip(zf.folder(child.resource.path.base), child.resource); }
            else { const c = await this.fileService.readFile(child.resource); zf.file(child.resource.path.base, c.value.buffer); }
        }
    }
    async loadProject(): Promise<void> {
        const input = document.createElement('input'); input.type = 'file'; input.accept = '.zip,application/zip';
        input.onchange = async () => {
            const file = input.files && input.files[0]; if (!file) { return; }
            this.setNotice('Loading ' + file.name + '…'); this.update();
            try { await this.loadProjectFromZip(await file.arrayBuffer()); }
            catch (e: any) { this.setNotice('Load failed: ' + (e?.message || e)); this.update(); }
        };
        input.click();
    }
    /** Materialize a project ZIP: write every cases/<name>/** file into the FS and
     *  restore the manifest (selection/edits). Also accepts the legacy shape
     *  (<name>/case.py or <name>.py). Shared by file-load and URL-load. */
    async loadProjectFromZip(buf: ArrayBuffer): Promise<void> {
        await ensureJSZip();
        const JSZip = (window as any).JSZip;
        if (!JSZip) { throw new Error('JSZip unavailable (offline?)'); }
        const zip = await JSZip.loadAsync(buf);
        const files: any[] = Object.values(zip.files).filter((f: any) => !f.dir);
        let manifest: any = null;
        const mf = zip.file('project.json'); if (mf) { try { manifest = JSON.parse(await mf.async('string')); } catch { /* ignore */ } }
        const hasCasesPrefix = files.some((f: any) => /(?:^|\/)cases\/[^/]+\//.test(f.name));
        const names = new Set<string>(); let count = 0; let first: string | undefined;
        for (const f of files) {
            if (/(?:^|\/)project\.json$/.test(f.name)) { continue; }
            let name: string | undefined; let rel: string | undefined;
            const mc = f.name.match(/(?:^|\/)cases\/([^/]+)\/(.+)$/);
            if (hasCasesPrefix) { if (mc) { name = mc[1]; rel = mc[2]; } }
            else {  // legacy: <name>/case.py  or  <name>.py
                const ml = f.name.match(/(?:^|\/)([^/]+)\/case\.py$/) || f.name.match(/(?:^|\/)([^/]+)\.py$/);
                if (ml) { name = ml[1]; rel = 'case.py'; }
            }
            if (!name || !rel) { continue; }
            name = name.replace(/[^a-zA-Z0-9_-]+/g, '_');
            const uri = new URI(PROJECT_ROOT + '/' + name + '/' + rel);
            if (!(await this.fileService.exists(uri.parent))) { await this.fileService.createFolder(uri.parent); }
            await this.fileService.createFile(uri, BinaryBuffer.wrap(await f.async('uint8array')), { overwrite: true });
            names.add(name); count++; if (rel === 'case.py' && !first) { first = name; }
        }
        await this.listCases();
        if (manifest) {
            Object.assign(this.selected, manifest.selected || {});
            this.edited.clear(); for (const [k, v] of (manifest.edited || [])) { this.edited.set(k, v); }
            if (manifest.active) { this.active = manifest.active; }
        }
        // Open a real leaf case (flat or coupled child) — never a coupling
        // parent (it has no case.py). listCases() above populated this.cases.
        const open = first || this.cases[0];
        if (open) { await this.openCaseByName(open); }
        this.setNotice('Loaded project — ' + names.size + ' case(s), ' + count + ' file(s).');
    }

    // --- #4 Connect a remote backend by URL. ---
    /** Report a connected adapter's handshake result: it declares its own
     *  identity (tag) + capabilities (backends) at /health — we never assume. */
    protected announceConnected(adapter: any): void {
        const caps: string[] = (adapter?.backends && adapter.backends.length) ? adapter.backends : (adapter?.tag ? [adapter.tag] : []);
        this.setNotice('Connected "' + (adapter?.tag || '?') + '" backend' + (caps.length ? ' — solvers: ' + caps.join(', ') : '') + '.');
    }
    async connectBackend(): Promise<void> {
        const url = this.backendUrl.trim(); if (!url) { return; }
        this.setNotice('Connecting to ' + url + '…');
        try {
            // connect() does the /health handshake: the server declares its tag +
            // capabilities; the adapter registers under the tag IT reports.
            const adapter = await this.cli.connect(url);
            if (!adapter) { this.setNotice('No healthy zoomy-server at ' + url + '. Is one running there?'); return; }
            this.refreshBackends();
            this.announceConnected(adapter);
            try { this.cli.onConnectionsChange && this.cli.onConnectionsChange(() => this.refreshBackends()); } catch { /* ignore */ }
        } catch (e: any) { this.setNotice('Connect failed: ' + (e?.message || e) + ' — is a zoomy-server running there?'); }
    }
    /** Auto-discover local backends: probe localhost:8080–8100 concurrently and
     *  connect any that answer the /health handshake (skipping the ones already
     *  connected). One-click convenience — no need to type URLs. */
    async scanBackends(): Promise<void> {
        this.setNotice('Scanning localhost:8080–8100 for backends…');
        const ports: number[] = []; for (let p = 8080; p <= 8100; p++) { ports.push(p); }
        const before = new Set(this.connectedTags);
        const results = await Promise.all(ports.map(async (p) => {
            try { return await this.cli.connect('http://localhost:' + p); } catch { return null; }
        }));
        this.refreshBackends();
        try { this.cli.onConnectionsChange && this.cli.onConnectionsChange(() => this.refreshBackends()); } catch { /* ignore */ }
        const fresh = results.filter((a: any) => a && a.tag && !before.has(a.tag));
        this.setNotice(fresh.length
            ? 'Connected ' + fresh.length + ' backend' + (fresh.length === 1 ? '' : 's') + ': ' + fresh.map((a: any) => a.tag).join(', ') + '.'
            : 'No new backends found on localhost:8080–8100. For a backend on another port, use "Connect backend…" with its full URL.');
    }
    /** Disconnect a connected backend by its tag — actually drops the CLI's
     *  HttpAdapter + heartbeat (cli.disconnect), not just the GUI list. The
     *  in-browser numpy (pyodide) runtime is always-on and cannot be removed. */
    async disconnectBackend(tag: string): Promise<void> {
        if (!tag || tag.indexOf('numpy') === 0) { return; }
        try { this.cli.disconnect && this.cli.disconnect(tag); this.setNotice('Disconnected backend: ' + tag); }
        catch (e: any) { this.setNotice('Disconnect failed: ' + (e?.message || e)); }
        this.refreshBackends();
    }
    /** Re-read connected tags and notify the status bar + Zoomy view. */
    protected refreshBackends(): void {
        this.connectedTags = this.cli?.availableTags ? this.cli.availableTags() : [];
        this.onBackendsChanged?.(this.connectedTags);
        emitBackendsChanged();
        this.update();
    }

        /** A card's backend "flag": solver/viz cards advertise built-in (runs in
     *  the browser, always green) vs a required backend tag (green when that
     *  backend is connected) vs post-processing. Cards can override via
     *  `card.flag` ('built-in' | 'post-processing') and `card.requires_tag`. */
    // TEMPORARY foam≡OpenFOAM alias — remove once the zoomy_openfoam container is
    // rebuilt to report OpenFOAM. Canonicalizes foam -> OpenFOAM so an OpenFOAM-
    // tagged card is satisfied by a connected foam backend (and vice versa). Route
    // ALL backend tag-matching through tagMatches — don't scatter string compares.
    protected canonTag(t: string): string { return t === 'foam' ? 'OpenFOAM' : t; }
    protected tagMatches(requiredTag: string, connectedTags: string[]): boolean {
        const want = this.canonTag(requiredTag);
        return (connectedTags || []).some(t => this.canonTag(t) === want);
    }
    protected cardFlag(card: any, dir: string, runnable: boolean): { label: string; available: boolean; tip: string } | null {
        if (dir !== 'solvers' && dir !== 'visualizations') { return card.requires_tag ? { label: card.requires_tag, available: this.tagMatches(card.requires_tag, this.connectedTags), tip: 'Needs a "' + card.requires_tag + '" backend' } : null; }
        // Built-in: has a local template (solver) or a snippet with no remote tag (viz).
        const builtin = card.flag === 'built-in' || (card.requires_tag ? runnable : (dir === 'visualizations' ? !!card.snippet : runnable));
        if (builtin) { return { label: 'built-in', available: true, tip: 'Built in — runs in the browser (Pyodide), no backend needed.' }; }
        if (card.requires_tag) { const ok = this.tagMatches(card.requires_tag, this.connectedTags); return { label: card.requires_tag, available: ok, tip: ok ? 'Backend "' + card.requires_tag + '" connected — can run.' : 'Needs a "' + card.requires_tag + '" backend (not connected).' }; }
        return { label: 'post-processing', available: false, tip: 'Needs a post-processing backend (e.g. 3-D interpolation on a general mesh).' };
    }

        protected renderCard(card: any, dir: string): React.ReactNode {
        const h = React.createElement;
        const runnable = !!cardCode(card, this.mergedInit(card));
        const out = this.outputs.get(card.id);
        const isViz = dir === 'visualizations';
        // Viz params show in the RIGHT panel (like every other card). Multi-select
        // (checkbox) = export inclusion; the active/expanded card = params shown.
        const hasParams = isViz || !!(card.params || card.class || (card.init && Object.keys(card.init).length));
        const isSel = this.selected[dir] === card.id;   // the active card (params + outline)
        const isChecked = isViz && this.selectedViz.has(card.id);
        const isExp = this.expanded === card.id;
        const cardStyle: React.CSSProperties = {
            border: '1px solid ' + (isSel ? 'var(--theia-focusBorder, var(--theia-button-background))' : 'var(--theia-editorWidget-border, var(--theia-panel-border))'),
            borderLeft: (isSel ? '3px solid var(--theia-button-background)' : '1px solid var(--theia-editorWidget-border, var(--theia-panel-border))'),
            borderRadius: 8, padding: 14, marginBottom: 12, background: 'var(--theia-editorWidget-background)',
        };
        const btn: React.CSSProperties = { cursor: runnable ? 'pointer' : 'not-allowed', border: 'none', borderRadius: 6, padding: '6px 14px', fontSize: 13, fontWeight: 600, background: runnable ? 'var(--theia-button-background)' : 'var(--theia-button-secondaryBackground)', color: 'var(--theia-button-foreground)', opacity: runnable ? 1 : 0.6 };
        // Plain, consistent secondary-button look — no toggled/grayed active state.
        const gearBtn: React.CSSProperties = { cursor: 'pointer', border: '1px solid var(--theia-panel-border)', borderRadius: 6, padding: '5px 10px', fontSize: 12.5, background: 'transparent', color: 'var(--theia-foreground)' };
        const mm = dir === 'models' ? this.modelMath.get(card.id) : undefined;
        const displaying = mm?.status === 'loading';
        const displayBtn: React.CSSProperties = { cursor: this.kernelReady && !displaying ? 'pointer' : 'not-allowed', border: 'none', borderRadius: 6, padding: '6px 14px', fontSize: 13, fontWeight: 600, background: 'var(--theia-button-background)', color: 'var(--theia-button-foreground)', opacity: this.kernelReady && !displaying ? 1 : 0.6 };
        const selectIcon = isViz
            ? h('span', { title: 'Include this viewer in the exported case (multi-select)', className: 'codicon codicon-' + (isChecked ? 'check-all' : 'circle-large-outline'), style: { color: isChecked ? 'var(--theia-button-background)' : 'var(--theia-descriptionForeground)', cursor: 'pointer' }, onClick: (e: any) => { e.stopPropagation(); this.toggleVizSelect(card); } })
            : h('span', { className: 'codicon codicon-' + (isSel ? 'pass-filled' : 'circle-large-outline'), style: { color: isSel ? 'var(--theia-button-background)' : 'var(--theia-descriptionForeground)' } });
        // Backend-dependency pill for solver + visualization cards (numpy / jax /
        // OpenFOAM / … from requires_tag, or "built-in"). Subtle gray box on the
        // right, just left of the chevron — shows on any subtab. Tooltip says
        // whether that backend is connected.
        const flag = (dir === 'solvers' || isViz) ? this.cardFlag(card, dir, runnable) : null;
        // Show the backend tag itself (numpy / jax / OpenFOAM / … from requires_tag)
        // when the card declares one, else cardFlag's label (built-in / post-processing).
        const flagLabel = flag ? (card.requires_tag || flag.label) : null;
        const flagPill = flag ? h('span', { title: flag.tip, style: {
            fontSize: 11, padding: '1px 7px', borderRadius: 10, whiteSpace: 'nowrap',
            background: 'var(--theia-badge-background, rgba(128,128,128,0.22))',
            color: 'var(--theia-descriptionForeground)',
            border: '1px solid var(--theia-panel-border)', opacity: flag.available ? 1 : 0.75 } }, flagLabel) : null;
        const header = h('div', { style: { display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer' }, onClick: () => this.pick(card, dir) },
            selectIcon,
            h('div', { style: { fontWeight: 600, fontSize: 14, flex: 1 } }, card.title || card.id),
            this.editMode ? h('button', { title: 'Remove this card', onClick: (e: any) => { e.stopPropagation(); this.removeCard(card); }, style: { cursor: 'pointer', border: 'none', background: 'transparent', color: 'var(--theia-errorForeground)', padding: 2 } }, h('span', { className: 'codicon codicon-trash' })) : null,
            flagPill,
            h('span', { className: 'codicon codicon-chevron-' + (isExp ? 'down' : 'right'), style: { color: 'var(--theia-descriptionForeground)' } }));
        // Collapsed: header only. Expanded: full detail (description + params + run + output).
        const vizReady = this.simRan && !this.vizBusy;
        const vizBtn: React.CSSProperties = { ...btn, cursor: vizReady && !(out && out.running) ? 'pointer' : 'not-allowed', background: 'var(--theia-button-background)', opacity: vizReady && !(out && out.running) ? 1 : 0.6 };
        const body = !isExp ? null : h('div', { style: { marginTop: 8 } },
            card.description ? h('div', { className: 'zoomy-md', style: { color: 'var(--theia-descriptionForeground)', fontSize: 12.5 }, dangerouslySetInnerHTML: { __html: renderMathMd(card.description) } }) : null,
            dir === 'meshes' ? this.renderMeshPreview(card) : null,
            h('div', { style: { display: 'flex', gap: 8, marginTop: 10 } },
                hasParams ? h('button', { style: gearBtn, onClick: () => this.openParams(card, dir) }, h('span', { className: 'codicon codicon-settings-gear', style: { verticalAlign: 'middle', marginRight: 4 } }), 'Parameters') : null,
                h('button', { style: gearBtn, title: 'Open this section of case.py in the editor', onClick: () => this.editCardFile(dir) }, h('span', { className: 'codicon codicon-edit', style: { verticalAlign: 'middle', marginRight: 4 } }), 'Edit'),
                dir === 'models'
                    ? h('button', { style: displayBtn, disabled: !this.kernelReady || displaying, title: 'Build the model and display its equations below', onClick: () => this.loadModelMath(card, true) }, h('span', { className: 'codicon codicon-' + (displaying ? 'loading codicon-modifier-spin' : 'symbol-structure'), style: { verticalAlign: 'middle', marginRight: 6 } }), displaying ? 'Displaying…' : 'Display model')
                    : isViz
                        ? h('button', { style: vizBtn, disabled: !vizReady || (out && out.running), title: this.simRan ? 'Render this visualization from the last run' : 'Run a simulation first', onClick: () => this.renderVizCard(card) }, h('span', { className: 'codicon codicon-' + ((out && out.running) ? 'loading codicon-modifier-spin' : 'graph'), style: { verticalAlign: 'middle', marginRight: 6 } }), (out && out.running) ? 'Rendering…' : 'Render visualization')
                        : h('button', { style: btn, disabled: !runnable || (out && out.running), onClick: () => runnable && this.runCard(card) }, out && out.running ? 'Running…' : 'Run')),
            // Output below the buttons: models → derived equations; mesh/solver →
            // run output; visualization → the rendered plot (shown under its card).
            dir === 'models' ? this.renderModelMath(card)
                : (out ? h('div', { style: { marginTop: 10, borderTop: '1px solid var(--theia-panel-border)', paddingTop: 8, color: out.status === 'error' ? 'var(--theia-errorForeground)' : undefined } },
                    out.running && !out.cells.length ? h('div', { style: { fontSize: 12, color: 'var(--theia-descriptionForeground)' } }, 'Rendering…') : null,
                    out.cells.map((c, i) => this.renderCell(c, 'c' + i)),
                    out.stdout ? h('pre', { style: { margin: '2px 0', whiteSpace: 'pre-wrap', fontSize: 12, fontFamily: 'var(--theia-code-font-family, monospace)' } }, out.stdout) : null) : null));
        return h('div', { key: card.id, style: cardStyle }, header, body);
    }

    /** Shown when no case is open: the GUI is gated until a case exists. */
    protected renderGate(): React.ReactNode {
        const h = React.createElement;
        const page: React.CSSProperties = { maxWidth: 640, margin: '0 auto', padding: '48px 24px', color: 'var(--theia-foreground)', fontFamily: 'var(--theia-font-family)' };
        const card: React.CSSProperties = { border: '1px solid var(--theia-panel-border)', borderRadius: 8, padding: 18, marginTop: 18, background: 'var(--theia-editorWidget-background)' };
        const inputS: React.CSSProperties = { background: 'var(--theia-input-background)', color: 'var(--theia-input-foreground)', border: '1px solid var(--theia-input-border, var(--theia-panel-border))', borderRadius: 4, padding: '7px 10px', fontSize: 13, flex: 1 };
        const primary: React.CSSProperties = { cursor: 'pointer', border: 'none', borderRadius: 6, padding: '7px 16px', fontSize: 13, fontWeight: 600, background: 'var(--theia-button-background)', color: 'var(--theia-button-foreground)' };
        const listBtn: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', border: '1px solid var(--theia-panel-border)', borderRadius: 6, padding: '8px 12px', fontSize: 13, background: 'transparent', color: 'var(--theia-foreground)', width: '100%', textAlign: 'left', marginTop: 6 };
        const label: React.CSSProperties = { fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '.05em', color: 'var(--theia-descriptionForeground)', marginBottom: 8 };
        return h('div', { style: page },
            h('h1', { style: { fontSize: 26, margin: '0 0 6px', fontWeight: 700 } }, 'Model configuration'),
            h('div', { style: { color: 'var(--theia-descriptionForeground)', fontSize: 13, lineHeight: 1.6 } },
                'A Zoomy project is a set of ', h('strong', null, 'cases'), ', and each case is a ', h('strong', null, 'folder'),
                ' that is the single source of truth. The configurator only edits an open case — every change is written straight back to the case, so it can never drift out of sync with the CLI. Create or open a case to begin.'),
            h('div', { style: card },
                h('div', { style: label }, 'New case'),
                h('div', { style: { display: 'flex', gap: 8 } },
                    h('input', { style: inputS, value: this.newCaseName, placeholder: 'case name (e.g. dam_break_1d)', onChange: (e: any) => { this.newCaseName = e.target.value; this.update(); }, onKeyDown: (e: any) => { if (e.key === 'Enter' && this.newCaseName.trim()) { this.newCase(this.newCaseName); } } }),
                    h('button', { style: primary, disabled: !this.newCaseName.trim(), onClick: () => this.newCase(this.newCaseName) }, 'Create case'))),
            this.cases.length ? h('div', { style: card },
                h('div', { style: label }, 'Open a case'),
                this.cases.map(name => h('button', { key: name, style: listBtn, onClick: () => this.openCaseByName(name) }, h('span', { className: 'codicon codicon-folder' }), name))) : null,
            this.notice ? h('div', { style: { fontSize: 12, marginTop: 16, color: 'var(--theia-notificationsInfoIcon-foreground, var(--theia-foreground))' } }, this.notice) : null);
    }

    protected render(): React.ReactNode {
        const h = React.createElement;
        const page: React.CSSProperties = { maxWidth: 900, margin: '0 auto', padding: '32px 24px', color: 'var(--theia-foreground)', fontFamily: 'var(--theia-font-family)' };
        if (this.error) { return h('div', { style: page }, h('h2', null, 'Model configuration'), h('pre', { style: { color: 'var(--theia-errorForeground)' } }, this.error)); }
        if (!this.loaded) { return h('div', { style: page }, h('h2', null, 'Model configuration'), h('div', { style: { color: 'var(--theia-descriptionForeground)' } }, 'Loading the card catalog + booting the in-browser kernel…')); }
        // Case-as-source-of-truth: no open case → the GUI is not usable; you must
        // create or open a case first (so the GUI can never be out of sync with a folder).
        if (!this.caseUri) { return this.renderGate(); }
        const tabBtn = (t: TabDef): React.ReactNode => h('button', {
            key: t.dir, onClick: () => { this.active = t.dir; this.closeParams(); this.update(); },
            style: { cursor: 'pointer', border: 'none', borderBottom: this.active === t.dir ? '2px solid var(--theia-button-background)' : '2px solid transparent', background: 'transparent', color: this.active === t.dir ? 'var(--theia-foreground)' : 'var(--theia-descriptionForeground)', padding: '8px 14px', fontSize: 13, fontWeight: 600 },
        }, t.label + ' (' + (this.cardsByTab[t.dir]?.length || 0) + ')');
        const allCards = this.cardsByTab[this.active] || [];
        const cats = [...new Set(allCards.map(c => c.category || 'General'))];
        const activeCat = cats.includes(this.activeSub[this.active]) ? this.activeSub[this.active] : cats[0];
        const cards = cats.length > 1 ? allCards.filter(c => (c.category || 'General') === activeCat) : allCards;
        // Selected subtab: change ONLY the (blue) outline, keep the inner
        // background transparent so the label stays readable (no graying).
        // Unified button look via the shared .zoomy-btn class — active state is
        // outline-only (no graying). Selected subtab / Edit-cards toggle add .active.
        const subBtn = (cat: string): React.ReactNode => h('button', {
            key: cat, className: 'zoomy-btn pill' + (cat === activeCat ? ' active' : ''),
            onClick: () => { this.activeSub[this.active] = cat; this.update(); },
        }, cat + ' (' + allCards.filter(c => (c.category || 'General') === cat).length + ')');
        const editToggle = h('button', { title: 'Toggle card editing (add / remove cards)', className: 'zoomy-btn' + (this.editMode ? ' active' : ''), onClick: () => { this.editMode = !this.editMode; this.update(); } },
            h('span', { className: 'codicon codicon-' + (this.editMode ? 'check' : 'edit'), style: { marginRight: 4 } }), this.editMode ? 'Done editing' : 'Edit cards');
        const subTabs = h('div', { style: { display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 14 } },
            cats.length > 1 ? h('div', { style: { display: 'flex', gap: 6, flexWrap: 'wrap', flex: 1 } }, cats.map(subBtn)) : h('div', { style: { flex: 1 } }),
            editToggle);
        const selName = (dir: string): string => { const c = this.pickedCard(dir); return c ? (c.title || c.id) : '—'; };
        const runBtn: React.CSSProperties = { cursor: this.simBusy ? 'default' : 'pointer', border: 'none', borderRadius: 6, padding: '9px 18px', fontSize: 14, fontWeight: 700, background: 'var(--theia-button-background)', color: 'var(--theia-button-foreground)', opacity: this.simBusy ? 0.7 : 1 };
        const chip = (label: string, val: string) => h('span', { style: { fontSize: 12, color: 'var(--theia-descriptionForeground)' } }, label + ': ', h('span', { style: { color: 'var(--theia-foreground)', fontWeight: 600 } }, val));
        const runBar = h('div', { style: { border: '1px solid var(--theia-panel-border)', borderRadius: 8, padding: 14, marginBottom: 16, background: 'var(--theia-editorWidget-background)' } },
            h('div', { style: { display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' } },
                this.simBusy
                    ? h('button', { style: { ...runBtn, background: 'var(--theia-errorForeground, #d13438)', opacity: 1, cursor: 'pointer' }, onClick: () => this.stopAssembly() },
                        h('span', { className: 'codicon codicon-debug-stop', style: { verticalAlign: 'middle', marginRight: 6 } }), 'Stop simulation')
                    : h('button', { style: runBtn, disabled: !this.kernelReady, onClick: () => this.runAssembly() },
                        h('span', { className: 'codicon codicon-play', style: { verticalAlign: 'middle', marginRight: 6 } }), 'Run simulation'),
                h('div', { style: { display: 'flex', gap: 14, flexWrap: 'wrap' } }, chip('model', selName('models')), chip('mesh', selName('meshes')), chip('solver', selName('solvers')))),
            this.simStatus ? h('div', { style: { fontSize: 12, color: this.simRan && !this.simError ? 'var(--theia-successForeground, #3fb950)' : 'var(--theia-descriptionForeground)', marginTop: 8 } }, this.simStatus) : null,
            // Run computes only — compute errors surface here; the visualization
            // itself renders in the Visualization tab's viewer, never here.
            this.simError ? h('div', { style: { marginTop: 12, borderTop: '1px solid var(--theia-panel-border)', paddingTop: 10, color: 'var(--theia-errorForeground)' } },
                this.simError.stdout ? h('pre', { style: { margin: '2px 0', whiteSpace: 'pre-wrap', fontSize: 12, fontFamily: 'var(--theia-code-font-family, monospace)' } }, this.simError.stdout) : null) : null);
        const tbtn: React.CSSProperties = { cursor: 'pointer', border: '1px solid var(--theia-panel-border)', borderRadius: 6, padding: '5px 11px', fontSize: 12.5, background: 'transparent', color: 'var(--theia-foreground)' };
        const inputS: React.CSSProperties = { background: 'var(--theia-input-background)', color: 'var(--theia-input-foreground)', border: '1px solid var(--theia-input-border, var(--theia-panel-border))', borderRadius: 4, padding: '4px 8px', fontSize: 12.5, minWidth: 220 };
        // Case / Project / Backend actions live in the Zoomy activity-bar view and
        // the top "Zoomy" menu now, not in a self-coded toolbar here. The git row
        // stays (kept, per feedback); native SCM binding is a follow-up.
        return h('div', { style: page },
            // Minimal header: a small kernel-status chip only. Branding, the active
            // case, connected backends and log messages all live in the left Zoomy
            // panel / bottom Log — no need to duplicate them here.
            h('div', { style: { display: 'flex', alignItems: 'center', gap: 6, fontSize: 11.5, marginBottom: 12, color: 'var(--theia-descriptionForeground)' } },
                h('span', { className: 'codicon codicon-' + (this.kernelReady ? 'pass-filled' : 'loading codicon-modifier-spin'), style: { fontSize: 12, color: this.kernelReady ? 'var(--theia-successForeground, #3fb950)' : undefined } }),
                'Kernel: ' + (this.kernelReady ? 'ready' : (this.kernelStatus || 'starting…'))),
            runBar,
            h('div', { style: { display: 'flex', gap: 4, borderBottom: '1px solid var(--theia-panel-border)', marginBottom: subTabs ? 12 : 16 } }, TABS.map(tabBtn)),
            subTabs,
            this.active === 'visualizations' ? h('div', { style: { fontSize: 12, color: 'var(--theia-descriptionForeground)', marginBottom: 10 } }, this.simRan ? 'Open a viewer to tune its Parameters (field + time, right panel) and Render — the plot appears under its card. Tick the ✓ on each viewer you want exported with the case (multi-select).' : 'Run a simulation (the bar above) first, then open a viewer and Render.') : null,
            this.active === 'visualizations' ? this.renderPostproc() : null,
            cards.length ? cards.map(c => this.renderCard(c, this.active)) : h('div', { style: { color: 'var(--theia-descriptionForeground)' } }, 'No cards in this tab.'),
            this.editMode ? h('button', { title: 'Add a card to this tab (duplicates the selected one to start from)', onClick: () => this.addCardToActive(), style: { cursor: 'pointer', border: '1px dashed var(--theia-panel-border)', borderRadius: 8, padding: '10px 14px', fontSize: 13, background: 'transparent', color: 'var(--theia-foreground)', width: '100%', textAlign: 'left' } }, h('span', { className: 'codicon codicon-add', style: { marginRight: 6 } }), 'Add card' + (cats.length > 1 ? ' to "' + activeCat + '"' : '')) : null);
    }

}
