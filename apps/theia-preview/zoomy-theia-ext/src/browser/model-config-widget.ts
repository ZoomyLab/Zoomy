import React from '@theia/core/shared/react';
import { injectable, inject, postConstruct } from '@theia/core/shared/inversify';
import { ReactWidget } from '@theia/core/lib/browser/widgets/react-widget';
import { URI } from '@theia/core';
import { FileService } from '@theia/filesystem/lib/browser/file-service';
import { getZoomyCli, setDisplaySink, setLogSink, ensureRenderLibs, DisplayCell } from './zoomy-cli-loader';

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
    const blob = new Blob([text], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
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
    protected loaded = false;
    protected error = '';
    protected kernelStatus = '';
    protected kernelReady = false;
    protected readonly outputs = new Map<string, CardOut>();
    // Param editing (the "gear"): open state, loaded schema, and edited values.
    protected readonly paramsOpen = new Set<string>();
    protected readonly schemas = new Map<string, any>();
    protected readonly edited = new Map<string, any>();
    // Accordion (one expanded card) + selection (one selected card per tab).
    protected expanded: string | undefined;
    protected readonly selected: Record<string, string> = {};
    // Assembled simulation: run selected model→mesh→solver→viz in shared scope.
    protected simRan = false;
    protected simBusy = false;
    protected simStatus = '';
    protected vizOut: CardOut | undefined;
    protected storeMeta: any;
    // Case interchange (#3/#5), project persistence (#6), backends (#4).
    protected notice = '';
    backendUrl = 'http://localhost:8080';
    connectedTags: string[] = [];
    // Case-as-source-of-truth. The GUI is only usable with an open case; every
    // edit is written back to the case folder's case.py.
    @inject(FileService) protected readonly fileService: FileService;
    protected caseUri: URI | undefined;
    protected caseName = '';
    protected cases: string[] = [];
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
                if (/Booting|Installing|Kernel ready|runtime ready|installing|cache|ready/i.test(msg)) {
                    this.kernelStatus = msg;
                    if (/runtime ready|Kernel ready/i.test(msg)) { this.kernelReady = true; }
                    this.update();
                }
            });
            this.cli = await getZoomyCli();
            // Warm the Pyodide worker NOW (it auto-boots on creation) so the first
            // Run isn't stuck behind the cold boot + param pre-extract.
            this.cli.runCode('pass').catch(() => { /* background warm-up */ });
            for (const t of TABS) {
                try { this.cardsByTab[t.dir] = await this.cli.listCards(t.dir); }
                catch (e) { this.cardsByTab[t.dir] = []; }
            }
            this.loaded = true;
            await this.listCases();
            // #4 URL-autoload: ?case=<url> imports a case into a new case.
            const caseUrl = new URLSearchParams(location.search).get('case');
            if (caseUrl) {
                try { const text = await (await fetch(caseUrl)).text(); await this.newCase('imported', this.cli.parseCase(text)); } catch { /* ignore */ }
            } else {
                // Restore the last open case (single source of truth); else stay gated.
                const last = (() => { try { return localStorage.getItem(CURRENT_CASE_KEY); } catch { return null; } })();
                if (last && this.cases.includes(last)) { await this.openCaseByName(last); }
            }
        } catch (e: any) {
            this.error = e?.message || String(e);
        }
        this.update();
    }

    // === Case as the single source of truth =================================
    protected caseFileUri(name: string): URI { return new URI(PROJECT_ROOT + '/' + name + '/case.py'); }

    /** List case folders under the project root (each has a case.py). */
    protected async listCases(): Promise<void> {
        try {
            const root = new URI(PROJECT_ROOT);
            if (!(await this.fileService.exists(root))) { this.cases = []; return; }
            const stat = await this.fileService.resolve(root);
            const names: string[] = [];
            for (const child of stat.children || []) {
                if (child.isDirectory && await this.fileService.exists(child.resource.resolve('case.py'))) { names.push(child.resource.path.base); }
            }
            this.cases = names.sort();
        } catch { this.cases = []; }
    }

    /** Create a new case folder with a case.py, then open it. If a spec is given
     *  (import), use it; otherwise start from the first runnable card in each tab. */
    async newCase(name: string, spec?: any): Promise<void> {
        const clean = (name || 'case').trim().replace(/[^a-zA-Z0-9_-]+/g, '_') || 'case';
        this.selected['models'] = ''; this.selected['meshes'] = ''; this.selected['solvers'] = ''; this.selected['visualizations'] = '';
        this.edited.clear();
        if (spec) { this.applySpec(spec); }
        else {
            for (const dir of ['models', 'meshes', 'solvers', 'visualizations']) { const c = this.pickedCard(dir); if (c) { this.selected[dir] = c.id; } }
        }
        this.caseUri = this.caseFileUri(clean); this.caseName = clean;
        await this.persistCase();
        await this.listCases();
        try { localStorage.setItem(CURRENT_CASE_KEY, clean); } catch { /* ignore */ }
        this.setNotice('Created case "' + clean + '".'); this.update();
        this.onBackendsChanged?.(this.connectedTags);
    }

    async openCaseByName(name: string): Promise<void> {
        try {
            const uri = this.caseFileUri(name);
            const content = await this.fileService.read(uri);
            this.applySpec(this.cli.parseCase(content.value));
            this.caseUri = uri; this.caseName = name;
            try { localStorage.setItem(CURRENT_CASE_KEY, name); } catch { /* ignore */ }
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
            await this.fileService.write(this.caseUri, py);
        } catch (e: any) { console.error('persistCase', e); }
    }
    protected schedulePersist(): void {
        if (!this.caseUri) { return; }
        clearTimeout(this.persistTimer);
        this.persistTimer = setTimeout(() => this.persistCase(), 600);
    }

    /** card.init overlaid with the user's edits from the param form. */
    protected mergedInit(card: any): any { return { ...(card.init || {}), ...(this.edited.get(card.id) || {}) }; }

    protected async toggleParams(card: any): Promise<void> {
        if (this.paramsOpen.has(card.id)) { this.paramsOpen.delete(card.id); this.update(); return; }
        this.paramsOpen.add(card.id); this.update();
        if (!this.schemas.has(card.id)) {
            // Inline params-card schema needs no worker; class-cards introspect
            // via extract_params (cached in the worker after boot); builtin cards
            // (mesh) expose their init directly ({placeholder} edits bite there).
            if (card.params) { this.schemas.set(card.id, card.params); }
            else if (card.class) {
                try { const res = await this.cli.extractParams(card.class, card.init || {}); this.schemas.set(card.id, res?.params || {}); }
                catch (e) { this.schemas.set(card.id, deriveSchema(card.init)); }
            } else { this.schemas.set(card.id, deriveSchema(card.init)); }
            this.update();
        }
    }

    protected setParam(card: any, name: string, value: any): void {
        const e = this.edited.get(card.id) || {}; e[name] = value; this.edited.set(card.id, e); this.update();
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
            } else if (type === 'Integer' || type === 'Number') {
                const step = type === 'Integer' ? 1 : (p.step || 'any');
                input = h('input', { type: 'number', step, style: inputS, value: cur == null ? '' : cur, onChange: (e: any) => { const v = e.target.value; this.setParam(card, name, v === '' ? null : (type === 'Integer' ? parseInt(v, 10) : parseFloat(v))); } });
            } else {
                input = h('span', { style: { fontSize: 12, color: 'var(--theia-descriptionForeground)' } }, JSON.stringify(cur));
            }
            return h('div', { key: name, style: rowS },
                h('label', { style: labelS, title: p.doc || '' }, name),
                input,
                (p.bounds && Array.isArray(p.bounds)) ? h('span', { style: { fontSize: 11, color: 'var(--theia-descriptionForeground)' } }, '[' + (p.bounds[0] ?? '') + ', ' + (p.bounds[1] ?? '') + ']') : null);
        };
        return h('div', { style: { marginTop: 10, borderTop: '1px dashed var(--theia-panel-border)', paddingTop: 8 } }, names.map(field));
    }

    /** Clicking a card selects it in its tab and expands it (accordion: one open).
     *  Selecting a visualization after a sim has run re-renders it (viz changes). */
    protected pick(card: any, dir: string): void {
        const wasExpanded = this.expanded === card.id;
        this.selected[dir] = card.id;
        this.expanded = wasExpanded ? undefined : card.id;
        this.update();
        this.schedulePersist();
        if (dir === 'visualizations' && this.simRan && !this.simBusy && card.snippet) { this.runViz(card); }
    }

    /** The selected card for a tab, else the first with runnable code (models/
     *  meshes/solvers) or the first snippet card (visualizations). */
    protected pickedCard(dir: string): any {
        const cards = this.cardsByTab[dir] || [];
        const sel = cards.find(c => c.id === this.selected[dir]);
        if (sel) { return sel; }
        if (dir === 'visualizations') { return cards.find(c => c.snippet); }
        return cards.find(c => !!cardCode(c, this.mergedInit(c)));
    }

    /** Run the selected model → mesh → solver in the shared scope, then the
     *  selected visualization. This is the "assembly → visualization" flow. */
    async runAssembly(): Promise<void> {
        if (this.simBusy) { return; }
        this.simBusy = true; this.simRan = false; this.vizOut = undefined;
        try {
            for (const [dir, label] of [['models', 'model'], ['meshes', 'mesh'], ['solvers', 'solver']] as const) {
                const card = this.pickedCard(dir);
                if (!card) { this.simStatus = 'No ' + label + ' selected.'; this.update(); return; }
                const code = cardCode(card, this.mergedInit(card));
                if (!code) { this.simStatus = label + ' "' + (card.title || card.id) + '" is a remote backend — connect a backend to run it.'; this.update(); return; }
                this.simStatus = 'Running ' + label + ': ' + (card.title || card.id) + '…'; this.update();
                const res = await this.cli.runCode(code);
                if (res?.status === 'error') { this.simStatus = 'Error in ' + label + ': see below'; this.vizOut = { cells: [], stdout: res.output || '', status: 'error', running: false }; this.update(); return; }
                if (res?.store_meta) { this.storeMeta = res.store_meta; }
            }
            this.simRan = true; this.simStatus = 'Simulation done — rendering visualization…'; this.update();
            await this.runViz(this.pickedCard('visualizations'));
            this.simStatus = 'Done.';
        } catch (e: any) {
            this.simStatus = 'Error: ' + (e?.message || String(e));
        } finally {
            this.simBusy = false; this.update();
        }
    }

    /** Run one visualization snippet against the current store and show its plot. */
    protected async runViz(vizCard: any): Promise<void> {
        if (!vizCard?.snippet) { return; }
        const out: CardOut = { cells: [], stdout: '', status: 'running', running: true };
        this.vizOut = out; this.update();
        setDisplaySink(cell => { out.cells.push(cell); this.update(); });
        try {
            const snippet = await this.cli.fetchSnippet(vizCard.snippet);
            // Let the snippet default field/step (proven path). field_name=None →
            // first store field; time_step=0 → first snapshot. (Field/timeline
            // selectors are a follow-up.)
            const code = 'time_step = 0\nfield_name = None\n' + snippet;
            const res = await this.cli.runCode(code);
            out.stdout = res?.output || ''; out.status = res?.status || 'success';
        } catch (e: any) {
            out.status = 'error'; out.stdout = e?.message || String(e);
        } finally {
            setDisplaySink(undefined); out.running = false; this.update();
        }
    }

    protected setNotice(msg: string): void { this.notice = msg; this.update(); if (msg) { setTimeout(() => { if (this.notice === msg) { this.notice = ''; this.update(); } }, 6000); } }

    // --- #3/#5 Case interchange via zoomy_prepost.case (through zoomy_cli). ---
    /** Build the canonical case spec from the current selection + edits. */
    protected async gatherSpec(): Promise<any> {
        const model = this.pickedCard('models'), mesh = this.pickedCard('meshes'), solver = this.pickedCard('solvers'), viz = this.pickedCard('visualizations');
        const spec: any = {
            meta: { title: (model?.title || 'Zoomy case'), description: 'Exported from the Zoomy model-config GUI.' },
            model: { code: cardCode(model, this.mergedInit(model)) || '', class_path: model?.class || null, init: this.mergedInit(model) },
            mesh: { code: cardCode(mesh, this.mergedInit(mesh)) || '', spec: this.mergedInit(mesh) },
            settings: {},
            solver: { tag: solver?.requires_tag || 'numpy', params: solver?.params ? this.mergedInit(solver) : {} },
        };
        const solverCode = cardCode(solver, this.mergedInit(solver));
        if (solverCode) { spec.run = { code: solverCode }; }
        if (viz?.snippet) {
            try { const snip = await this.cli.fetchSnippet(viz.snippet); spec.visualization = { code: this.cli.vizPrelude() + '\n' + snip }; } catch { /* skip viz */ }
        }
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
        const byClass = (dir: string, cls: string) => (this.cardsByTab[dir] || []).find(c => c.class === cls);
        if (spec?.model?.class_path) { const c = byClass('models', spec.model.class_path); if (c) { this.selected['models'] = c.id; if (spec.model.init) { this.edited.set(c.id, { ...spec.model.init }); } } }
        if (spec?.mesh?.spec) { const meshes = this.cardsByTab['meshes'] || []; const c = meshes[0]; if (c) { this.selected['meshes'] = c.id; this.edited.set(c.id, { ...spec.mesh.spec }); } }
        if (spec?.solver?.tag) { const c = (this.cardsByTab['solvers'] || []).find(s => (s.requires_tag || 'numpy') === spec.solver.tag); if (c) { this.selected['solvers'] = c.id; } }
        this.update();
    }

    // --- #6 Project persistence (IndexedDB via zoomy_cli storage). ---
    async saveProject(): Promise<void> {
        const data = { selected: this.selected, edited: Array.from(this.edited.entries()), active: this.active };
        try { await this.cli.storage.writeJson('projects/current.json', data); this.setNotice('Project saved to browser (IndexedDB).'); }
        catch (e: any) { this.setNotice('Save failed: ' + (e?.message || e)); }
        // also offer a download so it can be shared / version-controlled
        download('zoomy_project.json', JSON.stringify(data, null, 2), 'application/json');
    }
    async loadProject(): Promise<void> {
        try {
            const data = await this.cli.storage.tryReadJson('projects/current.json');
            if (!data) { this.setNotice('No saved project in this browser.'); return; }
            Object.assign(this.selected, data.selected || {});
            this.edited.clear(); for (const [k, v] of (data.edited || [])) { this.edited.set(k, v); }
            if (data.active) { this.active = data.active; }
            this.setNotice('Project loaded.'); this.update();
        } catch (e: any) { this.setNotice('Load failed: ' + (e?.message || e)); }
    }

    // --- #4 Connect a remote backend by URL. ---
    async connectBackend(): Promise<void> {
        const url = this.backendUrl.trim(); if (!url) { return; }
        this.setNotice('Connecting to ' + url + '…');
        try {
            const adapter = await this.cli.connect(url);
            const tag = adapter?.tag || (this.cli.availableTags ? this.cli.availableTags() : []).slice(-1)[0];
            this.connectedTags = this.cli.availableTags ? this.cli.availableTags() : (tag ? [tag] : []);
            this.setNotice('Connected backend: ' + (tag || url));
            this.onBackendsChanged?.(this.connectedTags);
            try { this.cli.onConnectionsChange && this.cli.onConnectionsChange(() => { this.connectedTags = this.cli.availableTags(); this.onBackendsChanged?.(this.connectedTags); this.update(); }); } catch { /* ignore */ }
            this.update();
        } catch (e: any) { this.setNotice('Connect failed: ' + (e?.message || e) + ' — is a zoomy-server running there?'); }
    }

        protected renderCard(card: any, dir: string): React.ReactNode {
        const h = React.createElement;
        const runnable = !!cardCode(card, this.mergedInit(card));
        const out = this.outputs.get(card.id);
        const hasParams = !!(card.params || card.class || (card.init && Object.keys(card.init).length));
        const open = this.paramsOpen.has(card.id);
        const isSel = this.selected[dir] === card.id;
        const isExp = this.expanded === card.id;
        const cardStyle: React.CSSProperties = {
            border: '1px solid ' + (isSel ? 'var(--theia-focusBorder, var(--theia-button-background))' : 'var(--theia-editorWidget-border, var(--theia-panel-border))'),
            borderLeft: (isSel ? '3px solid var(--theia-button-background)' : '1px solid var(--theia-editorWidget-border, var(--theia-panel-border))'),
            borderRadius: 8, padding: 14, marginBottom: 12, background: 'var(--theia-editorWidget-background)',
        };
        const btn: React.CSSProperties = { cursor: runnable ? 'pointer' : 'not-allowed', border: 'none', borderRadius: 6, padding: '6px 14px', fontSize: 13, fontWeight: 600, background: runnable ? 'var(--theia-button-background)' : 'var(--theia-button-secondaryBackground)', color: 'var(--theia-button-foreground)', opacity: runnable ? 1 : 0.6 };
        const gearBtn: React.CSSProperties = { cursor: 'pointer', border: '1px solid var(--theia-panel-border)', borderRadius: 6, padding: '5px 10px', fontSize: 12.5, background: open ? 'var(--theia-button-secondaryBackground)' : 'transparent', color: 'var(--theia-foreground)' };
        const header = h('div', { style: { display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer' }, onClick: () => this.pick(card, dir) },
            h('span', { className: 'codicon codicon-' + (isSel ? 'pass-filled' : 'circle-large-outline'), style: { color: isSel ? 'var(--theia-button-background)' : 'var(--theia-descriptionForeground)' } }),
            h('div', { style: { fontWeight: 600, fontSize: 14, flex: 1 } }, card.title || card.id),
            card.requires_tag ? h('span', { style: { fontSize: 11, padding: '1px 6px', borderRadius: 4, background: 'var(--theia-badge-background)', color: 'var(--theia-badge-foreground)' } }, card.requires_tag) : null,
            h('span', { className: 'codicon codicon-chevron-' + (isExp ? 'down' : 'right'), style: { color: 'var(--theia-descriptionForeground)' } }));
        // Collapsed: header only. Expanded: full detail (description + params + run + output).
        const body = !isExp ? null : h('div', { style: { marginTop: 8 } },
            card.description ? h('div', { className: 'zoomy-md', style: { color: 'var(--theia-descriptionForeground)', fontSize: 12.5 }, dangerouslySetInnerHTML: { __html: renderMathMd(card.description) } }) : null,
            !runnable ? h('div', { style: { color: 'var(--theia-descriptionForeground)', fontSize: 12, marginTop: 6, fontStyle: 'italic' } }, 'Remote backend card — connect a backend to run.') : null,
            h('div', { style: { display: 'flex', gap: 8, marginTop: 10 } },
                hasParams ? h('button', { style: gearBtn, onClick: () => this.toggleParams(card) }, h('span', { className: 'codicon codicon-settings-gear', style: { verticalAlign: 'middle', marginRight: 4 } }), 'Parameters') : null,
                h('button', { style: btn, disabled: !runnable || (out && out.running), onClick: () => runnable && this.runCard(card) }, out && out.running ? 'Running…' : 'Run')),
            open ? this.renderParamForm(card) : null,
            out ? h('div', { style: { marginTop: 10, borderTop: '1px solid var(--theia-panel-border)', paddingTop: 8, color: out.status === 'error' ? 'var(--theia-errorForeground)' : undefined } },
                out.cells.map((c, i) => this.renderCell(c, 'c' + i)),
                out.stdout ? h('pre', { style: { margin: '2px 0', whiteSpace: 'pre-wrap', fontSize: 12, fontFamily: 'var(--theia-code-font-family, monospace)' } }, out.stdout) : null) : null);
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
            key: t.dir, onClick: () => { this.active = t.dir; this.update(); },
            style: { cursor: 'pointer', border: 'none', borderBottom: this.active === t.dir ? '2px solid var(--theia-button-background)' : '2px solid transparent', background: 'transparent', color: this.active === t.dir ? 'var(--theia-foreground)' : 'var(--theia-descriptionForeground)', padding: '8px 14px', fontSize: 13, fontWeight: 600 },
        }, t.label + ' (' + (this.cardsByTab[t.dir]?.length || 0) + ')');
        const cards = this.cardsByTab[this.active] || [];
        const selName = (dir: string): string => { const c = this.pickedCard(dir); return c ? (c.title || c.id) : '—'; };
        const runBtn: React.CSSProperties = { cursor: this.simBusy ? 'default' : 'pointer', border: 'none', borderRadius: 6, padding: '9px 18px', fontSize: 14, fontWeight: 700, background: 'var(--theia-button-background)', color: 'var(--theia-button-foreground)', opacity: this.simBusy ? 0.7 : 1 };
        const chip = (label: string, val: string) => h('span', { style: { fontSize: 12, color: 'var(--theia-descriptionForeground)' } }, label + ': ', h('span', { style: { color: 'var(--theia-foreground)', fontWeight: 600 } }, val));
        const runBar = h('div', { style: { border: '1px solid var(--theia-panel-border)', borderRadius: 8, padding: 14, marginBottom: 16, background: 'var(--theia-editorWidget-background)' } },
            h('div', { style: { display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' } },
                h('button', { style: runBtn, disabled: this.simBusy || !this.kernelReady, onClick: () => this.runAssembly() },
                    h('span', { className: 'codicon codicon-play', style: { verticalAlign: 'middle', marginRight: 6 } }), this.simBusy ? 'Running…' : 'Run simulation'),
                h('div', { style: { display: 'flex', gap: 14, flexWrap: 'wrap' } }, chip('model', selName('models')), chip('mesh', selName('meshes')), chip('solver', selName('solvers')), chip('viz', selName('visualizations')))),
            this.simStatus ? h('div', { style: { fontSize: 12, color: 'var(--theia-descriptionForeground)', marginTop: 8 } }, this.simStatus) : null,
            this.vizOut ? h('div', { style: { marginTop: 12, borderTop: '1px solid var(--theia-panel-border)', paddingTop: 10, color: this.vizOut.status === 'error' ? 'var(--theia-errorForeground)' : undefined } },
                this.vizOut.cells.map((c, i) => this.renderCell(c, 'v' + i)),
                this.vizOut.stdout ? h('pre', { style: { margin: '2px 0', whiteSpace: 'pre-wrap', fontSize: 12, fontFamily: 'var(--theia-code-font-family, monospace)' } }, this.vizOut.stdout) : null) : null);
        const tbtn: React.CSSProperties = { cursor: 'pointer', border: '1px solid var(--theia-panel-border)', borderRadius: 6, padding: '5px 11px', fontSize: 12.5, background: 'transparent', color: 'var(--theia-foreground)' };
        const inputS: React.CSSProperties = { background: 'var(--theia-input-background)', color: 'var(--theia-input-foreground)', border: '1px solid var(--theia-input-border, var(--theia-panel-border))', borderRadius: 4, padding: '4px 8px', fontSize: 12.5, minWidth: 220 };
        // Case / Project / Backend actions live in the Zoomy activity-bar view and
        // the top "Zoomy" menu now, not in a self-coded toolbar here. The git row
        // stays (kept, per feedback); native SCM binding is a follow-up.
        return h('div', { style: page },
            h('div', { style: { display: 'flex', alignItems: 'baseline', gap: 12, margin: '0 0 4px' } },
                h('h1', { style: { fontSize: 26, fontWeight: 700, margin: 0 } }, 'Model configuration'),
                h('span', { style: { fontSize: 13, color: 'var(--theia-descriptionForeground)' } }, 'case: ', h('span', { style: { color: 'var(--theia-foreground)', fontWeight: 600 } }, this.caseName || '—'))),
            h('div', { style: { color: 'var(--theia-descriptionForeground)', fontSize: 13, marginBottom: 10 } },
                'Editing case ', h('strong', null, this.caseName), ' — the folder is the source of truth; every change is saved back to it. Select a model, mesh, solver and visualization, then Run.'),
            h('div', { style: { display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, marginBottom: 14, color: this.kernelReady ? 'var(--theia-descriptionForeground)' : 'var(--theia-foreground)' } },
                h('span', { className: 'codicon codicon-' + (this.kernelReady ? 'pass-filled' : 'loading codicon-modifier-spin'), style: { color: this.kernelReady ? 'var(--theia-successForeground, #3fb950)' : undefined } }),
                'Kernel: ' + (this.kernelReady ? 'ready' : (this.kernelStatus || 'starting…')) + ' (first boot takes ~2–3 min, then cached)',
                this.connectedTags.length ? h('span', { style: { fontSize: 11, marginLeft: 12, color: 'var(--theia-successForeground, #3fb950)' } }, '● backend: ' + this.connectedTags.join(', ')) : null),
            this.notice ? h('div', { style: { fontSize: 12, color: 'var(--theia-notificationsInfoIcon-foreground, var(--theia-foreground))', marginBottom: 12 } }, this.notice) : null,
            runBar,
            h('div', { style: { display: 'flex', gap: 4, borderBottom: '1px solid var(--theia-panel-border)', marginBottom: 16 } }, TABS.map(tabBtn)),
            cards.length ? cards.map(c => this.renderCard(c, this.active)) : h('div', { style: { color: 'var(--theia-descriptionForeground)' } }, 'No cards in this tab.'));
    }
}
