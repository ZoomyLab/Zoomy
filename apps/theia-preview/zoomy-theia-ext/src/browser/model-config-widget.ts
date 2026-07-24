import React from '@theia/core/shared/react';
import { injectable, postConstruct } from '@theia/core/shared/inversify';
import { ReactWidget } from '@theia/core/lib/browser/widgets/react-widget';
import { getZoomyCli, setDisplaySink, setLogSink, DisplayCell } from './zoomy-cli-loader';

interface CardOut { cells: DisplayCell[]; stdout: string; status: string; running: boolean; }
interface TabDef { dir: string; label: string; }
const TABS: TabDef[] = [
    { dir: 'models', label: 'Model' },
    { dir: 'meshes', label: 'Mesh' },
    { dir: 'solvers', label: 'Solver' },
    { dir: 'visualizations', label: 'Visualization' },
];

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
    protected readonly outputs = new Map<string, CardOut>();
    // Param editing (the "gear"): open state, loaded schema, and edited values.
    protected readonly paramsOpen = new Set<string>();
    protected readonly schemas = new Map<string, any>();
    protected readonly edited = new Map<string, any>();

    @postConstruct()
    protected init(): void {
        this.id = ZoomyModelConfigWidget.ID;
        this.title.label = 'Model configuration';
        this.title.caption = 'Zoomy — model configuration';
        this.title.iconClass = 'codicon codicon-settings-gear';
        this.title.closable = true;
        this.addClass('zoomy-modelconfig-widget');
        this.node.style.overflow = 'auto';
        this.load();
        this.update();
    }

    protected async load(): Promise<void> {
        try {
            setLogSink((lvl, msg) => console.log('[zoomy-cli]', lvl, msg));
            this.cli = await getZoomyCli();
            for (const t of TABS) {
                try { this.cardsByTab[t.dir] = await this.cli.listCards(t.dir); }
                catch (e) { this.cardsByTab[t.dir] = []; }
            }
            this.loaded = true;
        } catch (e: any) {
            this.error = e?.message || String(e);
        }
        this.update();
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
        if (mime === 'text/html') { return h('div', { key, dangerouslySetInnerHTML: { __html: cell.content } }); }
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

    protected renderCard(card: any): React.ReactNode {
        const h = React.createElement;
        const runnable = !!cardCode(card, this.mergedInit(card));
        const out = this.outputs.get(card.id);
        const hasParams = !!(card.params || card.class || (card.init && Object.keys(card.init).length));
        const open = this.paramsOpen.has(card.id);
        const cardStyle: React.CSSProperties = { border: '1px solid var(--theia-editorWidget-border, var(--theia-panel-border))', borderRadius: 8, padding: 14, marginBottom: 14, background: 'var(--theia-editorWidget-background)' };
        const btn: React.CSSProperties = { cursor: runnable ? 'pointer' : 'not-allowed', border: 'none', borderRadius: 6, padding: '6px 14px', fontSize: 13, fontWeight: 600, background: runnable ? 'var(--theia-button-background)' : 'var(--theia-button-secondaryBackground)', color: 'var(--theia-button-foreground)', opacity: runnable ? 1 : 0.6 };
        const gearBtn: React.CSSProperties = { cursor: 'pointer', border: '1px solid var(--theia-panel-border)', borderRadius: 6, padding: '5px 10px', fontSize: 12.5, background: open ? 'var(--theia-button-secondaryBackground)' : 'transparent', color: 'var(--theia-foreground)' };
        return h('div', { key: card.id, style: cardStyle },
            h('div', { style: { display: 'flex', alignItems: 'center', gap: 10 } },
                h('div', { style: { fontWeight: 600, fontSize: 14, flex: 1 } }, card.title || card.id),
                card.requires_tag ? h('span', { style: { fontSize: 11, padding: '1px 6px', borderRadius: 4, background: 'var(--theia-badge-background)', color: 'var(--theia-badge-foreground)' } }, card.requires_tag) : null,
                hasParams ? h('button', { style: gearBtn, onClick: () => this.toggleParams(card) }, h('span', { className: 'codicon codicon-settings-gear', style: { verticalAlign: 'middle', marginRight: 4 } }), 'Parameters') : null,
                h('button', { style: btn, disabled: !runnable || (out && out.running), onClick: () => runnable && this.runCard(card) },
                    out && out.running ? 'Running…' : 'Run')),
            card.description ? h('div', { style: { color: 'var(--theia-descriptionForeground)', fontSize: 12.5, marginTop: 6, whiteSpace: 'pre-wrap' }, dangerouslySetInnerHTML: { __html: mdInline(card.description) } }) : null,
            !runnable ? h('div', { style: { color: 'var(--theia-descriptionForeground)', fontSize: 12, marginTop: 6, fontStyle: 'italic' } }, 'Remote backend card — connect a backend to run.') : null,
            open ? this.renderParamForm(card) : null,
            out ? h('div', { style: { marginTop: 10, borderTop: '1px solid var(--theia-panel-border)', paddingTop: 8, fontFamily: 'var(--theia-code-font-family, monospace)', color: out.status === 'error' ? 'var(--theia-errorForeground)' : undefined } },
                out.cells.map((c, i) => this.renderCell(c, 'c' + i)),
                out.stdout ? h('pre', { style: { margin: '2px 0', whiteSpace: 'pre-wrap', fontSize: 12 } }, out.stdout) : null) : null);
    }

    protected render(): React.ReactNode {
        const h = React.createElement;
        const page: React.CSSProperties = { maxWidth: 900, margin: '0 auto', padding: '32px 24px', color: 'var(--theia-foreground)', fontFamily: 'var(--theia-font-family)' };
        if (this.error) { return h('div', { style: page }, h('h2', null, 'Model configuration'), h('pre', { style: { color: 'var(--theia-errorForeground)' } }, this.error)); }
        if (!this.loaded) { return h('div', { style: page }, h('h2', null, 'Model configuration'), h('div', { style: { color: 'var(--theia-descriptionForeground)' } }, 'Loading the card catalog + booting the in-browser kernel…')); }
        const tabBtn = (t: TabDef): React.ReactNode => h('button', {
            key: t.dir, onClick: () => { this.active = t.dir; this.update(); },
            style: { cursor: 'pointer', border: 'none', borderBottom: this.active === t.dir ? '2px solid var(--theia-button-background)' : '2px solid transparent', background: 'transparent', color: this.active === t.dir ? 'var(--theia-foreground)' : 'var(--theia-descriptionForeground)', padding: '8px 14px', fontSize: 13, fontWeight: 600 },
        }, t.label + ' (' + (this.cardsByTab[t.dir]?.length || 0) + ')');
        const cards = this.cardsByTab[this.active] || [];
        return h('div', { style: page },
            h('h1', { style: { fontSize: 26, margin: '0 0 4px', fontWeight: 700 } }, 'Model configuration'),
            h('div', { style: { color: 'var(--theia-descriptionForeground)', fontSize: 13, marginBottom: 16 } },
                'The real Zoomy card catalog, loaded through ', h('code', null, 'zoomy_cli'), ' and run on the in-browser Pyodide kernel. Pick a card and Run.'),
            h('div', { style: { display: 'flex', gap: 4, borderBottom: '1px solid var(--theia-panel-border)', marginBottom: 16 } }, TABS.map(tabBtn)),
            cards.length ? cards.map(c => this.renderCard(c)) : h('div', { style: { color: 'var(--theia-descriptionForeground)' } }, 'No cards in this tab.'));
    }
}
