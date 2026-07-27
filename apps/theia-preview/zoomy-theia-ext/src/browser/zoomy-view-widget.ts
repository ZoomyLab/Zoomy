import React from '@theia/core/shared/react';
import { injectable, inject, postConstruct } from '@theia/core/shared/inversify';
import { ReactWidget } from '@theia/core/lib/browser/widgets/react-widget';
import { WidgetManager } from '@theia/core/lib/browser';
import { CommandRegistry } from '@theia/core';
import { onCasesChanged, onBackendsChanged } from './zoomy-cli-loader';
import { ZoomyModelConfigWidget, ensureZoomyStyles } from './model-config-widget';

/** The Zoomy activity-bar view (left panel): the project's CASES (each a folder =
 *  source of truth) plus the case/project/backend actions in a native slot.
 *  Every item just executes a command; the case list mirrors the config widget. */
@injectable()
export class ZoomyViewWidget extends ReactWidget {
    static readonly ID = 'zoomy-view';
    @inject(CommandRegistry) protected readonly commands: CommandRegistry;
    @inject(WidgetManager) protected readonly widgetManager: WidgetManager;
    protected cases: string[] = [];
    protected couplings: Array<{ name: string; children: string[] }> = [];
    protected selected = new Set<string>();   // multi-select for couple/disconnect
    protected current = '';
    protected connected: string[] = [];
    protected commit = '';   // build commit hash from version.json (deploy-injected)

    @postConstruct()
    protected init(): void {
        this.id = ZoomyViewWidget.ID;
        this.title.label = 'Zoomy';
        this.title.caption = 'Zoomy';
        this.title.iconClass = 'codicon codicon-beaker';
        this.title.closable = true;
        this.addClass('zoomy-view-widget');
        ensureZoomyStyles();
        this.node.style.overflow = 'auto';
        this.loadVersion();
        onCasesChanged(() => this.refresh());
        onBackendsChanged(() => this.refresh());
        this.refresh();
        this.update();
    }

    protected async refresh(): Promise<void> {
        try {
            const w = (await this.widgetManager.getWidget(ZoomyModelConfigWidget.ID)) as ZoomyModelConfigWidget | undefined;
            if (w) { this.cases = w.cases || []; this.couplings = w.couplings || []; this.current = w.caseName || ''; this.connected = w.connectedTags || []; for (const s of [...this.selected]) { if (!this.cases.includes(s) && !this.couplings.some(c => c.name === s)) { this.selected.delete(s); } } if (this.selected.size === 0 && this.current) { this.selected.add(this.current); } this.update(); }
        } catch { /* ignore */ }
    }

    /** Re-scan the case folders from the FS whenever the panel is (re)activated,
     *  so a case copy/pasted in the Explorer shows up here. */
    protected onActivateRequest(msg: any): void {
        super.onActivateRequest(msg);
        this.rescanCases();
    }
    protected async rescanCases(): Promise<void> {
        try { const w = (await this.widgetManager.getWidget(ZoomyModelConfigWidget.ID)) as ZoomyModelConfigWidget | undefined; if (w) { await w.rescan(); } }
        catch { /* ignore */ }
        this.refresh();
    }

    protected group(title: string, items: Array<[string, string, string]>): React.ReactNode {
        const h = React.createElement;
        const btn: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 8, width: '100%', cursor: 'pointer', border: 'none', background: 'transparent', color: 'var(--theia-foreground)', padding: '6px 8px', fontSize: 13, textAlign: 'left', borderRadius: 4 };
        return h('div', { key: title, style: { marginBottom: 10 } },
            h('div', { style: { fontSize: 11, textTransform: 'uppercase', letterSpacing: '.06em', color: 'var(--theia-descriptionForeground)', padding: '4px 8px' } }, title),
            items.map(([icon, label, cmd]) => h('button', {
                key: cmd, style: btn, onClick: () => this.commands.executeCommand(cmd),
                onMouseEnter: (e: any) => { e.currentTarget.style.background = 'var(--theia-list-hoverBackground)'; },
                onMouseLeave: (e: any) => { e.currentTarget.style.background = 'transparent'; },
            }, h('span', { className: 'codicon codicon-' + icon }), label)));
    }

    /** Ctrl/Shift-click toggles multi-select; plain click opens + single-selects. */
    protected selectClick(name: string, e: any, opts: { openCmd?: string } = {}): void {
        if (e.shiftKey || e.ctrlKey || e.metaKey) {
            if (this.selected.has(name)) { this.selected.delete(name); } else { this.selected.add(name); }
        } else {
            this.selected.clear(); this.selected.add(name);
            if (opts.openCmd) { this.commands.executeCommand(opts.openCmd, name); }
        }
        this.update();
    }

    protected renderCases(): React.ReactNode {
        const h = React.createElement;
        const sel = [...this.selected];
        const anyChildSelected = sel.some(n => n.includes('/'));
        const coupleReady = sel.length >= 2 && sel.every(n => !n.includes('/') && !this.couplings.some(c => c.name === n));
        const topLeaves = this.cases.filter(n => !n.includes('/'));

        const row = (name: string, opts: { child?: boolean; coupling?: boolean } = {}): React.ReactNode => {
            const active = name === this.current;
            const isSel = this.selected.has(name);
            const label = opts.child ? name.split('/').slice(1).join('/') : name;
            // Selection = thin blue OUTER BORDER only, no background fill (the fill
            // is never used anywhere). Single highlight driven purely by `selected`.
            const s: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer',
                border: '1px solid ' + (isSel ? 'var(--theia-focusBorder, var(--theia-button-background))' : 'transparent'),
                borderRadius: 4, background: 'transparent', color: 'var(--theia-foreground)',
                padding: '4px 8px', margin: '1px 0', marginLeft: opts.child ? 20 : 0, fontSize: 13 };
            const icon = opts.coupling ? 'type-hierarchy-sub' : (opts.child ? 'file-submodule' : (active ? 'folder-active' : 'folder'));
            const rowBtn = (title: string, ic: string, cmd: string): React.ReactNode => h('span', { title, className: 'codicon codicon-' + ic, style: { fontSize: 13, opacity: .55 }, onClick: (e: any) => { e.stopPropagation(); this.commands.executeCommand(cmd, name); } });
            return h('div', { key: name, style: s,
                onClick: (e: any) => this.selectClick(name, e, { openCmd: opts.coupling ? 'zoomy.openCoupling' : 'zoomy.openNamedCase' }),
                onMouseEnter: (e: any) => { if (!isSel) { e.currentTarget.style.background = 'var(--theia-list-hoverBackground)'; } },
                onMouseLeave: (e: any) => { e.currentTarget.style.background = 'transparent'; } },
                h('span', { className: 'codicon codicon-' + icon, style: { color: active ? 'var(--theia-button-background)' : undefined } }),
                h('span', { style: { flex: 1, fontWeight: opts.coupling ? 600 : 400 } }, label),
                opts.coupling ? h('span', { style: { fontSize: 10, opacity: .6 } }, 'coupled') : null,
                opts.coupling ? rowBtn('Run coupled — launch all participants on the foam backend', 'run-all', 'zoomy.runCoupling') : null,
                opts.coupling ? rowBtn('Uncouple all — dissolve this group', 'link-external', 'zoomy.dissolveCoupling') : null,
                !opts.coupling ? rowBtn('Rename case', 'edit', 'zoomy.renameCase') : null,
                !opts.coupling ? rowBtn('Duplicate case', 'copy', 'zoomy.duplicateCase') : null,
                !opts.coupling ? rowBtn('Remove case', 'close', 'zoomy.removeCase') : null);
        };

        const iconBtn = (icon: string, title: string, on: boolean, click: () => void): React.ReactNode =>
            h('button', { title, disabled: !on, style: { cursor: on ? 'pointer' : 'default', border: 'none', background: 'transparent', color: 'var(--theia-foreground)', marginRight: 4, opacity: on ? 1 : 0.35 }, onClick: () => { if (on) { click(); } } }, h('span', { className: 'codicon codicon-' + icon }));

        const coupleOrDisconnect = anyChildSelected
            ? iconBtn('link-external', 'Disconnect the selected case(s) from their coupling', true, () => { sel.filter(n => n.includes('/')).forEach(n => this.commands.executeCommand('zoomy.decoupleCase', n)); this.selected.clear(); this.update(); })
            : iconBtn('link', coupleReady ? 'Couple the selected cases' : 'Ctrl/Shift-click ≥2 cases to couple them', coupleReady, () => { this.commands.executeCommand('zoomy.coupleCases', sel); this.selected.clear(); this.update(); });

        const items: React.ReactNode[] = [];
        for (const cp of this.couplings) { items.push(row(cp.name, { coupling: true })); for (const ch of cp.children) { items.push(row(ch, { child: true })); } }
        for (const n of topLeaves) { items.push(row(n)); }

        return h('div', { style: { marginBottom: 10 } },
            h('div', { style: { display: 'flex', alignItems: 'center', padding: '4px 8px' } },
                h('div', { style: { flex: 1, fontSize: 11, textTransform: 'uppercase', letterSpacing: '.06em', color: 'var(--theia-descriptionForeground)' } }, 'Cases'),
                coupleOrDisconnect,
                h('button', { title: 'Rescan cases (pick up Explorer changes)', style: { cursor: 'pointer', border: 'none', background: 'transparent', color: 'var(--theia-foreground)', marginRight: 4 }, onClick: () => this.rescanCases() }, h('span', { className: 'codicon codicon-refresh' })),
                h('button', { title: 'New case', style: { cursor: 'pointer', border: 'none', background: 'transparent', color: 'var(--theia-foreground)' }, onClick: () => this.commands.executeCommand('zoomy.newCase') }, h('span', { className: 'codicon codicon-new-folder' }))),
            items.length ? items : h('div', { style: { fontSize: 12, color: 'var(--theia-descriptionForeground)', padding: '4px 10px' } }, 'No cases yet — create one.'));
    }

    protected render(): React.ReactNode {
        const h = React.createElement;
        return h('div', { style: { display: 'flex', flexDirection: 'column', minHeight: '100%', fontFamily: 'var(--theia-font-family)' } },
            this.renderBrand(),
            h('div', { style: { flex: '1 1 auto', padding: '8px 4px' } },
                this.renderCases(),
                this.group('Configuration', [
                    ['settings-gear', 'Open model configuration', 'zoomy.openModelConfig'],
                    ['notebook', 'Open in Notebook Mode', 'zoomy.openInNotebook'],
                    ['file-code', 'Open case.py in editor', 'zoomy.openCaseFile'],
                    ['play', 'Run simulation', 'zoomy.run'],
                ]),
                this.group('Project', [
                    ['save', 'Save project', 'zoomy.saveProject'],
                    ['folder-opened', 'Load project', 'zoomy.loadProject'],
                    ['link', 'Generate GUI link', 'zoomy.generateGuiLink'],
                    ['arrow-down', 'Export case (.py)', 'zoomy.exportPy'],
                    ['arrow-down', 'Export case (.ipynb)', 'zoomy.exportIpynb'],
                    ['arrow-up', 'Import case…', 'zoomy.importCase'],
                ]),
                this.renderBackends()),
            this.renderFooter());
    }

    /** Absolute URL for a bundled gui/ asset (served next to the app). */
    protected asset(file: string): string { try { return new URL('gui/assets/' + file, document.baseURI).href; } catch { return 'gui/assets/' + file; } }

    /** Read the deploy-injected version.json ({commit}) for the footer build tag. */
    protected async loadVersion(): Promise<void> {
        try {
            const r = await fetch(new URL('version.json', document.baseURI).href, { cache: 'no-store' });
            if (!r.ok) { return; }
            const v = await r.json();
            this.commit = String(v.commit || v.sha || '').slice(0, 8);
            this.update();
        } catch { /* local/dev build: no version.json */ }
    }

    /** The Zoomy logo + tagline at the top of the panel — clickable, jumps to
     *  the Model configuration tab. */
    protected renderBrand(): React.ReactNode {
        const h = React.createElement;
        return h('div', { title: 'Open model configuration', style: { display: 'flex', alignItems: 'center', gap: 10, padding: '12px 12px 10px', borderBottom: '1px solid var(--theia-panel-border)', cursor: 'pointer' }, onClick: () => this.commands.executeCommand('zoomy.openModelConfig'),
            onMouseEnter: (e: any) => { e.currentTarget.style.background = 'var(--theia-list-hoverBackground)'; },
            onMouseLeave: (e: any) => { e.currentTarget.style.background = 'transparent'; } },
            h('img', { src: this.asset('zoomy-logo.svg'), alt: 'Zoomy', style: { height: 34, width: 'auto', flex: '0 0 auto' } }),
            h('div', null,
                h('div', { style: { fontSize: 17, fontWeight: 800, letterSpacing: '.02em', lineHeight: 1.1 } }, 'Zoomy'),
                h('div', { style: { fontSize: 10.5, color: 'var(--theia-descriptionForeground)' } }, 'Free Surface Flow Modeling')));
    }

    /** Footer: GitHub + MBD-chair links. Swap the text for the real logos when
     *  the assets/URL are provided. */
    protected renderFooter(): React.ReactNode {
        const h = React.createElement;
        const link = (icon: string | null, label: string, href: string, title: string): React.ReactNode => h('a', {
            key: label, href, target: '_blank', rel: 'noreferrer', title,
            style: { display: 'flex', alignItems: 'center', gap: 8, textDecoration: 'none', color: 'var(--theia-foreground)', fontSize: 12, padding: '4px 4px', borderRadius: 4 },
            onMouseEnter: (e: any) => { e.currentTarget.style.background = 'var(--theia-list-hoverBackground)'; },
            onMouseLeave: (e: any) => { e.currentTarget.style.background = 'transparent'; },
        }, icon ? h('span', { className: 'codicon codicon-' + icon }) : null, label);
        return h('div', { style: { flex: '0 0 auto', padding: '8px 8px 12px', borderTop: '1px solid var(--theia-panel-border)' } },
            link('github-inverted', 'GitHub', 'https://github.com/ZoomyLab/Zoomy', 'Open the Zoomy repository on GitHub'),
            link('book', 'Documentation', 'https://zoomylab.github.io/Zoomy/', 'Open the Zoomy documentation'),
            // MBD chair + RWTH Aachen lockup. The .zoomy-mbd-logo class keeps it
            // chip-free on light themes and adds a subtle light backing on dark
            // themes so the dark-navy wordmarks stay legible.
            h('a', { className: 'zoomy-mbd-logo', href: 'https://www.mbd.rwth-aachen.de/', target: '_blank', rel: 'noreferrer', title: 'MBD — RWTH Aachen University' },
                h('img', { src: this.asset('mbd-logo.png'), alt: 'MBD — RWTH Aachen University', onError: (e: any) => { e.currentTarget.style.display = 'none'; } })),
            // Build commit hash (deploy-injected version.json) — for reporting which
            // build is live. Links to the exact commit on GitHub.
            this.commit ? h('a', { href: 'https://github.com/ZoomyLab/Zoomy/commit/' + this.commit, target: '_blank', rel: 'noreferrer', title: 'This build’s commit', style: { display: 'block', marginTop: 8, fontSize: 11, color: 'var(--theia-descriptionForeground)', textDecoration: 'none', fontFamily: 'var(--theia-code-font-family, monospace)' } }, 'build ' + this.commit) : null);
    }

    /** The Backend group: a "Connect backend…" action plus each connected backend
     *  with an ✕ to disconnect it. */
    protected renderBackends(): React.ReactNode {
        const h = React.createElement;
        const rowBtn: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 8, width: '100%', cursor: 'pointer', border: 'none', background: 'transparent', color: 'var(--theia-foreground)', padding: '6px 8px', fontSize: 13, textAlign: 'left', borderRadius: 4 };
        // The in-browser numpy (pyodide) runtime is always-on → no disconnect.
        const item = (tag: string): React.ReactNode => h('div', { key: tag, style: { display: 'flex', alignItems: 'center', gap: 8, padding: '4px 8px', fontSize: 13 } },
            h('span', { className: 'codicon codicon-pass-filled', style: { color: 'var(--theia-successForeground, #3fb950)' } }),
            h('span', { style: { flex: 1 } }, tag),
            tag.indexOf('numpy') === 0 ? null
                : h('button', { title: 'Disconnect ' + tag, style: { cursor: 'pointer', border: 'none', background: 'transparent', color: 'var(--theia-descriptionForeground)' }, onClick: () => this.commands.executeCommand('zoomy.disconnectBackend', tag) }, h('span', { className: 'codicon codicon-close' })));
        return h('div', { key: 'backend', style: { marginBottom: 10 } },
            h('div', { style: { display: 'flex', alignItems: 'center', padding: '4px 8px' } },
                h('div', { style: { flex: 1, fontSize: 11, textTransform: 'uppercase', letterSpacing: '.06em', color: 'var(--theia-descriptionForeground)' } }, 'Backend'),
                h('button', { title: 'Scan localhost:8080–8100 for backends', style: { cursor: 'pointer', border: 'none', background: 'transparent', color: 'var(--theia-foreground)' }, onClick: () => this.commands.executeCommand('zoomy.scanBackends') }, h('span', { className: 'codicon codicon-refresh' }))),
            this.connected.length ? this.connected.map(item) : h('div', { style: { fontSize: 12, color: 'var(--theia-descriptionForeground)', padding: '2px 10px' } }, 'None — running in-browser.'),
            h('button', {
                style: rowBtn, onClick: () => this.commands.executeCommand('zoomy.connectBackend'),
                onMouseEnter: (e: any) => { e.currentTarget.style.background = 'var(--theia-list-hoverBackground)'; },
                onMouseLeave: (e: any) => { e.currentTarget.style.background = 'transparent'; },
            }, h('span', { className: 'codicon codicon-plug' }), 'Connect backend…'));
    }
}
