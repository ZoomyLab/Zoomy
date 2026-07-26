import React from '@theia/core/shared/react';
import { injectable, inject, postConstruct } from '@theia/core/shared/inversify';
import { ReactWidget } from '@theia/core/lib/browser/widgets/react-widget';
import { WidgetManager } from '@theia/core/lib/browser';
import { CommandRegistry } from '@theia/core';
import { onCasesChanged, onBackendsChanged } from './zoomy-cli-loader';
import { ZoomyModelConfigWidget } from './model-config-widget';

/** The Zoomy activity-bar view (left panel): the project's CASES (each a folder =
 *  source of truth) plus the case/project/backend actions in a native slot.
 *  Every item just executes a command; the case list mirrors the config widget. */
@injectable()
export class ZoomyViewWidget extends ReactWidget {
    static readonly ID = 'zoomy-view';
    @inject(CommandRegistry) protected readonly commands: CommandRegistry;
    @inject(WidgetManager) protected readonly widgetManager: WidgetManager;
    protected cases: string[] = [];
    protected current = '';
    protected connected: string[] = [];

    @postConstruct()
    protected init(): void {
        this.id = ZoomyViewWidget.ID;
        this.title.label = 'Zoomy';
        this.title.caption = 'Zoomy';
        this.title.iconClass = 'codicon codicon-beaker';
        this.title.closable = true;
        this.addClass('zoomy-view-widget');
        this.node.style.overflow = 'auto';
        onCasesChanged(() => this.refresh());
        onBackendsChanged(() => this.refresh());
        this.refresh();
        this.update();
    }

    protected async refresh(): Promise<void> {
        try {
            const w = (await this.widgetManager.getWidget(ZoomyModelConfigWidget.ID)) as ZoomyModelConfigWidget | undefined;
            if (w) { this.cases = w.cases || []; this.current = w.caseName || ''; this.connected = w.connectedTags || []; this.update(); }
        } catch { /* ignore */ }
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

    protected renderCases(): React.ReactNode {
        const h = React.createElement;
        const item = (name: string): React.ReactNode => {
            const active = name === this.current;
            const s: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 8, width: '100%', cursor: 'pointer', border: 'none', borderLeft: active ? '2px solid var(--theia-button-background)' : '2px solid transparent', background: active ? 'var(--theia-list-activeSelectionBackground, var(--theia-list-hoverBackground))' : 'transparent', color: active ? 'var(--theia-list-activeSelectionForeground, var(--theia-foreground))' : 'var(--theia-foreground)', padding: '5px 8px', fontSize: 13, textAlign: 'left' };
            return h('button', { key: name, style: s, onClick: () => this.commands.executeCommand('zoomy.openNamedCase', name) },
                h('span', { className: 'codicon codicon-' + (active ? 'folder-active' : 'folder') }), name);
        };
        return h('div', { style: { marginBottom: 10 } },
            h('div', { style: { display: 'flex', alignItems: 'center', padding: '4px 8px' } },
                h('div', { style: { flex: 1, fontSize: 11, textTransform: 'uppercase', letterSpacing: '.06em', color: 'var(--theia-descriptionForeground)' } }, 'Cases'),
                h('button', { title: 'New case', style: { cursor: 'pointer', border: 'none', background: 'transparent', color: 'var(--theia-foreground)' }, onClick: () => this.commands.executeCommand('zoomy.newCase') }, h('span', { className: 'codicon codicon-new-folder' }))),
            this.cases.length ? this.cases.map(item) : h('div', { style: { fontSize: 12, color: 'var(--theia-descriptionForeground)', padding: '4px 10px' } }, 'No cases yet — create one.'));
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
                    ['arrow-down', 'Export case (.py)', 'zoomy.exportPy'],
                    ['notebook', 'Export case (.ipynb)', 'zoomy.exportIpynb'],
                    ['arrow-up', 'Import case…', 'zoomy.importCase'],
                ]),
                this.renderBackends()),
            this.renderFooter());
    }

    /** Absolute URL for a bundled gui/ asset (served next to the app). */
    protected asset(file: string): string { try { return new URL('gui/assets/' + file, document.baseURI).href; } catch { return 'gui/assets/' + file; } }

    /** The Zoomy logo + tagline at the top of the panel. */
    protected renderBrand(): React.ReactNode {
        const h = React.createElement;
        return h('div', { style: { display: 'flex', alignItems: 'center', gap: 10, padding: '12px 12px 10px', borderBottom: '1px solid var(--theia-panel-border)' } },
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
            link('github-inverted', 'GitHub repository', 'https://github.com/ZoomyLab/Zoomy', 'Open the Zoomy repository on GitHub'),
            // MBD chair + RWTH Aachen lockup — transparent PNG, no chip. Hidden
            // until the asset is present so there's no broken image.
            h('a', { href: 'https://www.mbd.rwth-aachen.de/', target: '_blank', rel: 'noreferrer', title: 'MBD — RWTH Aachen University', style: { display: 'block', marginTop: 10 } },
                h('img', { src: this.asset('mbd-rwth-logo.png'), alt: 'MBD — RWTH Aachen University', onError: (e: any) => { e.currentTarget.style.display = 'none'; }, style: { width: '100%', maxWidth: 260, height: 'auto', display: 'block' } })));
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
                h('button', { title: 'Scan localhost:8080–8090 for backends', style: { cursor: 'pointer', border: 'none', background: 'transparent', color: 'var(--theia-foreground)' }, onClick: () => this.commands.executeCommand('zoomy.scanBackends') }, h('span', { className: 'codicon codicon-refresh' }))),
            this.connected.length ? this.connected.map(item) : h('div', { style: { fontSize: 12, color: 'var(--theia-descriptionForeground)', padding: '2px 10px' } }, 'None — running in-browser.'),
            h('button', {
                style: rowBtn, onClick: () => this.commands.executeCommand('zoomy.connectBackend'),
                onMouseEnter: (e: any) => { e.currentTarget.style.background = 'var(--theia-list-hoverBackground)'; },
                onMouseLeave: (e: any) => { e.currentTarget.style.background = 'transparent'; },
            }, h('span', { className: 'codicon codicon-plug' }), 'Connect backend…'));
    }
}
