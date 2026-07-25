import React from '@theia/core/shared/react';
import { injectable, inject, postConstruct } from '@theia/core/shared/inversify';
import { ReactWidget } from '@theia/core/lib/browser/widgets/react-widget';
import { WidgetManager } from '@theia/core/lib/browser';
import { CommandRegistry } from '@theia/core';
import { onCasesChanged } from './zoomy-cli-loader';
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
        this.refresh();
        this.update();
    }

    protected async refresh(): Promise<void> {
        try {
            const w = (await this.widgetManager.getWidget(ZoomyModelConfigWidget.ID)) as ZoomyModelConfigWidget | undefined;
            if (w) { this.cases = w.cases || []; this.current = w.caseName || ''; this.update(); }
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
        return h('div', { style: { padding: '8px 4px', fontFamily: 'var(--theia-font-family)' } },
            this.renderCases(),
            this.group('Configuration', [
                ['settings-gear', 'Open model configuration', 'zoomy.openModelConfig'],
                ['notebook', 'Open in Notebook Mode', 'zoomy.openInNotebook'],
                ['play', 'Run simulation', 'zoomy.run'],
            ]),
            this.group('Case', [
                ['arrow-down', 'Export case (.py)', 'zoomy.exportPy'],
                ['notebook', 'Export case (.ipynb)', 'zoomy.exportIpynb'],
                ['arrow-up', 'Import case…', 'zoomy.importCase'],
            ]),
            this.group('Project', [
                ['save', 'Save project', 'zoomy.saveProject'],
                ['folder-opened', 'Load project', 'zoomy.loadProject'],
            ]),
            this.group('Backend', [
                ['server', 'Connect backend…', 'zoomy.connectBackend'],
            ]));
    }
}
