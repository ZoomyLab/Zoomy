import React from '@theia/core/shared/react';
import { injectable, inject, postConstruct } from '@theia/core/shared/inversify';
import { ReactWidget } from '@theia/core/lib/browser/widgets/react-widget';
import { CommandRegistry } from '@theia/core';

/** The Zoomy activity-bar view (left panel): the "burger" menu that holds the
 *  case/project actions in a proper native slot, instead of a self-coded toolbar
 *  inside the model-config editor. Each item just executes a command. */
@injectable()
export class ZoomyViewWidget extends ReactWidget {
    static readonly ID = 'zoomy-view';
    @inject(CommandRegistry) protected readonly commands: CommandRegistry;

    @postConstruct()
    protected init(): void {
        this.id = ZoomyViewWidget.ID;
        this.title.label = 'Zoomy';
        this.title.caption = 'Zoomy';
        this.title.iconClass = 'codicon codicon-beaker';
        this.title.closable = true;
        this.addClass('zoomy-view-widget');
        this.node.style.overflow = 'auto';
        this.update();
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

    protected render(): React.ReactNode {
        const h = React.createElement;
        return h('div', { style: { padding: '8px 4px', fontFamily: 'var(--theia-font-family)' } },
            this.group('Configuration', [
                ['settings-gear', 'Open model configuration', 'zoomy.openModelConfig'],
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
