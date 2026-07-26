import React from '@theia/core/shared/react';
import { injectable, postConstruct } from '@theia/core/shared/inversify';
import { ReactWidget } from '@theia/core/lib/browser/widgets/react-widget';
import { DisposableCollection, Disposable } from '@theia/core';
import { onSimOutput, SimOutputEvent } from './zoomy-cli-loader';

interface Line { level: string; text: string; }

/** The bottom-panel "Simulation" view: a console that streams the assembly Run's
 *  progress + stdout. Auto-revealed on Run (by the frontend module). */
@injectable()
export class ZoomySimOutputWidget extends ReactWidget {
    static readonly ID = 'zoomy-sim-output';
    protected lines: Line[] = [];
    protected readonly toDispose = new DisposableCollection();

    @postConstruct()
    protected init(): void {
        this.id = ZoomySimOutputWidget.ID;
        this.title.label = 'Log';
        this.title.caption = 'Zoomy — run output + log';
        this.title.iconClass = 'codicon codicon-pulse';
        this.title.closable = true;
        this.addClass('zoomy-sim-output-widget');
        this.node.style.overflow = 'auto';
        this.toDispose.push(Disposable.create(onSimOutput((e: SimOutputEvent) => this.onEvent(e))));
        this.update();
    }

    override dispose(): void { this.toDispose.dispose(); super.dispose(); }

    protected onEvent(e: SimOutputEvent): void {
        if (e.kind === 'clear') { this.lines = []; }
        else if (e.kind === 'line' && e.text != null) { this.lines.push({ level: e.level || 'info', text: e.text }); }
        this.update();
        // Keep the newest output in view.
        setTimeout(() => { this.node.scrollTop = this.node.scrollHeight; }, 0);
    }

    protected render(): React.ReactNode {
        const h = React.createElement;
        const color = (lvl: string): string => lvl === 'error' ? 'var(--theia-errorForeground)'
            : lvl === 'ok' ? 'var(--theia-successForeground, #3fb950)'
            : lvl === 'info' ? 'var(--theia-descriptionForeground)' : 'var(--theia-foreground)';
        const iconBtn: React.CSSProperties = { cursor: 'pointer', border: 'none', background: 'transparent', color: 'var(--theia-descriptionForeground)', padding: 2 };
        const bar = h('div', { style: { position: 'sticky', top: 0, display: 'flex', alignItems: 'center', gap: 6, padding: '4px 8px', background: 'var(--theia-editor-background, var(--theia-editorWidget-background))', borderBottom: '1px solid var(--theia-panel-border)' } },
            h('span', { className: 'codicon codicon-pulse', style: { color: 'var(--theia-descriptionForeground)' } }),
            h('span', { style: { flex: 1, fontSize: 11, textTransform: 'uppercase', letterSpacing: '.05em', color: 'var(--theia-descriptionForeground)' } }, 'Log'),
            h('button', { title: 'Clear', style: iconBtn, onClick: () => { this.lines = []; this.update(); } }, h('span', { className: 'codicon codicon-clear-all' })),
            h('button', { title: 'Close panel', style: iconBtn, onClick: () => this.close() }, h('span', { className: 'codicon codicon-close' })));
        const bodyStyle: React.CSSProperties = !this.lines.length
            ? { padding: 12, fontSize: 12.5, color: 'var(--theia-descriptionForeground)', fontFamily: 'var(--theia-font-family)' }
            : { padding: '8px 12px', fontFamily: 'var(--theia-code-font-family, monospace)', fontSize: 12, lineHeight: 1.5 };
        const body = !this.lines.length
            ? h('div', { style: bodyStyle }, 'Run a simulation to see its output here.')
            : h('div', { style: bodyStyle }, this.lines.map((l, i) => h('div', { key: i, style: { whiteSpace: 'pre-wrap', color: color(l.level) } }, l.text)));
        return h('div', null, bar, body);
    }
}
