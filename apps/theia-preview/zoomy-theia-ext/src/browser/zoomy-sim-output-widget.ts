import React from '@theia/core/shared/react';
import { injectable, postConstruct } from '@theia/core/shared/inversify';
import { ReactWidget } from '@theia/core/lib/browser/widgets/react-widget';
import { onSimOutput, SimOutputEvent } from './zoomy-cli-loader';

interface Line { level: string; text: string; }

/** The bottom-panel "Simulation" view: a console that streams the assembly Run's
 *  progress + stdout. Auto-revealed on Run (by the frontend module). */
@injectable()
export class ZoomySimOutputWidget extends ReactWidget {
    static readonly ID = 'zoomy-sim-output';
    protected lines: Line[] = [];

    @postConstruct()
    protected init(): void {
        this.id = ZoomySimOutputWidget.ID;
        this.title.label = 'Simulation';
        this.title.caption = 'Zoomy — simulation output';
        this.title.iconClass = 'codicon codicon-pulse';
        this.title.closable = true;
        this.addClass('zoomy-sim-output-widget');
        this.node.style.overflow = 'auto';
        onSimOutput((e: SimOutputEvent) => this.onEvent(e));
        this.update();
    }

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
        if (!this.lines.length) {
            return h('div', { style: { padding: 12, fontSize: 12.5, color: 'var(--theia-descriptionForeground)', fontFamily: 'var(--theia-font-family)' } },
                'Run a simulation to see its output here.');
        }
        return h('div', { style: { padding: '8px 12px', fontFamily: 'var(--theia-code-font-family, monospace)', fontSize: 12, lineHeight: 1.5 } },
            this.lines.map((l, i) => h('div', { key: i, style: { whiteSpace: 'pre-wrap', color: color(l.level) } }, l.text)));
    }
}
