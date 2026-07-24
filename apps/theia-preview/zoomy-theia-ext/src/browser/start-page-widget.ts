import React from '@theia/core/shared/react';
import { injectable, inject, postConstruct } from '@theia/core/shared/inversify';
import { ReactWidget } from '@theia/core/lib/browser/widgets/react-widget';
import { CommandRegistry } from '@theia/core';

export const OPEN_EDITOR = 'zoomy.openEditor';
export const OPEN_NOTEBOOK = 'zoomy.openNotebook';
export const OPEN_MODELCONFIG = 'zoomy.openModelConfig';

/** The Baukasten "start page" — the single GUI surface that will host model
 *  configuration. In the prototype it is the landing view and the launch pad:
 *  one route into the Theia code editor, one into the native Pyodide notebook.
 *  Styled with Theia/VS-Code theme tokens so it matches Baukasten's look; the
 *  real baukasten-ui React components (already proven on /baukasten-preview/)
 *  drop in here next. */
@injectable()
export class ZoomyStartWidget extends ReactWidget {
    static readonly ID = 'zoomy-start';
    @inject(CommandRegistry) protected readonly commands: CommandRegistry;

    @postConstruct()
    protected init(): void {
        this.id = ZoomyStartWidget.ID;
        this.title.label = 'Zoomy';
        this.title.caption = 'Zoomy — start';
        this.title.iconClass = 'codicon codicon-home';
        this.title.closable = true;
        this.addClass('zoomy-start-widget');
        this.node.style.overflow = 'auto';
        this.update();
    }

    protected go(cmd: string): void { this.commands.executeCommand(cmd); }

    protected render(): React.ReactNode {
        const h = React.createElement;
        const page: React.CSSProperties = { maxWidth: 760, margin: '0 auto', padding: '48px 28px', color: 'var(--theia-foreground)', fontFamily: 'var(--theia-font-family)' };
        const badge: React.CSSProperties = { display: 'inline-block', fontSize: 11, letterSpacing: '.08em', textTransform: 'uppercase', padding: '2px 8px', borderRadius: 4, background: 'var(--theia-badge-background)', color: 'var(--theia-badge-foreground)' };
        const sub: React.CSSProperties = { color: 'var(--theia-descriptionForeground)', fontSize: 14, lineHeight: 1.6, marginTop: 8 };
        const card: React.CSSProperties = { border: '1px solid var(--theia-editorWidget-border, var(--theia-panel-border))', borderRadius: 8, padding: 18, marginTop: 22, background: 'var(--theia-editorWidget-background)' };
        const btnRow: React.CSSProperties = { display: 'flex', gap: 12, marginTop: 26, flexWrap: 'wrap' };
        const primaryBtn: React.CSSProperties = { cursor: 'pointer', border: 'none', borderRadius: 6, padding: '12px 18px', fontSize: 14, fontWeight: 600, background: 'var(--theia-button-background)', color: 'var(--theia-button-foreground)' };
        const secondaryBtn: React.CSSProperties = { ...primaryBtn, background: 'var(--theia-button-secondaryBackground)', color: 'var(--theia-button-secondaryForeground)' };
        const iconOf = (n: string): React.ReactNode => h('span', { className: 'codicon codicon-' + n, style: { marginRight: 8, verticalAlign: 'middle' } });

        return h('div', { style: page },
            h('span', { style: badge }, 'prototype'),
            h('h1', { style: { fontSize: 34, margin: '14px 0 0', fontWeight: 700 } }, 'Zoomy'),
            h('div', { style: sub },
                'One GUI, written once in ', h('strong', null, 'Baukasten'), ', running on backend-less ',
                h('strong', null, 'Theia'), ' — the same code targets web, a VS Code / Theia extension, and an Electron app.'),
            h('div', { style: card },
                h('div', { style: { fontSize: 13, fontWeight: 600, color: 'var(--theia-descriptionForeground)' } },
                    iconOf('settings-gear'), 'Model configuration'),
                h('div', { style: { ...sub, marginTop: 10 } },
                    'Pick a model, mesh, solver and visualization from the real Zoomy card catalog and run it on the in-browser kernel. Or drop into a notebook or the code editor.')),
            h('div', { style: btnRow },
                h('button', { style: primaryBtn, onClick: () => this.go(OPEN_MODELCONFIG) },
                    iconOf('settings-gear'), 'Open model configuration'),
                h('button', { style: secondaryBtn, onClick: () => this.go(OPEN_NOTEBOOK) },
                    iconOf('notebook'), 'Open Pyodide notebook'),
                h('button', { style: secondaryBtn, onClick: () => this.go(OPEN_EDITOR) },
                    iconOf('code'), 'Open code editor')),
            h('div', { style: { ...sub, marginTop: 30, fontSize: 12 } },
                'backend-less Theia · Baukasten UI · native notebook on an in-browser Pyodide kernel (zoomy-core + zoomy-plotting). ',
                'Use the ', h('span', { className: 'codicon codicon-home', style: { verticalAlign: 'middle' } }),
                ' “Zoomy start” item in the status bar to come back here from anywhere.'));
    }
}
