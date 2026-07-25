import React from '@theia/core/shared/react';
import { injectable, inject, postConstruct } from '@theia/core/shared/inversify';
import { ReactWidget } from '@theia/core/lib/browser/widgets/react-widget';
import { WidgetManager } from '@theia/core/lib/browser';
import { DisposableCollection } from '@theia/core';
import { ZoomyModelConfigWidget } from './model-config-widget';

/** The right-hand "Zoomy Parameters" panel (native right-area view, Zoomy icon).
 *  It is a thin host: the model-config widget owns the schema + edited values and
 *  renders the active card's parameter form; this widget just displays it and
 *  re-renders whenever the params target or a value changes. */
@injectable()
export class ZoomyParamsWidget extends ReactWidget {
    static readonly ID = 'zoomy-params-view';
    @inject(WidgetManager) protected readonly widgetManager: WidgetManager;
    protected config: ZoomyModelConfigWidget | undefined;
    protected readonly toDispose = new DisposableCollection();

    @postConstruct()
    protected init(): void {
        this.id = ZoomyParamsWidget.ID;
        this.title.label = 'Parameters';
        this.title.caption = 'Zoomy — parameters';
        // Zoomy-branded, parameter-flavoured icon for the right activity bar.
        this.title.iconClass = 'codicon codicon-settings';
        this.title.closable = true;
        this.addClass('zoomy-params-widget');
        this.node.style.overflow = 'auto';
        this.bindConfig();
        this.update();
    }

    /** Subscribe to the config widget's onParamsChanged so we re-render in step. */
    protected async bindConfig(): Promise<void> {
        try {
            const w = (await this.widgetManager.getOrCreateWidget(ZoomyModelConfigWidget.ID)) as ZoomyModelConfigWidget;
            this.config = w;
            this.toDispose.push(w.onParamsChanged(() => this.update()));
            this.update();
        } catch { /* config not ready yet — will show the placeholder */ }
    }

    override dispose(): void { this.toDispose.dispose(); super.dispose(); }

    protected render(): React.ReactNode {
        const h = React.createElement;
        if (!this.config) { return h('div', { style: { padding: 14, color: 'var(--theia-descriptionForeground)', fontSize: 13 } }, 'Loading…'); }
        return this.config.renderActiveParams();
    }
}
