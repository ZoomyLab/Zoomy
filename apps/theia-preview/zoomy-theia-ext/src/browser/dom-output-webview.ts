import React from '@theia/core/shared/react';
import * as ReactDOMNS from '@theia/core/shared/react-dom';
const createPortal = (ReactDOMNS as any).createPortal as (node: React.ReactNode, container: Element, key?: string) => React.ReactPortal;
import { Emitter, Event } from '@theia/core';
import { CellOutputWebview, OutputRenderEvent } from '@theia/notebook/lib/browser/renderers/cell-output-webview';

/** DOM (no-iframe) notebook output surface for browser-only Theia, where the
 *  default iframe webview factory is unbound. Instead of one overlay stacked at
 *  the top of the notebook, each code cell's outputs are portalled INTO that
 *  cell's own `.theia-notebook-cell-content`, so they flow directly under the
 *  cell (correct per-cell layout, no absolute positioning). Renders text
 *  streams, rich html/markdown/latex, matplotlib PNGs and Python errors. */
export class DomOutputWebview implements CellOutputWebview {
    readonly id = 'zoomy-dom-output';
    protected notebook: any;
    protected readonly emitter = new Emitter<OutputRenderEvent>();
    readonly onDidRenderOutput: Event<OutputRenderEvent> = this.emitter.event;
    protected forceUpdate: () => void = () => { };
    /** cellHandle -> the mount node we portal that cell's outputs into. */
    readonly nodes = new Map<number, HTMLElement>();

    init(notebook: any): void {
        this.notebook = notebook;
        notebook.onDidChangeContent?.(() => this.forceUpdate());
    }
    render(): React.ReactNode { return React.createElement(OutputsHost, { webview: this }); }
    getNotebook(): any { return this.notebook; }
    bindUpdate(fn: () => void): void { this.forceUpdate = fn; }

    /** get-or-create the detached mount node for a cell. */
    nodeFor(handle: number): HTMLElement {
        let n = this.nodes.get(handle);
        if (!n) { n = document.createElement('div'); n.className = 'zoomy-cell-output'; n.style.padding = '4px 0'; this.nodes.set(handle, n); }
        return n;
    }
    /** attach each mount node into its cell's content div; drop stale ones. */
    reattach(handles: number[]): void {
        const live = new Set(handles);
        for (const [h, node] of this.nodes) {
            if (!live.has(h)) { node.remove(); this.nodes.delete(h); continue; }
            const cell = document.querySelector(`.theia-notebook-cell[data-cell-handle="${h}"] .theia-notebook-cell-content`);
            if (cell && node.parentElement !== cell) { cell.appendChild(node); }
        }
    }

    setCellHeight(): void { }
    cellsChanged(): void { this.forceUpdate(); }
    requestOutputPresentationUpdate(): void { this.forceUpdate(); }
    isAttached(): boolean { return true; }
    dispose(): void { for (const n of this.nodes.values()) { n.remove(); } this.nodes.clear(); this.emitter.dispose(); }
}

function bytesToBase64(bytes: Uint8Array): string {
    let s = '';
    for (let i = 0; i < bytes.length; i++) { s += String.fromCharCode(bytes[i]); }
    return btoa(s);
}
function text(item: any): string {
    try { return item.data?.toString?.() ?? ''; } catch { return ''; }
}

function renderItem(item: any, key: string): React.ReactNode {
    const mime: string = item.mime || 'text/plain';
    if (mime.startsWith('image/')) {
        const bytes: Uint8Array = item.data?.buffer ?? new Uint8Array();
        return React.createElement('img', { key, src: 'data:' + mime + ';base64,' + bytesToBase64(bytes), style: { maxWidth: '100%', display: 'block', margin: '4px 0' } });
    }
    if (mime === 'text/html') {
        return React.createElement('div', { key, className: 'zoomy-html-output', dangerouslySetInnerHTML: { __html: text(item) } });
    }
    const isErr = mime.includes('error');
    return React.createElement('pre', { key, style: { margin: '2px 0', whiteSpace: 'pre-wrap', fontFamily: 'var(--theia-code-font-family, monospace)', fontSize: 12, color: isErr ? 'var(--theia-errorForeground, #f66)' : undefined } }, text(item));
}

const CellOutputs: React.FC<{ cell: any }> = ({ cell }) =>
    React.createElement(React.Fragment, null,
        (cell.outputs || []).flatMap((o: any, oi: number) =>
            (o.outputs || []).map((item: any, ii: number) => renderItem(item, oi + '-' + ii))));

const OutputsHost: React.FC<{ webview: DomOutputWebview }> = ({ webview }) => {
    const [, setTick] = React.useState(0);
    React.useEffect(() => { webview.bindUpdate(() => setTick(t => t + 1)); }, [webview]);
    const cells = (webview.getNotebook()?.cells || []).filter((c: any) => c.cellKind === 2 && c.outputs?.length);
    const handles = cells.map((c: any) => c.handle);
    React.useLayoutEffect(() => { webview.reattach(handles); });
    return React.createElement(React.Fragment, null,
        cells.map((c: any) => createPortal(React.createElement(CellOutputs, { cell: c }), webview.nodeFor(c.handle), 'zc-' + c.handle)));
};
