import { Emitter, Event, URI } from '@theia/core';
import { BinaryBuffer } from '@theia/core/lib/common/buffer';
import { NotebookKernel, NotebookKernelChangeEvent } from '@theia/notebook/lib/browser/service/notebook-kernel-service';
import { NotebookService } from '@theia/notebook/lib/browser';
import { NotebookExecutionStateService } from '@theia/notebook/lib/browser/service/notebook-execution-state-service';
import { CellExecutionUpdateType } from '@theia/notebook/lib/common';
import { PyodideClient, CellOut } from './pyodide-runtime';

/** A Jupyter-style output item carries binary data; images are base64 -> bytes. */
function toItem(mime: string, value: string): { mime: string; data: BinaryBuffer } {
    if (mime.startsWith('image/')) {
        const bin = atob(value);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) { bytes[i] = bin.charCodeAt(i); }
        return { mime, data: BinaryBuffer.wrap(bytes) };
    }
    return { mime, data: BinaryBuffer.fromString(value) };
}

// Collapse a cell's CellOut[] into notebook output items. Consecutive stream
// chunks merge into one text/plain block; data/error each become their own item.
function toOutputItems(outs: CellOut[]): Array<{ mime: string; data: BinaryBuffer }> {
    const items: Array<{ mime: string; data: BinaryBuffer }> = [];
    let stream = '';
    const flush = () => { if (stream) { items.push(toItem('text/plain', stream)); stream = ''; } };
    for (const o of outs) {
        if (o.type === 'stream') { stream += o.text; }
        else if (o.type === 'data') { flush(); items.push(toItem(o.mime, o.value)); }
        else { flush(); items.push(toItem('application/vnd.code.notebook.error', o.ename + ': ' + o.evalue)); }
    }
    flush();
    return items;
}

export class PyodideKernel implements NotebookKernel {
    readonly id = 'zoomy-pyodide';
    readonly viewType = 'zoomy-notebook';
    readonly extensionId = 'zoomy';
    readonly localResourceRoot = new URI('file:///');
    readonly preloadUris: URI[] = [];
    readonly preloadProvides: string[] = [];
    readonly handle = 1;
    label = 'Zoomy Pyodide (in-browser)';
    description = 'Python in the browser via Pyodide — zoomy-core + zoomy-plotting';
    supportedLanguages = ['python'];
    implementsInterrupt = false;
    implementsExecutionOrder = true;
    protected readonly onDidChangeEmitter = new Emitter<NotebookKernelChangeEvent>();
    readonly onDidChange: Event<NotebookKernelChangeEvent> = this.onDidChangeEmitter.event;

    constructor(
        protected readonly notebookService: NotebookService,
        protected readonly execService: NotebookExecutionStateService,
        protected readonly client: PyodideClient,
        protected readonly log: (m: string) => void,
    ) { }

    async executeNotebookCellsRequest(uri: URI, cellHandles: number[]): Promise<void> {
        const model = this.notebookService.getNotebookEditorModel(uri);
        if (!model) { return; }
        let order = 1;
        for (const handle of cellHandles) {
            const cell = model.getCellByHandle(handle);
            if (!cell) { continue; }
            const exec = this.execService.getOrCreateCellExecution(uri, handle);
            exec.confirm();
            exec.update([{ editType: CellExecutionUpdateType.ExecutionState, executionOrder: order++, runStartTime: Date.now() }]);
            const outs = await this.client.runCell(cell.source);
            const items = toOutputItems(outs);
            const cellOutput = { outputId: 'o' + handle + '-' + Date.now(), outputs: items };
            exec.update([{ editType: CellExecutionUpdateType.Output, cellHandle: handle, outputs: [cellOutput], append: false }]);
            exec.complete({ runEndTime: Date.now(), lastRunSuccess: !outs.some(o => o.type === 'error') });
        }
    }

    async cancelNotebookCellExecution(): Promise<void> { }
}
