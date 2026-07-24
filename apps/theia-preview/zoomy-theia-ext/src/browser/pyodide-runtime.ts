/* Main-thread client for the Pyodide Web Worker. Spawns the worker from a Blob
 * URL (so there is no separate asset to serve), warms it immediately, and routes
 * request/response by id. Exposes cell execution and jedi autocomplete. */
import { WORKER_SOURCE } from './pyodide-worker-source';

export interface CellStreamOutput { type: 'stream'; text: string; }
export interface CellDataOutput { type: 'data'; mime: string; value: string; }
export interface CellErrorOutput { type: 'error'; ename: string; evalue: string; }
export type CellOut = CellStreamOutput | CellDataOutput | CellErrorOutput;

export interface CompletionItem { name: string; type: string; signature: string; docstring: string; module: string; }
export interface CompletionResult { completions: CompletionItem[]; error?: string; }

type Pending = { resolve: (v: any) => void; reject: (e: any) => void };

let singleton: PyodideClient | undefined;

/** One worker per frontend; constructed lazily and warmed on first touch. */
export function getPyodideClient(log: (m: string) => void): PyodideClient {
    if (!singleton) { singleton = new PyodideClient(log); }
    return singleton;
}

export class PyodideClient {
    protected worker: Worker;
    protected seq = 0;
    protected readonly pending = new Map<number, Pending>();
    protected resolveReady!: () => void;
    /** resolves when Pyodide + zoomy-core are up (kernel usable). */
    readonly ready: Promise<void>;
    backgroundReady = false;

    constructor(protected readonly log: (m: string) => void) {
        const blob = new Blob([WORKER_SOURCE], { type: 'application/javascript' });
        this.worker = new Worker(URL.createObjectURL(blob));
        this.worker.onmessage = e => this.onMessage(e.data);
        this.worker.onerror = e => this.log('worker error: ' + (e.message || e.toString()));
        this.ready = new Promise<void>(res => { this.resolveReady = res; });
    }

    protected onMessage(m: any): void {
        switch (m?.type) {
            case 'log': this.log(m.msg); return;
            case 'ready': this.resolveReady(); if (m.id != null) { this.settle(m.id, undefined); } return;
            case 'background_ready': this.backgroundReady = true; this.log('kernel fully warm (autocomplete + plotting ready)'); return;
            case 'result': this.settle(m.id, m.outputs !== undefined ? m.outputs : m.data); return;
            case 'error': this.fail(m.id, m.error); return;
        }
    }
    protected settle(id: number, val: any): void { const p = this.pending.get(id); if (p) { this.pending.delete(id); p.resolve(val); } }
    protected fail(id: number, err: any): void { const p = this.pending.get(id); if (p) { this.pending.delete(id); p.reject(new Error(String(err))); } }
    protected call(cmd: string, extra: any): Promise<any> {
        const id = ++this.seq;
        return new Promise((resolve, reject) => { this.pending.set(id, { resolve, reject }); this.worker.postMessage({ cmd, id, ...extra }); });
    }

    warm(): Promise<void> { return this.call('warm', {}); }
    runCell(code: string): Promise<CellOut[]> { return this.call('run', { code }) as Promise<CellOut[]>; }
    complete(code: string, row: number, col: number): Promise<CompletionResult> { return this.call('complete', { code, row, col }) as Promise<CompletionResult>; }
}
