import { injectable } from '@theia/core/shared/inversify';
import { Emitter, Event, URI, Disposable } from '@theia/core';
import {
    FileType, FileSystemProviderCapabilities, FileSystemProviderErrorCode,
    createFileSystemProviderError, FileChangeType,
} from '@theia/filesystem/lib/common/files';

/** A reliable in-browser FileSystemProvider (in-memory tree, persisted to
 *  IndexedDB) that replaces Theia's OPFS provider. OPFS fails to initialize in a
 *  blob worker on some browsers ("Failed to initialize OPFS"), which breaks the
 *  whole FileService — and with it the workspace + our case-as-source-of-truth
 *  model. This provider works everywhere and keeps the cases persistent. */
interface FsNode { type: FileType; content?: Uint8Array; mtime: number; ctime: number; }

function now(): number { return Date.now(); }
function b64encode(b: Uint8Array): string { let s = ''; for (let i = 0; i < b.length; i++) { s += String.fromCharCode(b[i]); } return btoa(s); }
function b64decode(s: string): Uint8Array { const bin = atob(s); const b = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) { b[i] = bin.charCodeAt(i); } return b; }

@injectable()
export class MemoryFileSystemProvider {
    readonly capabilities = FileSystemProviderCapabilities.FileReadWrite | FileSystemProviderCapabilities.PathCaseSensitive;
    readonly onDidChangeCapabilities = Event.None;
    readonly onFileWatchError = Event.None;
    protected readonly onDidChangeFileEmitter = new Emitter<any>();
    readonly onDidChangeFile = this.onDidChangeFileEmitter.event;

    protected readonly nodes = new Map<string, FsNode>();
    protected readonly ready: Promise<void>;
    protected saveTimer: any;

    constructor() {
        this.nodes.set('/', { type: FileType.Directory, mtime: now(), ctime: now() });
        this.ready = this.load().then(() => {
            // Seed the project root so a workspace can open on it.
            this.ensureDir('/zoomy'); this.ensureDir('/zoomy/cases');
        });
    }

    protected p(resource: URI): string { const s = resource.path.toString(); return s === '' ? '/' : s; }
    protected parent(path: string): string { const i = path.lastIndexOf('/'); return i <= 0 ? '/' : path.slice(0, i); }
    protected base(path: string): string { return path.slice(path.lastIndexOf('/') + 1); }
    protected ensureDir(path: string): void {
        if (this.nodes.has(path)) { return; }
        if (path !== '/') { this.ensureDir(this.parent(path)); }
        this.nodes.set(path, { type: FileType.Directory, mtime: now(), ctime: now() });
    }

    watch(): Disposable { return Disposable.create(() => { }); }

    async stat(resource: URI): Promise<any> {
        await this.ready;
        const n = this.nodes.get(this.p(resource));
        if (!n) { throw createFileSystemProviderError('Not found: ' + this.p(resource), FileSystemProviderErrorCode.FileNotFound); }
        return { type: n.type, ctime: n.ctime, mtime: n.mtime, size: n.content?.byteLength || 0 };
    }
    async mkdir(resource: URI): Promise<void> {
        await this.ready;
        this.ensureDir(this.p(resource));
        this.onDidChangeFileEmitter.fire([{ resource, type: FileChangeType.ADDED }]);
        this.scheduleSave();
    }
    async readdir(resource: URI): Promise<[string, FileType][]> {
        await this.ready;
        const dir = this.p(resource);
        const n = this.nodes.get(dir);
        if (!n) { throw createFileSystemProviderError('Not found: ' + dir, FileSystemProviderErrorCode.FileNotFound); }
        if (n.type !== FileType.Directory) { throw createFileSystemProviderError('Not a directory: ' + dir, FileSystemProviderErrorCode.FileNotADirectory); }
        const prefix = dir === '/' ? '/' : dir + '/';
        const out: [string, FileType][] = [];
        for (const [path, node] of this.nodes) {
            if (path === dir) { continue; }
            if (path.startsWith(prefix) && path.slice(prefix.length).indexOf('/') === -1) { out.push([this.base(path), node.type]); }
        }
        return out;
    }
    async readFile(resource: URI): Promise<Uint8Array> {
        await this.ready;
        const n = this.nodes.get(this.p(resource));
        if (!n) { throw createFileSystemProviderError('Not found: ' + this.p(resource), FileSystemProviderErrorCode.FileNotFound); }
        if (n.type !== FileType.File) { throw createFileSystemProviderError('Is a directory', FileSystemProviderErrorCode.FileIsADirectory); }
        return n.content ? n.content.slice() : new Uint8Array();
    }
    async writeFile(resource: URI, content: Uint8Array, opts: { create: boolean; overwrite: boolean }): Promise<void> {
        await this.ready;
        const path = this.p(resource);
        const existing = this.nodes.get(path);
        if (existing && existing.type === FileType.Directory) { throw createFileSystemProviderError('Is a directory', FileSystemProviderErrorCode.FileIsADirectory); }
        if (existing && !opts.overwrite) { throw createFileSystemProviderError('Exists', FileSystemProviderErrorCode.FileExists); }
        if (!existing && !opts.create) { throw createFileSystemProviderError('Not found', FileSystemProviderErrorCode.FileNotFound); }
        this.ensureDir(this.parent(path));
        this.nodes.set(path, { type: FileType.File, content: new Uint8Array(content), mtime: now(), ctime: existing?.ctime || now() });
        this.onDidChangeFileEmitter.fire([{ resource, type: existing ? FileChangeType.UPDATED : FileChangeType.ADDED }]);
        this.scheduleSave();
    }
    async delete(resource: URI, opts: { recursive: boolean }): Promise<void> {
        await this.ready;
        const path = this.p(resource);
        if (!this.nodes.has(path)) { return; }
        const prefix = path + '/';
        for (const key of Array.from(this.nodes.keys())) { if (key === path || (opts.recursive && key.startsWith(prefix))) { this.nodes.delete(key); } }
        this.onDidChangeFileEmitter.fire([{ resource, type: FileChangeType.DELETED }]);
        this.scheduleSave();
    }
    async rename(from: URI, to: URI, opts: { overwrite: boolean }): Promise<void> {
        await this.ready;
        const src = this.p(from); const dst = this.p(to);
        if (this.nodes.has(dst) && !opts.overwrite) { throw createFileSystemProviderError('Exists', FileSystemProviderErrorCode.FileExists); }
        const srcPrefix = src + '/';
        for (const key of Array.from(this.nodes.keys())) {
            if (key === src) { this.nodes.set(dst, this.nodes.get(key)!); this.nodes.delete(key); }
            else if (key.startsWith(srcPrefix)) { this.nodes.set(dst + key.slice(src.length), this.nodes.get(key)!); this.nodes.delete(key); }
        }
        this.ensureDir(this.parent(dst));
        this.onDidChangeFileEmitter.fire([{ resource: from, type: FileChangeType.DELETED }, { resource: to, type: FileChangeType.ADDED }]);
        this.scheduleSave();
    }
    async copy(from: URI, to: URI, opts: { overwrite: boolean }): Promise<void> {
        await this.ready;
        const src = this.p(from); const dst = this.p(to);
        const srcPrefix = src + '/';
        for (const key of Array.from(this.nodes.keys())) {
            const n = this.nodes.get(key)!;
            if (key === src) { this.nodes.set(dst, { ...n, content: n.content?.slice(), mtime: now(), ctime: now() }); }
            else if (key.startsWith(srcPrefix)) { this.nodes.set(dst + key.slice(src.length), { ...n, content: n.content?.slice(), mtime: now(), ctime: now() }); }
        }
        this.ensureDir(this.parent(dst));
        this.onDidChangeFileEmitter.fire([{ resource: to, type: FileChangeType.ADDED }]);
        this.scheduleSave();
    }

    // --- IndexedDB persistence (a single serialized tree record). ---
    protected idb(): Promise<IDBDatabase> {
        return new Promise((res, rej) => {
            const req = indexedDB.open('zoomy-fs', 1);
            req.onupgradeneeded = () => { req.result.createObjectStore('kv'); };
            req.onsuccess = () => res(req.result); req.onerror = () => rej(req.error);
        });
    }
    protected async load(): Promise<void> {
        try {
            const db = await this.idb();
            const data: any = await new Promise((res, rej) => { const t = db.transaction('kv', 'readonly').objectStore('kv').get('tree'); t.onsuccess = () => res(t.result); t.onerror = () => rej(t.error); });
            if (data && typeof data === 'object') {
                for (const [path, n] of Object.entries<any>(data)) {
                    this.nodes.set(path, { type: n.t, mtime: n.m, ctime: n.c, content: n.d != null ? b64decode(n.d) : undefined });
                }
            }
        } catch { /* first run / no IDB — start empty */ }
    }
    protected scheduleSave(): void { clearTimeout(this.saveTimer); this.saveTimer = setTimeout(() => this.save(), 300); }
    protected async save(): Promise<void> {
        try {
            const out: any = {};
            for (const [path, n] of this.nodes) { out[path] = { t: n.type, m: n.mtime, c: n.ctime, d: n.content ? b64encode(n.content) : null }; }
            const db = await this.idb();
            await new Promise((res, rej) => { const t = db.transaction('kv', 'readwrite').objectStore('kv').put(out, 'tree'); t.onsuccess = () => res(undefined); t.onerror = () => rej(t.error); });
        } catch { /* ignore persistence errors */ }
    }
}
