/**
 * Storage abstraction for the CLI.
 *
 * Two flavours, same interface:
 *  - FetchStorage (browser): HTTP GET against the page's origin.
 *  - FsStorage (Node):       readFileSync from a local root.
 *
 * Every `read*` method returns a Promise for symmetry; concrete
 * classes resolve synchronously where they can.
 *
 * The GUI only needs the fetch-backed one. Node consumers (tests,
 * shell) can reach for FsStorage directly.
 */

export class FetchStorage {
    /**
     * @param {object} [options]
     * @param {string} [options.baseUrl]  Base URL to resolve paths against
     *                                    (defaults to the page origin).
     */
    constructor(options) {
        options = options || {};
        this.baseUrl = options.baseUrl || "";
    }

    _url(path) {
        if (/^https?:\/\//.test(path)) return path;
        // Relative paths resolve against the document base.
        return this.baseUrl + path.replace(/^\//, "");
    }

    async readJson(path) {
        const r = await fetch(this._url(path));
        if (!r.ok) throw new Error("storage: HTTP " + r.status + " at " + path);
        return await r.json();
    }

    async readText(path) {
        const r = await fetch(this._url(path));
        if (!r.ok) throw new Error("storage: HTTP " + r.status + " at " + path);
        return await r.text();
    }

    async readBytes(path) {
        const r = await fetch(this._url(path));
        if (!r.ok) throw new Error("storage: HTTP " + r.status + " at " + path);
        return await r.arrayBuffer();
    }

    async tryReadJson(path) {
        try { return await this.readJson(path); } catch (e) { return null; }
    }

    async tryReadText(path) {
        try { return await this.readText(path); } catch (e) { return null; }
    }
}

/** Node-side storage. Only imported under the node.mjs entry. */
export class FsStorage {
    constructor(options) {
        options = options || {};
        this.root = options.root || ".";
        this._fs = options.fs;           // injected to keep this file ESM-pure
        this._path = options.path;
    }

    _resolve(p) {
        return this._path.isAbsolute(p) ? p : this._path.join(this.root, p);
    }

    async readJson(path) {
        const text = await this.readText(path);
        return JSON.parse(text);
    }

    async readText(path) {
        return this._fs.readFileSync(this._resolve(path), "utf8");
    }

    async readBytes(path) {
        const buf = this._fs.readFileSync(this._resolve(path));
        return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    }

    async tryReadJson(path) {
        try { return await this.readJson(path); } catch (e) { return null; }
    }

    async tryReadText(path) {
        try { return await this.readText(path); } catch (e) { return null; }
    }
}
