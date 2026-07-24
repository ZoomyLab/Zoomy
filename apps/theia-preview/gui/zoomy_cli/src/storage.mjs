/**
 * Storage abstraction for the CLI.
 *
 * Two flavours, same interface (read + write):
 *  - FetchStorage (browser): HTTP GET against the page's origin for
 *    reads; an injected `overlay` (IdbStorage in the browser) handles
 *    reads under `cards/sessions/` and all writes/deletes/lists. The
 *    static origin never receives writes.
 *  - FsStorage (Node): readFileSync / writeFileSync / rmSync under a
 *    local root.
 *
 * Every method returns a Promise for symmetry; concrete classes resolve
 * synchronously where they can.
 *
 * Write surface (added when the project grew user-authored cards):
 *   writeJson / writeText / writeBytes  — upsert at `path`.
 *   deletePath                           — recursive (file or folder).
 *   listDir                              — immediate child names.
 *   exists                               — bool.
 */

export class FetchStorage {
    /**
     * @param {object} [options]
     * @param {string} [options.baseUrl]  Base URL to resolve paths against
     *                                    (defaults to the page origin).
     * @param {object} [options.overlay]  Optional writable overlay
     *   (IdbStorage). When present, reads under `cards/sessions/` try
     *   the overlay first and fall back to fetch; writes/deletes/lists
     *   all go to the overlay (fetch origin is read-only).
     */
    constructor(options) {
        options = options || {};
        this.baseUrl = options.baseUrl || "";
        this.overlay = options.overlay || null;
    }

    _url(path) {
        if (/^https?:\/\//.test(path)) return path;
        // Relative paths resolve against the document base.
        return this.baseUrl + path.replace(/^\//, "");
    }

    /**
     * Overlay-first applies under `cards/sessions/` (per-session user
     * cards) and `catalog/` (the writable card-catalog overlay). Both are
     * mutable state that only ever lives in the overlay; keeping the match
     * narrow avoids shadowing the shipped read-only registry (the
     * `cards/<dir>/default.json` files, `snippets/`) with stale IndexedDB
     * entries.
     */
    _overlayFirst(path) {
        return !!this.overlay && (/^cards\/sessions\//.test(path) || /^catalog\//.test(path));
    }

    async readJson(path) {
        if (this._overlayFirst(path)) {
            const hit = await this.overlay.tryReadJson(path);
            if (hit !== null) return hit;
        }
        const r = await fetch(this._url(path));
        if (!r.ok) throw new Error("storage: HTTP " + r.status + " at " + path);
        return await r.json();
    }

    async readText(path) {
        if (this._overlayFirst(path)) {
            const hit = await this.overlay.tryReadText(path);
            if (hit !== null) return hit;
        }
        const r = await fetch(this._url(path));
        if (!r.ok) throw new Error("storage: HTTP " + r.status + " at " + path);
        return await r.text();
    }

    async readBytes(path) {
        if (this._overlayFirst(path) && this.overlay.tryReadBytes) {
            const hit = await this.overlay.tryReadBytes(path);
            if (hit) return hit;
        }
        const r = await fetch(this._url(path));
        if (!r.ok) throw new Error("storage: HTTP " + r.status + " at " + path);
        return await r.arrayBuffer();
    }

    async tryReadJson(path)  { try { return await this.readJson(path); }  catch (e) { return null; } }
    async tryReadText(path)  { try { return await this.readText(path); }  catch (e) { return null; } }
    async tryReadBytes(path) { try { return await this.readBytes(path); } catch (e) { return null; } }

    // ------------------------------------------------------------------
    // Writes. The static origin can't accept POST/PUT, so every mutation
    // goes to the overlay. Without an overlay these throw so callers see
    // the problem early.
    // ------------------------------------------------------------------

    _requireOverlay(method) {
        if (!this.overlay) {
            throw new Error("FetchStorage." + method + ": no writable overlay");
        }
    }

    async writeJson(path, obj)    { this._requireOverlay("writeJson");  return this.overlay.writeJson(path, obj); }
    async writeText(path, text)   { this._requireOverlay("writeText");  return this.overlay.writeText(path, text); }
    async writeBytes(path, bytes) { this._requireOverlay("writeBytes"); return this.overlay.writeBytes(path, bytes); }
    async deletePath(path)        { this._requireOverlay("deletePath"); return this.overlay.deletePath(path); }

    async listDir(path) {
        if (!this.overlay) return [];
        return await this.overlay.listDir(path);
    }

    /**
     * Recursive enumeration — every full path under `prefix`. The
     * static origin has no way to enumerate itself, so this reports
     * only what lives in the overlay. Used by the ZIP save path to
     * collect user-authored cards for export.
     */
    async allPaths(prefix) {
        if (!this.overlay) return [];
        return await this.overlay.allPaths(prefix);
    }

    async exists(path) {
        if (this.overlay && await this.overlay.exists(path)) return true;
        try {
            const r = await fetch(this._url(path), { method: "HEAD" });
            return r.ok;
        } catch (e) { return false; }
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

    async tryReadJson(path)  { try { return await this.readJson(path); }  catch (e) { return null; } }
    async tryReadText(path)  { try { return await this.readText(path); }  catch (e) { return null; } }
    async tryReadBytes(path) { try { return await this.readBytes(path); } catch (e) { return null; } }

    // ------------------------------------------------------------------
    // Writes. Parent dirs are created on demand so callers don't have to
    // mkdir before each write.
    // ------------------------------------------------------------------

    async writeJson(path, obj) {
        await this.writeText(path, JSON.stringify(obj, null, 2) + "\n");
    }

    async writeText(path, text) {
        const abs = this._resolve(path);
        this._fs.mkdirSync(this._path.dirname(abs), { recursive: true });
        this._fs.writeFileSync(abs, text, "utf8");
    }

    async writeBytes(path, bytes) {
        const abs = this._resolve(path);
        this._fs.mkdirSync(this._path.dirname(abs), { recursive: true });
        let buf;
        if (bytes instanceof Uint8Array) {
            buf = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        } else if (bytes instanceof ArrayBuffer) {
            buf = Buffer.from(bytes);
        } else {
            buf = Buffer.from(bytes);
        }
        this._fs.writeFileSync(abs, buf);
    }

    async deletePath(path) {
        const abs = this._resolve(path);
        if (!this._fs.existsSync(abs)) return false;
        this._fs.rmSync(abs, { recursive: true, force: true });
        return true;
    }

    async listDir(path) {
        const abs = this._resolve(path);
        if (!this._fs.existsSync(abs)) return [];
        const stat = this._fs.statSync(abs);
        if (!stat.isDirectory()) return [];
        return this._fs.readdirSync(abs);
    }

    /**
     * Recursive walk — every file path under `prefix`, returned with
     * the same leading prefix the caller passed in so paths round-trip
     * cleanly through writeJson / writeBytes.
     */
    async allPaths(prefix) {
        const abs = this._resolve(prefix);
        if (!this._fs.existsSync(abs)) return [];
        const out = [];
        const rootPrefix = prefix.replace(/\/$/, "");
        const walk = (dir, rel) => {
            for (const entry of this._fs.readdirSync(dir, { withFileTypes: true })) {
                const absChild = this._path.join(dir, entry.name);
                const relChild = rel ? (rel + "/" + entry.name) : entry.name;
                if (entry.isDirectory()) walk(absChild, relChild);
                else out.push(rootPrefix + "/" + relChild);
            }
        };
        walk(abs, "");
        return out;
    }

    async exists(path) {
        return this._fs.existsSync(this._resolve(path));
    }
}
