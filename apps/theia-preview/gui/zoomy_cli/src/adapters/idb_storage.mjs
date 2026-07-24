/**
 * IndexedDB-backed storage for the browser. Exposes the same method
 * surface as FsStorage so user_cards helpers (and the
 * FetchStorage.overlay slot) can treat it as a drop-in.
 *
 * Schema (v1):
 *   DB:     "zoomy_userdata"
 *   Store:  "files", keyPath = "path"
 *   Value:  { path, kind: "json" | "text" | "bytes",
 *             payload: object | string | Uint8Array,
 *             mtime: Number }
 *
 * IndexedDB has no directory concept — folders are implicit in the
 * slash-separated key. listDir / deletePath operate by prefix scan.
 * For user-scale data (hundreds of cards per session, at most a few
 * sessions per device) a full-keyset scan is cheap enough.
 */

const DB_NAME = "zoomy_userdata";
const DB_VERSION = 1;
const STORE = "files";

function _awaitRequest(req) {
    return new Promise((resolve, reject) => {
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error || new Error("IDB request failed"));
    });
}

export class IdbStorage {
    /**
     * @param {object} [options]
     * @param {IDBFactory} [options.indexedDB]   Override the factory (tests).
     * @param {string} [options.dbName]          Override the DB name (tests).
     */
    constructor(options) {
        options = options || {};
        this._indexedDB = options.indexedDB
            || (typeof indexedDB !== "undefined" ? indexedDB : null);
        if (!this._indexedDB) {
            throw new Error("IdbStorage: IndexedDB is not available in this runtime");
        }
        this._dbName = options.dbName || DB_NAME;
        this.__dbPromise = null;
    }

    async _db() {
        if (!this.__dbPromise) {
            const factory = this._indexedDB;
            this.__dbPromise = new Promise((resolve, reject) => {
                const req = factory.open(this._dbName, DB_VERSION);
                req.onupgradeneeded = () => {
                    const db = req.result;
                    if (!db.objectStoreNames.contains(STORE)) {
                        db.createObjectStore(STORE, { keyPath: "path" });
                    }
                };
                req.onsuccess = () => resolve(req.result);
                req.onerror = () => reject(req.error || new Error("IDB open failed"));
            });
        }
        return await this.__dbPromise;
    }

    async _store(mode) {
        const db = await this._db();
        return db.transaction([STORE], mode).objectStore(STORE);
    }

    async _get(path) {
        const store = await this._store("readonly");
        return await _awaitRequest(store.get(path));
    }

    async _put(path, kind, payload) {
        const store = await this._store("readwrite");
        return await _awaitRequest(store.put({ path, kind, payload, mtime: Date.now() }));
    }

    async _delete(path) {
        const store = await this._store("readwrite");
        return await _awaitRequest(store.delete(path));
    }

    async _allKeys() {
        const store = await this._store("readonly");
        return await _awaitRequest(store.getAllKeys());
    }

    // ------------------------------------------------------------------
    // Reads.
    // ------------------------------------------------------------------

    async readJson(path) {
        const rec = await this._get(path);
        if (!rec) throw new Error("IdbStorage: missing " + path);
        if (rec.kind === "json") return rec.payload;
        if (rec.kind === "text") return JSON.parse(rec.payload);
        throw new Error("IdbStorage: not JSON at " + path + " (kind=" + rec.kind + ")");
    }

    async readText(path) {
        const rec = await this._get(path);
        if (!rec) throw new Error("IdbStorage: missing " + path);
        if (rec.kind === "text") return rec.payload;
        if (rec.kind === "json") return JSON.stringify(rec.payload);
        throw new Error("IdbStorage: not text at " + path + " (kind=" + rec.kind + ")");
    }

    async readBytes(path) {
        const rec = await this._get(path);
        if (!rec) throw new Error("IdbStorage: missing " + path);
        if (rec.kind !== "bytes") throw new Error("IdbStorage: not bytes at " + path);
        const u8 = rec.payload;
        return u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
    }

    async tryReadJson(path)  { try { return await this.readJson(path); }  catch (e) { return null; } }
    async tryReadText(path)  { try { return await this.readText(path); }  catch (e) { return null; } }
    async tryReadBytes(path) { try { return await this.readBytes(path); } catch (e) { return null; } }

    // ------------------------------------------------------------------
    // Writes.
    // ------------------------------------------------------------------

    async writeJson(path, obj)  { await this._put(path, "json", obj); }
    async writeText(path, text) { await this._put(path, "text", String(text)); }

    async writeBytes(path, bytes) {
        let u8;
        if (bytes instanceof Uint8Array) u8 = bytes;
        else if (bytes instanceof ArrayBuffer) u8 = new Uint8Array(bytes);
        else u8 = new Uint8Array(bytes);
        // Detach from any SharedArrayBuffer / external view by copying into
        // a standalone Uint8Array; IDB structured-clones the payload so
        // a plain owned buffer is simplest to reason about.
        await this._put(path, "bytes", u8.slice());
    }

    async deletePath(path) {
        const keys = await this._allKeys();
        const prefix = path.endsWith("/") ? path : path + "/";
        let deleted = 0;
        for (const k of keys) {
            const kStr = String(k);
            if (kStr === path || kStr.startsWith(prefix)) {
                await this._delete(k);
                deleted++;
            }
        }
        return deleted > 0;
    }

    async listDir(path) {
        const keys = await this._allKeys();
        const prefix = path.endsWith("/") ? path : path + "/";
        const seen = new Set();
        for (const k of keys) {
            const kStr = String(k);
            if (!kStr.startsWith(prefix)) continue;
            const rest = kStr.slice(prefix.length);
            const first = rest.split("/")[0];
            if (first) seen.add(first);
        }
        return Array.from(seen);
    }

    async exists(path) {
        const rec = await this._get(path);
        if (rec) return true;
        const keys = await this._allKeys();
        const prefix = path.endsWith("/") ? path : path + "/";
        for (const k of keys) {
            if (String(k).startsWith(prefix)) return true;
        }
        return false;
    }

    /**
     * Every stored key that starts with `prefix`. IDB has no folder
     * notion so this is a flat prefix-filter over getAllKeys. Returned
     * with full paths so callers can feed them straight back into
     * readText / readBytes.
     */
    async allPaths(prefix) {
        const keys = await this._allKeys();
        const norm = prefix.endsWith("/") ? prefix : prefix + "/";
        return keys.map(String).filter((k) => k.startsWith(norm));
    }
}
