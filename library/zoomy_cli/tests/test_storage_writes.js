/**
 * FsStorage write-surface round-trips. Covers the methods added when
 * the project grew user-authored cards (writeJson/Text/Bytes, listDir,
 * deletePath, exists). Browser-side IdbStorage is exercised end-to-end
 * by the puppeteer tests under zoomy_gui/tests; keeping node-side
 * coverage to FsStorage avoids pulling a fake-indexeddb devDep.
 */
const { test, describe, before, after } = require("node:test");
const assert = require("node:assert");
const path = require("path");
const fs = require("fs");
const os = require("os");
const url = require("url");

const CLI_DIR = path.resolve(__dirname, "..");

describe("FsStorage write surface", () => {
    let mod;
    let root;
    let storage;

    before(async () => {
        mod = await import(url.pathToFileURL(path.join(CLI_DIR, "node.mjs")).href);
        root = fs.mkdtempSync(path.join(os.tmpdir(), "zoomy-fsstore-"));
        storage = new mod.FsStorage({ root, fs, path });
    });

    after(() => {
        fs.rmSync(root, { recursive: true, force: true });
    });

    test("writeJson / readJson round-trips and creates parent dirs", async () => {
        await storage.writeJson("a/b/c.json", { hello: "world", n: 42 });
        const got = await storage.readJson("a/b/c.json");
        assert.deepStrictEqual(got, { hello: "world", n: 42 });
    });

    test("writeText / readText round-trips", async () => {
        await storage.writeText("plain.txt", "line1\nline2\n");
        assert.strictEqual(await storage.readText("plain.txt"), "line1\nline2\n");
    });

    test("writeBytes / readBytes round-trips a Uint8Array", async () => {
        const data = new Uint8Array([0xde, 0xad, 0xbe, 0xef, 0x00, 0x01, 0x02, 0x03]);
        await storage.writeBytes("bin/payload.bin", data);
        const got = new Uint8Array(await storage.readBytes("bin/payload.bin"));
        assert.deepStrictEqual(Array.from(got), Array.from(data));
    });

    test("writeBytes accepts an ArrayBuffer", async () => {
        const ab = new Uint8Array([1, 2, 3]).buffer;
        await storage.writeBytes("bin/ab.bin", ab);
        const got = new Uint8Array(await storage.readBytes("bin/ab.bin"));
        assert.deepStrictEqual(Array.from(got), [1, 2, 3]);
    });

    test("exists reports presence for both files and directories", async () => {
        assert.strictEqual(await storage.exists("a/b/c.json"), true);
        assert.strictEqual(await storage.exists("a/b"), true);
        assert.strictEqual(await storage.exists("a/nope.json"), false);
    });

    test("listDir returns only immediate children, empty for missing", async () => {
        await storage.writeText("zoo/sub1/x.txt", "x");
        await storage.writeText("zoo/sub2/y.txt", "y");
        await storage.writeText("zoo/top.txt", "top");
        const kids = (await storage.listDir("zoo")).sort();
        assert.deepStrictEqual(kids, ["sub1", "sub2", "top.txt"]);
        assert.deepStrictEqual(await storage.listDir("missing/dir"), []);
    });

    test("deletePath removes files and folders recursively", async () => {
        await storage.writeText("tree/a/1.txt", "1");
        await storage.writeText("tree/a/2.txt", "2");
        await storage.writeText("tree/b/3.txt", "3");
        assert.strictEqual(await storage.deletePath("tree/a"), true);
        assert.strictEqual(await storage.exists("tree/a"), false);
        assert.strictEqual(await storage.exists("tree/b/3.txt"), true);
        // Deleting a missing path is a no-op that returns false.
        assert.strictEqual(await storage.deletePath("tree/a"), false);
    });

    test("tryReadJson returns null for missing and invalid JSON", async () => {
        assert.strictEqual(await storage.tryReadJson("missing.json"), null);
        await storage.writeText("broken.json", "{ not json");
        assert.strictEqual(await storage.tryReadJson("broken.json"), null);
    });
});

describe("FetchStorage overlay plumbing (without network)", () => {
    let mod;

    before(async () => {
        mod = await import(url.pathToFileURL(path.join(CLI_DIR, "node.mjs")).href);
    });

    test("writes and overlay-gated reads delegate to the overlay", async () => {
        // A minimal in-memory overlay implementing the IdbStorage surface.
        const mem = new Map();
        const overlay = {
            async writeJson(p, o)  { mem.set(p, { k: "json", v: o }); },
            async writeText(p, s)  { mem.set(p, { k: "text", v: String(s) }); },
            async writeBytes(p, b) { mem.set(p, { k: "bytes", v: new Uint8Array(b) }); },
            async tryReadJson(p)   { const r = mem.get(p); return r && r.k === "json" ? r.v : null; },
            async tryReadText(p)   { const r = mem.get(p); return r && r.k === "text" ? r.v : null; },
            async tryReadBytes(p)  { const r = mem.get(p); return r && r.k === "bytes" ? r.v.buffer : null; },
            async deletePath(p)    { return mem.delete(p); },
            async listDir(p)       {
                const prefix = p.endsWith("/") ? p : p + "/";
                const out = new Set();
                for (const k of mem.keys()) {
                    if (k.startsWith(prefix)) out.add(k.slice(prefix.length).split("/")[0]);
                }
                return Array.from(out);
            },
            async exists(p)        { return mem.has(p); },
        };
        const storage = new mod.FetchStorage({ overlay });

        await storage.writeJson("cards/sessions/s1/models/user/abc/card.json", { id: "abc" });
        // Overlay-first applies under cards/sessions/.
        const got = await storage.readJson("cards/sessions/s1/models/user/abc/card.json");
        assert.deepStrictEqual(got, { id: "abc" });

        assert.deepStrictEqual(
            await storage.listDir("cards/sessions/s1/models/user"),
            ["abc"],
        );
        assert.strictEqual(await storage.exists("cards/sessions/s1/models/user/abc/card.json"), true);
    });

    test("writes throw cleanly when no overlay is configured", async () => {
        const storage = new mod.FetchStorage();
        await assert.rejects(() => storage.writeJson("x.json", {}), /no writable overlay/);
        await assert.rejects(() => storage.writeText("x.txt", "x"), /no writable overlay/);
        await assert.rejects(() => storage.deletePath("x"),        /no writable overlay/);
    });
});
