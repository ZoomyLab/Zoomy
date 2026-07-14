/**
 * user_cards helpers + listCards merge precedence. Uses FsStorage over
 * a tmpdir so the full write→read→list→delete cycle is exercised.
 */
const { test, describe, before, after } = require("node:test");
const assert = require("node:assert");
const path = require("path");
const fs = require("fs");
const os = require("os");
const url = require("url");

const CLI_DIR = path.resolve(__dirname, "..");

describe("user_cards CRUD", () => {
    let mod;
    let root;
    let storage;
    let userCards;

    before(async () => {
        mod = await import(url.pathToFileURL(path.join(CLI_DIR, "node.mjs")).href);
        root = fs.mkdtempSync(path.join(os.tmpdir(), "zoomy-ucards-"));
        storage = new mod.FsStorage({ root, fs, path });
        userCards = mod.userCards;
    });

    after(() => fs.rmSync(root, { recursive: true, force: true }));

    test("writeUserCard writes card.json and snippet.py under the right folder", async () => {
        const meta = await userCards.writeUserCard(storage, {
            session: "session-1", type: "models", id: "user-mymodel",
            meta:    { title: "My Model" },
            snippet: "model = None\n",
        });
        assert.strictEqual(meta.id, "user-mymodel");
        assert.strictEqual(meta.source, "user");
        assert.strictEqual(meta.title, "My Model");
        // snippet: attribute defaulted to the card's own snippet.py path.
        assert.strictEqual(
            meta.snippet,
            "cards/sessions/session-1/models/user/user-mymodel/snippet.py",
        );

        const dir = userCards.userCardDir("session-1", "models", "user-mymodel");
        assert.strictEqual(await storage.exists(dir + "/card.json"), true);
        assert.strictEqual(await storage.exists(dir + "/snippet.py"), true);
        assert.strictEqual(await storage.readText(dir + "/snippet.py"), "model = None\n");
    });

    test("readUserCard returns meta + snippet", async () => {
        const got = await userCards.readUserCard(storage, "session-1", "models", "user-mymodel");
        assert.ok(got);
        assert.strictEqual(got.meta.id, "user-mymodel");
        assert.strictEqual(got.snippet, "model = None\n");
    });

    test("listUserCards finds every card under a (session, type)", async () => {
        await userCards.writeUserCard(storage, {
            session: "session-1", type: "models", id: "user-second",
            meta: { title: "Second" }, snippet: "",
        });
        const cards = await userCards.listUserCards(storage, "session-1", "models");
        const ids = cards.map(c => c.id).sort();
        assert.deepStrictEqual(ids, ["user-mymodel", "user-second"]);
        for (const c of cards) assert.strictEqual(c.source, "user");
    });

    test("session isolation — cards in one session don't bleed into another", async () => {
        await userCards.writeUserCard(storage, {
            session: "session-2", type: "models", id: "user-a",
            meta: { title: "A" }, snippet: "",
        });
        const s1 = (await userCards.listUserCards(storage, "session-1", "models")).map(c => c.id).sort();
        const s2 = (await userCards.listUserCards(storage, "session-2", "models")).map(c => c.id).sort();
        assert.deepStrictEqual(s1, ["user-mymodel", "user-second"]);
        assert.deepStrictEqual(s2, ["user-a"]);
    });

    test("writeUserFile / readUserFile round-trip bytes for mesh uploads", async () => {
        const bytes = new Uint8Array([0x4d, 0x53, 0x48, 0x01, 0x02]);  // "MSH" + junk
        await userCards.writeUserFile(
            storage, "session-1", "meshes", "user-mesh-a", "mesh.msh", bytes,
        );
        const got = await userCards.readUserFile(storage, "session-1", "meshes", "user-mesh-a", "mesh.msh");
        assert.ok(got);
        assert.deepStrictEqual(Array.from(new Uint8Array(got)), Array.from(bytes));
    });

    test("deleteUserCard removes the folder recursively", async () => {
        assert.strictEqual(
            await userCards.deleteUserCard(storage, "session-1", "models", "user-mymodel"),
            true,
        );
        const remaining = (await userCards.listUserCards(storage, "session-1", "models")).map(c => c.id);
        assert.deepStrictEqual(remaining, ["user-second"]);
    });

    test("listUserCards skips id folders with a missing card.json", async () => {
        // Simulate a partial write (snippet landed, card.json didn't).
        await storage.writeText(
            "cards/sessions/session-1/models/user/broken/snippet.py", "x = 1\n",
        );
        const ids = (await userCards.listUserCards(storage, "session-1", "models")).map(c => c.id);
        assert.ok(!ids.includes("broken"), "should skip folders without card.json");
    });
});

describe("ZoomyCLI.listCards with session override", () => {
    let mod;
    let root;
    let cli;

    before(async () => {
        mod = await import(url.pathToFileURL(path.join(CLI_DIR, "node.mjs")).href);
        root = fs.mkdtempSync(path.join(os.tmpdir(), "zoomy-listcards-"));
        const storage = new mod.FsStorage({ root, fs, path });

        // Seed the authored registry (the sole static tier).
        await storage.writeJson("cards/models/default.json", [
            { id: "sme-l0", title: "SME L0" },
            { id: "sme-l1", title: "SME L1" },
        ]);

        // Session-scoped user card that overrides sme-l0 by id.
        await mod.userCards.writeUserCard(storage, {
            session: "sess-42", type: "models", id: "sme-l0",
            meta: { title: "My Override" }, snippet: "",
        });
        // Session-scoped user card that is brand-new.
        await mod.userCards.writeUserCard(storage, {
            session: "sess-42", type: "models", id: "user-novel",
            meta: { title: "Novel" }, snippet: "",
        });

        const pyodide = { runCode: async () => "", interrupt: () => ({}) };
        cli = new mod.ZoomyCLI({ pyodide, storage });
    });

    after(() => fs.rmSync(root, { recursive: true, force: true }));

    test("without session, only the authored default shows", async () => {
        const cards = await cli.listCards("models");
        const ids = cards.map(c => c.id);
        assert.deepStrictEqual(ids, ["sme-l0", "sme-l1"]);
        assert.strictEqual(cards[0].title, "SME L0");
    });

    test("with session, per-session user cards override the default by id", async () => {
        const cards = await cli.listCards("models", { session: "sess-42" });
        const ids = cards.map(c => c.id);
        assert.deepStrictEqual(ids, ["sme-l0", "sme-l1", "user-novel"]);
        // sme-l0 was overridden by the session's user card.
        const override = cards.find(c => c.id === "sme-l0");
        assert.strictEqual(override.title, "My Override");
        assert.strictEqual(override.source, "user");
        // Novel card tacked on at the end with source=user.
        const novel = cards.find(c => c.id === "user-novel");
        assert.strictEqual(novel.source, "user");
    });

    test("unknown session returns the authored default only", async () => {
        const cards = await cli.listCards("models", { session: "sess-does-not-exist" });
        const ids = cards.map(c => c.id);
        assert.deepStrictEqual(ids, ["sme-l0", "sme-l1"]);
        assert.strictEqual(cards[0].title, "SME L0");
    });
});
