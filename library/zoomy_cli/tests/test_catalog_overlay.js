/**
 * Catalog-overlay round-trip contract.
 *
 * The GUI card list = shipped cards/<dir>/default.json ⊕ a per-dir overlay
 * {removed:[ids], added:[cards]}. This test drives the CLI mechanism that
 * the GUI's save/load zip code calls (effectiveCatalog / adoptCatalog):
 *
 *   1. construct an overlay with a removal + an addition,
 *   2. export the effective catalog into a zip as cards/<dir>/default.json,
 *   3. assert every exported default.json satisfies the SAME schema the
 *      registry gate (tests/test_registry.py) enforces,
 *   4. clear the overlay (fresh browser) and adopt the zip's catalog,
 *   5. assert the effective catalog is byte-identical to step 2, and
 *   6. loading an existing zip WITHOUT a catalog (projects/zoomy-cases.zip)
 *      leaves the overlay untouched.
 */
const { test, describe, before, after } = require("node:test");
const assert = require("node:assert");
const path = require("path");
const fs = require("fs");
const os = require("os");
const url = require("url");
const JSZip = require("jszip");

const CLI_DIR = path.resolve(__dirname, "..");
const GUI_DIR = path.resolve(CLI_DIR, "..", "zoomy_gui");
const CARD_DIRS = ["models", "solvers", "meshes", "visualizations"];
const CATALOG_CONTENT = ["template", "snippet", "class", "mesh_file"];

/* Replicates tests/test_registry.py::test_card_schema for one card so the
   exported files are validated against the SAME rules. */
function assertCardSchema(dir, card) {
    assert.ok(typeof card.id === "string" && card.id, `${dir}: card missing string id`);
    assert.ok(typeof card.title === "string" && card.title, `${dir}: card ${card.id} missing title`);
    if (dir === "models") {
        assert.ok(card.template || card.snippet || card.class,
            `model card ${card.id} needs one of template/snippet/class`);
    } else if (dir === "meshes") {
        assert.ok(card.template || card.mesh_file,
            `mesh card ${card.id} needs a template or a mesh_file`);
    } else {
        const present = CATALOG_CONTENT.some((f) => card[f]);
        assert.ok(present || card.requires_tag,
            `card ${card.id} in ${dir} has no content field or requires_tag`);
    }
}

/* Effective catalog emitted into a zip exactly as app.js _addCatalogToZip. */
async function emitCatalogZip(cli) {
    const zip = new JSZip();
    for (const dir of CARD_DIRS) {
        const eff = await cli.effectiveCatalog(dir);
        zip.file(`cards/${dir}/default.json`, JSON.stringify(eff, null, 2) + "\n");
    }
    return await zip.generateAsync({ type: "nodebuffer" });
}

/* Adopt a zip's catalog exactly as app.js _adoptCatalogFromZip. */
async function adoptCatalogZip(cli, bytes) {
    const zip = await JSZip.loadAsync(bytes);
    const entries = {};
    zip.forEach((rel, entry) => {
        const m = /^cards\/(models|solvers|meshes|visualizations)\/default\.json$/.exec(rel);
        if (m && !entry.dir) entries[m[1]] = entry;
    });
    const dirs = Object.keys(entries);
    for (const dir of dirs) {
        const arr = JSON.parse(await entries[dir].async("string"));
        await cli.adoptCatalog(dir, arr);
    }
    return dirs.length;
}

describe("catalog overlay round-trip", () => {
    let mod;
    let root;
    let cli;

    before(async () => {
        mod = await import(url.pathToFileURL(path.join(CLI_DIR, "node.mjs")).href);
        root = fs.mkdtempSync(path.join(os.tmpdir(), "zoomy-catalog-"));
        // Copy the real shipped default.json for every tab into the temp root
        // so effectiveCatalog(dir) resolves the actual shipped catalog.
        for (const dir of CARD_DIRS) {
            const src = path.join(GUI_DIR, "cards", dir, "default.json");
            const dst = path.join(root, "cards", dir, "default.json");
            fs.mkdirSync(path.dirname(dst), { recursive: true });
            fs.copyFileSync(src, dst);
        }
        const storage = new mod.FsStorage({ root, fs, path });
        const pyodide = { runCode: async () => "", interrupt: () => ({}) };
        cli = new mod.ZoomyCLI({ pyodide, storage });
    });

    after(() => fs.rmSync(root, { recursive: true, force: true }));

    test("effectiveCatalog with no overlay == shipped default.json", async () => {
        for (const dir of CARD_DIRS) {
            const shipped = JSON.parse(fs.readFileSync(path.join(GUI_DIR, "cards", dir, "default.json"), "utf8"));
            const eff = await cli.effectiveCatalog(dir);
            assert.deepStrictEqual(eff, shipped, `${dir}: empty overlay must reproduce shipped`);
        }
    });

    test("removal + addition round-trips through export → adopt", async () => {
        // --- 1. shipped baseline for models.
        const shippedModels = await cli.effectiveCatalog("models");
        const removedId = shippedModels[0].id;               // remove the first shipped card
        assert.ok(removedId, "need a shipped model id to remove");
        const addedCard = {
            id: "catalog-my-model-abcd",
            title: "My Catalog Model",
            source: "catalog",
            description: "A user-curated catalog card.",
            template: "from zoomy_core.model.models import SWE\nmodel = SWE(dimension=1).system_model\n",
            untested: true,
        };

        // --- 2. write the overlay and compute the effective catalog.
        await cli.writeCatalogOverlay("models", { removed: [removedId], added: [addedCard] });
        const effBefore = {};
        for (const dir of CARD_DIRS) effBefore[dir] = await cli.effectiveCatalog(dir);

        const modelIds = effBefore.models.map((c) => c.id);
        assert.ok(!modelIds.includes(removedId), "removed card must be gone from effective");
        assert.ok(modelIds.includes(addedCard.id), "added card must be present in effective");

        // --- 3. export to a zip and validate every default.json's schema.
        const bytes = await emitCatalogZip(cli);
        const zip = await JSZip.loadAsync(bytes);
        for (const dir of CARD_DIRS) {
            const entry = zip.file(`cards/${dir}/default.json`);
            assert.ok(entry, `zip must contain cards/${dir}/default.json`);
            const arr = JSON.parse(await entry.async("string"));
            assert.ok(Array.isArray(arr), `${dir}: exported catalog must be a JSON array`);
            const seen = new Set();
            for (const card of arr) {
                assertCardSchema(dir, card);
                assert.ok(!seen.has(card.id), `${dir}: duplicate id ${card.id}`);
                seen.add(card.id);
            }
        }

        // --- 4. fresh browser: clear the overlay, then adopt the zip.
        for (const dir of CARD_DIRS) await cli.clearCatalogOverlay(dir);
        const clearedModels = (await cli.effectiveCatalog("models")).map((c) => c.id);
        assert.ok(clearedModels.includes(removedId), "clear must restore the removed card");
        assert.ok(!clearedModels.includes(addedCard.id), "clear must drop the added card");

        const nAdopted = await adoptCatalogZip(cli, bytes);
        assert.strictEqual(nAdopted, CARD_DIRS.length, "must adopt all four tabs");

        // --- 5. effective catalog identical after the round-trip.
        for (const dir of CARD_DIRS) {
            const effAfter = await cli.effectiveCatalog(dir);
            assert.deepStrictEqual(effAfter, effBefore[dir], `${dir}: catalog must survive export → adopt`);
        }
    });

    test("loading a catalog-less zip leaves the overlay untouched", async () => {
        // Seed a distinctive overlay.
        await cli.writeCatalogOverlay("solvers", { removed: ["ghost-solver"], added: [] });
        const before = await cli.readCatalogOverlay("solvers");

        // An existing session zip carries no cards/<dir>/default.json.
        const casesZip = path.join(GUI_DIR, "projects", "zoomy-cases.zip");
        const zip = await JSZip.loadAsync(fs.readFileSync(casesZip));
        const hasCatalog = Object.keys(zip.files).some(
            (k) => /^cards\/(models|solvers|meshes|visualizations)\/default\.json$/.test(k) && !zip.files[k].dir);
        assert.strictEqual(hasCatalog, false, "zoomy-cases.zip must not carry a catalog");

        const nAdopted = await adoptCatalogZip(cli, fs.readFileSync(casesZip));
        assert.strictEqual(nAdopted, 0, "no catalog dirs to adopt");

        const after = await cli.readCatalogOverlay("solvers");
        assert.deepStrictEqual(after, before, "overlay must be untouched by a catalog-less load");
    });
});
