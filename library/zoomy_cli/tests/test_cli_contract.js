/**
 * Node:test contract checks for the isomorphic zoomy_cli package.
 *
 * We import the ESM surface via dynamic import(), so this file can live
 * as a CommonJS test runner without forcing "type":"module" on the
 * whole package (which would break the existing Node-only cli.js bin).
 */
const { test, describe, before } = require("node:test");
const path = require("path");
const fs = require("fs");
const url = require("url");

const CLI_DIR = path.resolve(__dirname, "..");

describe("zoomy_cli package shape", () => {
    test("package.json declares browser/node exports", () => {
        const pkg = JSON.parse(fs.readFileSync(path.join(CLI_DIR, "package.json"), "utf8"));
        if (!pkg.exports || !pkg.exports["."]) throw new Error("exports map missing");
        const root = pkg.exports["."];
        if (!root.browser || !root.node) throw new Error("browser/node exports missing");
    });

    test("browser.mjs and node.mjs exist", () => {
        for (const f of ["browser.mjs", "node.mjs"]) {
            if (!fs.existsSync(path.join(CLI_DIR, f))) throw new Error(f + " missing");
        }
    });

    test("src/cli.mjs, storage.mjs, adapters/{pyodide,http}_adapter.mjs exist", () => {
        for (const f of ["src/cli.mjs", "src/storage.mjs",
                         "src/adapters/pyodide_adapter.mjs",
                         "src/adapters/http_adapter.mjs"]) {
            if (!fs.existsSync(path.join(CLI_DIR, f))) throw new Error(f + " missing");
        }
    });
});

describe("zoomy_cli public surface", () => {
    let mod;
    before(async () => {
        mod = await import(url.pathToFileURL(path.join(CLI_DIR, "node.mjs")).href);
    });

    test("node.mjs re-exports ZoomyCLI, PyodideAdapter, HttpAdapter, FetchStorage, FsStorage, NotSupportedError", () => {
        const expected = ["ZoomyCLI", "PyodideAdapter", "HttpAdapter",
                          "FetchStorage", "FsStorage", "NotSupportedError"];
        for (const name of expected) {
            if (!(name in mod)) throw new Error("export '" + name + "' missing from node.mjs");
        }
    });

    test("ZoomyCLI constructs with a PyodideAdapter stub + storage", async () => {
        const pyodide = { tag: "pyodide", runCode: async () => "{}", extractParams: async () => "{}", describeModel: async () => "", openHdf5: async () => "ok", writeHdf5Bytes: async () => "ok", preloadParams: async () => "ok", interrupt: () => ({ mode: "noop" }) };
        const storage = { readJson: async () => ({}), readText: async () => "", tryReadJson: async () => null, tryReadText: async () => null };
        const cli = new mod.ZoomyCLI({ pyodide, storage });
        if (!cli) throw new Error("no cli");
        if (cli.pyodide !== pyodide) throw new Error("adapter not wired");
    });

    test("ZoomyCLI.runCode forwards to PyodideAdapter", async () => {
        let seen = null;
        const pyodide = { runCode: async (c) => { seen = c; return '{"status":"success","output":"ok"}'; } };
        const storage = { readJson: async () => ({}), readText: async () => "", tryReadJson: async () => null, tryReadText: async () => null };
        const cli = new mod.ZoomyCLI({ pyodide, storage });
        const out = await cli.runCode("print('x')");
        if (seen !== "print('x')") throw new Error("runCode did not forward code");
        if (out !== '{"status":"success","output":"ok"}') throw new Error("runCode did not forward result");
    });

    test("ZoomyCLI.listCards merges default/generated/user sources", async () => {
        const pyodide = { runCode: async () => "", interrupt: () => ({}) };
        const storage = {
            tryReadJson: async (path) => {
                if (path === "cards/models/default.json") return [{ id: "a" }];
                if (path === "cards/models/generated.json") return [{ id: "b" }];
                return null;              // user.json missing
            },
            readJson: async () => ({}), readText: async () => "", tryReadText: async () => null,
        };
        const cli = new mod.ZoomyCLI({ pyodide, storage });
        const cards = await cli.listCards("models");
        if (cards.length !== 2) throw new Error("expected 2 cards, got " + cards.length);
        if (cards[0].id !== "a" || cards[1].id !== "b") throw new Error("merge order wrong");
    });

    test("ZoomyCLI.listRegistry returns null with no HTTP adapters", async () => {
        const pyodide = { runCode: async () => "" };
        const storage = { tryReadJson: async () => null };
        const cli = new mod.ZoomyCLI({ pyodide, storage });
        if (await cli.listRegistry() !== null) throw new Error("expected null registry");
    });

    test("ZoomyCLI.cancel without tag calls pyodide.interrupt", async () => {
        let called = false;
        const pyodide = { interrupt: () => { called = true; return { mode: "cooperative" }; } };
        const storage = { tryReadJson: async () => null };
        const cli = new mod.ZoomyCLI({ pyodide, storage });
        const res = await cli.cancel({});
        if (!called) throw new Error("interrupt not called");
        if (res.mode !== "cooperative") throw new Error("unexpected result");
    });
});
