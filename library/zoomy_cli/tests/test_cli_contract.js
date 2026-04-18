/**
 * Phase 1 stub: CLI contract tests. Populated in Phase 3 once the
 * isomorphic zoomy_cli lands. For now the only assertion is that the
 * public entry points we plan to add eventually exist, so the test
 * surface shows up in `node --test` and doesn't silently fall out of
 * the CI matrix.
 *
 * Intentionally permissive: `t.todo` for work that isn't implemented.
 */
const { test, describe } = require("node:test");
const path = require("path");
const fs = require("fs");

const CLI_DIR = path.resolve(__dirname, "..");

describe("zoomy_cli — Phase 1 stub", () => {
    test("package.json exists and declares the cli.js main entry", () => {
        const pkg = JSON.parse(fs.readFileSync(path.join(CLI_DIR, "package.json"), "utf8"));
        if (pkg.name !== "zoomy-cli") throw new Error("expected name=zoomy-cli");
        if (!pkg.main) throw new Error("package.json.main missing");
    });

    test("current Node-only entry (cli.js) is a real file", () => {
        const p = path.join(CLI_DIR, "cli.js");
        if (!fs.existsSync(p)) throw new Error("cli.js missing");
    });

    // --- Phase 3 targets (to fill in when we refactor) ---

    test.todo("dual-export: package.json.exports has node and browser entries");
    test.todo("browser entry (browser.js) exists and exports a ZoomyCLI class");
    test.todo("PyodideAdapter implements runCode / extractParams / describeModel / openHdf5 / writeHdf5Bytes");
    test.todo("HttpAdapter implements connect / submitCase / cancelJob / listRegistry");
    test.todo("storage layer abstracts fs vs fetch behind the same interface");
});
