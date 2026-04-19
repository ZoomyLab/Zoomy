/**
 * End-to-end check of the `zoomy card` / `zoomy mesh` shell commands.
 *
 * Spawns a real child `node cli.js ...` inside a tmpdir (each suite
 * gets its own so tests don't step on each other's state.json / card
 * folders), asserts stdout + on-disk layout after each invocation.
 */
const { test, describe, before, after } = require("node:test");
const assert = require("node:assert");
const { spawnSync } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");
const os = require("node:os");

const CLI = path.resolve(__dirname, "..", "cli.js");

function runCli(cwd, args) {
    const r = spawnSync(process.execPath, [CLI].concat(args), {
        cwd, encoding: "utf8",
        env: Object.assign({}, process.env, { ZOOMY_NO_INTERACTIVE: "1" }),
    });
    return { status: r.status, stdout: r.stdout, stderr: r.stderr };
}

/**
 * Start-ish without the interactive "add to .bashrc?" prompt —
 * initialise just what the rest of the commands need: a state.json
 * with a default session id. Keeps the test non-interactive.
 */
function bootstrapProject(cwd) {
    fs.mkdirSync(path.join(cwd, ".zoomy"), { recursive: true });
    fs.writeFileSync(path.join(cwd, ".zoomy", "state.json"), JSON.stringify({
        selections: {},
        sessions: [{ id: "test-session", title: "Test", description: "", selections: {}, cardOverrides: {} }],
        activeSession: "test-session",
        backends: {},
    }, null, 2));
}

describe("zoomy card / zoomy mesh shell commands", () => {
    let tmp;

    before(() => {
        tmp = fs.mkdtempSync(path.join(os.tmpdir(), "zoomy-cli-cmd-"));
        bootstrapProject(tmp);
    });
    after(() => { fs.rmSync(tmp, { recursive: true, force: true }); });

    test("card new model creates a folder with card.json + snippet.py", () => {
        const r = runCli(tmp, ["card", "new", "model", "My Model"]);
        assert.strictEqual(r.status, 0, "stderr: " + r.stderr);
        assert.match(r.stdout, /Created models\/user\/user-my-model-/);
        assert.match(r.stdout, /session: test-session/);

        const userDir = path.join(tmp, "cards", "sessions", "test-session", "models", "user");
        const ids = fs.readdirSync(userDir);
        assert.strictEqual(ids.length, 1);
        const id = ids[0];
        assert.match(id, /^user-my-model-/);
        const meta = JSON.parse(fs.readFileSync(path.join(userDir, id, "card.json"), "utf8"));
        assert.strictEqual(meta.title, "My Model");
        assert.strictEqual(meta.source, "user");
        const snippet = fs.readFileSync(path.join(userDir, id, "snippet.py"), "utf8");
        assert.match(snippet, /SMEInviscid/);
    });

    test("card new respects --session=<other>", () => {
        const r = runCli(tmp, ["card", "new", "solver", "sol", "--session", "other-session"]);
        assert.strictEqual(r.status, 0, "stderr: " + r.stderr);
        assert.match(r.stdout, /session: other-session/);
        const otherDir = path.join(tmp, "cards", "sessions", "other-session", "solvers", "user");
        assert.ok(fs.existsSync(otherDir), "other-session/solvers/user dir missing");
        const ids = fs.readdirSync(otherDir);
        assert.strictEqual(ids.length, 1);
    });

    test("card list enumerates the active session's cards only", () => {
        const r = runCli(tmp, ["card", "list"]);
        assert.strictEqual(r.status, 0, "stderr: " + r.stderr);
        assert.match(r.stdout, /User cards in session test-session/);
        assert.match(r.stdout, /user-my-model-/);
        // Other-session's solver card must NOT show.
        assert.ok(!/solver/i.test(r.stdout), "solver leaked across sessions: " + r.stdout);
    });

    test("card list with --session shows a different session", () => {
        const r = runCli(tmp, ["card", "list", "--session", "other-session"]);
        assert.strictEqual(r.status, 0, "stderr: " + r.stderr);
        assert.match(r.stdout, /User cards in session other-session/);
        assert.match(r.stdout, /solvers:/);
    });

    test("card delete by partial name removes the folder", () => {
        const before = fs.readdirSync(
            path.join(tmp, "cards", "sessions", "test-session", "models", "user"),
        );
        assert.strictEqual(before.length, 1);
        const r = runCli(tmp, ["card", "delete", "model", "My Model"]);
        assert.strictEqual(r.status, 0, "stderr: " + r.stderr);
        assert.match(r.stdout, /Deleted models\/user\/user-my-model-/);
        const after = fs.readdirSync(
            path.join(tmp, "cards", "sessions", "test-session", "models", "user"),
        );
        assert.strictEqual(after.length, 0);
    });

    test("card delete of unknown id exits non-zero with a helpful message", () => {
        const r = runCli(tmp, ["card", "delete", "model", "does-not-exist"]);
        assert.notStrictEqual(r.status, 0);
        assert.match(r.stderr, /No user card 'does-not-exist'/);
    });

    test("mesh upload copies .msh bytes and writes a meshio snippet", () => {
        const meshFile = path.join(tmp, "tiny.msh");
        fs.writeFileSync(meshFile,
            "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n1\n1 0 0 0\n$EndNodes\n$Elements\n0\n$EndElements\n",
        );
        const r = runCli(tmp, ["mesh", "upload", "tiny.msh", "--name", "tinymesh"]);
        assert.strictEqual(r.status, 0, "stderr: " + r.stderr);
        assert.match(r.stdout, /Uploaded tiny\.msh/);
        assert.match(r.stdout, /id:\s+user-mesh-tinymesh-/);

        const userDir = path.join(tmp, "cards", "sessions", "test-session", "meshes", "user");
        const ids = fs.readdirSync(userDir);
        assert.strictEqual(ids.length, 1);
        const id = ids[0];
        assert.ok(fs.existsSync(path.join(userDir, id, "mesh.msh")), "mesh.msh missing");
        const meta = JSON.parse(fs.readFileSync(path.join(userDir, id, "card.json"), "utf8"));
        assert.strictEqual(meta.title, "tinymesh");
        assert.strictEqual(meta.mesh_file, "mesh.msh");
        assert.match(meta.mesh_vpath, /\/tmp\/zoomy_user\/test-session\/meshes\/user-mesh-tinymesh-/);
        assert.match(meta.category, /Uploaded/);
        const snippet = fs.readFileSync(path.join(userDir, id, "snippet.py"), "utf8");
        assert.match(snippet, /meshio\.read/);
    });

    test("card new with unknown type exits non-zero", () => {
        const r = runCli(tmp, ["card", "new", "bogus", "name"]);
        assert.notStrictEqual(r.status, 0);
        assert.match(r.stderr, /Unknown card type: bogus/);
    });
});
