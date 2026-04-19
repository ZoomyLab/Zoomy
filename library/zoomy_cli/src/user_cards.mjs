/**
 * Session-scoped CRUD helpers for user-authored cards. Layered on top
 * of any Storage implementation (FsStorage on Node, FetchStorage with
 * IdbStorage overlay in the browser) so the same code path serves both
 * GUI writes and CLI writes.
 *
 * On-disk / in-IDB layout:
 *
 *   cards/sessions/<session_id>/<type>/user/<card_id>/card.json
 *   cards/sessions/<session_id>/<type>/user/<card_id>/snippet.py
 *   cards/sessions/<session_id>/<type>/user/<card_id>/mesh.msh   (mesh cards only)
 *
 * `<type>` is `models` | `solvers` | `meshes` | `visualizations`.
 * `<session_id>` is whatever SessionManager mints
 * (`"session-<timestamp>-<rand>"`); the CLI defaults to the literal
 * "default" when none is supplied.
 */

/** Folder that holds one user card's files. */
export function userCardDir(session, type, id) {
    if (!session) throw new Error("userCardDir: session required");
    if (!type)    throw new Error("userCardDir: type required");
    if (!id)      throw new Error("userCardDir: id required");
    return "cards/sessions/" + session + "/" + type + "/user/" + id;
}

/** Parent folder enumerated by listUserCards. */
export function userCardsRoot(session, type) {
    return "cards/sessions/" + session + "/" + type + "/user";
}

/**
 * Write a user card's metadata + snippet. Returns the stored meta
 * (with `source: "user"` injected so the GUI can gate the trash icon).
 */
export async function writeUserCard(storage, opts) {
    opts = opts || {};
    const { session, type, id, meta, snippet } = opts;
    if (!session) throw new Error("writeUserCard: session required");
    if (!type)    throw new Error("writeUserCard: type required");
    if (!id)      throw new Error("writeUserCard: id required");

    const dir = userCardDir(session, type, id);
    const full = Object.assign({}, meta || {}, { id, source: "user" });
    // Point the card at its own snippet.py by default; callers can
    // override by setting meta.snippet / meta.template / meta.class
    // explicitly.
    if (snippet !== undefined && !full.snippet && !full.template && !full.class) {
        full.snippet = dir + "/snippet.py";
    }
    await storage.writeJson(dir + "/card.json", full);
    if (snippet !== undefined) {
        await storage.writeText(dir + "/snippet.py", snippet);
    }
    return full;
}

/** Read one user card's {meta, snippet} — returns null if missing. */
export async function readUserCard(storage, session, type, id) {
    const dir = userCardDir(session, type, id);
    const meta = await storage.tryReadJson(dir + "/card.json");
    if (!meta) return null;
    const snippet = await storage.tryReadText(dir + "/snippet.py");
    return { meta, snippet };
}

/**
 * List every user card under a (session, type). Skips any folders that
 * don't hold a readable card.json so a partial write (e.g. interrupted
 * delete) can't poison the tab.
 */
export async function listUserCards(storage, session, type) {
    if (!session || !type) return [];
    const root = userCardsRoot(session, type);
    const ids = await storage.listDir(root);
    const out = [];
    for (const id of ids) {
        const meta = await storage.tryReadJson(root + "/" + id + "/card.json");
        if (!meta) continue;
        // Always mark source so the GUI gates its trash icon on it,
        // even if the stored JSON somehow lost the field.
        out.push(Object.assign({}, meta, { source: "user" }));
    }
    return out;
}

/** Recursive delete. Returns true when at least one file was removed. */
export async function deleteUserCard(storage, session, type, id) {
    return await storage.deletePath(userCardDir(session, type, id));
}

/** Writes an auxiliary file under a card folder (e.g. mesh.msh). */
export async function writeUserFile(storage, session, type, id, filename, bytes) {
    await storage.writeBytes(userCardDir(session, type, id) + "/" + filename, bytes);
}

/** Reads an auxiliary file; returns null if missing. */
export async function readUserFile(storage, session, type, id, filename) {
    return await storage.tryReadBytes(userCardDir(session, type, id) + "/" + filename);
}

/**
 * Starter meta + snippet for a freshly-minted user card. Kept in the
 * shared CLI module so the browser GUI (`+ New card`) and the Node
 * shell (`zoomy card new`) produce identical skeletons.
 *
 * @param {string} cardType  One of: "model", "solver", "mesh", "vis".
 * @param {string} title     User-supplied display title.
 */
export function cardStarter(cardType, title) {
    const description = "User-authored " + (cardType || "card") + ". Edit the code below and hit Run.";
    if (cardType === "model") {
        return {
            meta: { title, description },
            snippet:
"from zoomy_core.model.models.sme_model import SMEInviscid\n" +
"\n" +
"# Subclass SMEInviscid (or any zoomy_core model) and tweak what you need.\n" +
"class UserModel(SMEInviscid):\n" +
"    pass\n" +
"\n" +
"model = UserModel(level=0)\n" +
"display(model.describe())\n",
        };
    }
    if (cardType === "solver") {
        return {
            meta: { title, description, requires_tag: "numpy" },
            snippet:
"# Solver card. `model` and `mesh` come from the currently selected\n" +
"# Model and Mesh cards. Adjust t_end / reconstruction_order to taste.\n" +
"from zoomy_core.solver.numpy_solver import NumpySolver\n" +
"\n" +
"solver = NumpySolver(model, mesh, t_end=1.0, reconstruction_order=1)\n" +
"result = solver.run()\n" +
"print('simulation finished — %d steps' % result.n_steps)\n",
        };
    }
    if (cardType === "vis") {
        return {
            meta: { title, description },
            snippet:
"import matplotlib.pyplot as plt\n" +
"\n" +
"fig, ax = plt.subplots()\n" +
"ax.set_title('" + String(title).replace(/'/g, "") + "')\n" +
"ax.plot([0, 1, 2], [0, 1, 4])\n" +
"display(fig)\n",
        };
    }
    if (cardType === "mesh") {
        return {
            meta: { title, description },
            snippet:
"# Starter mesh card. Replace with your mesh construction code, or use\n" +
"# `zoomy mesh upload <file.msh>` (CLI) / the Upload .msh button (GUI)\n" +
"# to wrap a gmsh file.\n" +
"from zoomy_core.mesh.structured_mesh import StructuredMesh\n" +
"\n" +
"mesh = StructuredMesh(dim=1, n_cells=100, x_min=0.0, x_max=1.0)\n" +
"display(mesh)\n",
        };
    }
    return { meta: { title, description }, snippet: "# New user card\n" };
}

/**
 * Starter for an uploaded .msh. The meta carries `mesh_file` and
 * `mesh_vpath` so the GUI's executeCard can hydrate IDB bytes into
 * Pyodide's VFS before `meshio.read(vpath)` runs.
 */
export function uploadedMeshStarter({ title, filename, vpath, size }) {
    return {
        meta: {
            title,
            description: "Uploaded mesh: " + filename + (size ? " (" + size + " bytes)" : ""),
            category: "Uploaded",
            mesh_file: "mesh.msh",
            mesh_vpath: vpath,
        },
        snippet:
"# Auto-generated for an uploaded .msh. The bytes are hydrated into the\n" +
"# Pyodide VFS by the runtime before this snippet runs.\n" +
"import meshio\n" +
"\n" +
"mesh = meshio.read(\"" + vpath + "\")\n" +
"n_pts = mesh.points.shape[0]\n" +
"n_cells = sum(len(c.data) for c in mesh.cells)\n" +
"print(f\"Loaded " + String(filename).replace(/"/g, '\\"') + ": {n_pts} points, {n_cells} cells\")\n" +
"display(mesh)\n",
    };
}

/** Normalize a tab-alias into the on-disk dir name. */
export function typeDir(type) {
    const map = {
        model: "models", models: "models",
        solver: "solvers", solvers: "solvers",
        mesh: "meshes", meshes: "meshes",
        vis: "visualizations", viz: "visualizations",
        visualization: "visualizations", visualizations: "visualizations",
    };
    return map[type] || null;
}

/** Slugify a user-supplied name into a URL-safe id fragment. */
export function slugify(s) {
    return String(s || "").toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "")
        .slice(0, 40) || "card";
}
