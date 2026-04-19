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
