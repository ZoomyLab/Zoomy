// The GUI's own case composition, for the harness.
//   node compose.mjs CLI_MJS GUI_DIR < items.json  ->  [{spec, casePy}, ...]
// items: [{session, timeEnd}] (a packed session) or [{solverCard, timeEnd}] (a catalog solver
// card in a fresh GUI). The spec is built as zoomy-theia-ext/src/browser/model-config-widget.ts
// builds it (`materializeSessions`, resp. `gatherSpec` with the default cards), `timeEnd` is
// the Parameters panel's time_end; the case.py is `ZoomyCLI.exportCase(spec, "py")` of the
// tested checkout's library/zoomy_cli, which the deployed GUI loads; `route` is where the GUI's
// Run button sends it (browser | backend).
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const [cliPath, guiDir] = process.argv.slice(2);
const { ZoomyCLI } = await import(pathToFileURL(cliPath).href);
const cli = new ZoomyCLI({ storage: {}, pyodide: {} });   // composition needs neither
const cards = {};
for (const t of ["models", "meshes", "solvers", "visualizations"]) {
    cards[t] = JSON.parse(readFileSync(join(guiDir, "cards", t, "default.json"), "utf8"));
}
const snippet = (c) => readFileSync(join(guiDir, c.snippet), "utf8");

// --- model-config-widget.ts, verbatim in substance ---------------------------------------
function fillTemplate(tpl, init) {
    if (!tpl) { return ""; }
    return tpl.replace(/\{(\w+)\}/g, (m, k) => (init && k in init ? String(init[k]) : m));
}
function cardCode(card, init) {
    if (card.template) { return fillTemplate(card.template, init); }
    if (card.class) {
        const dot = card.class.lastIndexOf(".");
        const mod = card.class.slice(0, dot), cls = card.class.slice(dot + 1);
        const kw = Object.entries(init || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
        return `from ${mod} import ${cls}\n\nmodel = ${cls}(${kw})\ndisplay(model.describe())`;
    }
    return undefined;
}
const slug = (t) => String(t || "session").trim().replace(/[^a-zA-Z0-9_-]+/g, "_") || "session";

function materialize(sess) {                       // materializeSessions, one session
    const catalog = (dir, id) => {
        if (!id) { return undefined; }
        const cs = cards[dir] || [];
        return cs.find((c) => c.id === id) || cs.find((c) => "card-" + c.id === id);
    };
    const sel = sess.selections || {}; const ov = sess.cardOverrides || {};
    const over = (id) => (id && ov[id]) || {};
    const params = (card, id) => ({ ...(card?.init || {}), ...(over(id).params || {}) });
    const code = (card, id) => {
        const o = over(id);
        return o.code != null ? String(o.code) : (card ? (cardCode(card, params(card, id)) || "") : "");
    };
    const model = catalog("models", sel.model), mesh = catalog("meshes", sel.mesh), solver = catalog("solvers", sel.solver);
    const viz = catalog("visualizations", sel.visualization) || cards.visualizations.find((c) => c.snippet);
    const hasSolver = !!(solver || over(sel.solver).code != null);
    const spec = {
        meta: { title: sess.title || slug(sess.id), description: sess.description || "" },
        model: { code: code(model, sel.model), class_path: model?.class || null, init: params(model, sel.model), card: model?.id || null },
        steps: Array.isArray(sess.steps) ? sess.steps.map((st) => ({ title: String(st.title || "Step"), code: String(st.code || "") })) : [],
        mesh: (mesh || over(sel.mesh).code != null)
            ? { code: code(mesh, sel.mesh), spec: params(mesh, sel.mesh), card: mesh?.id || null }
            : null,
        settings: hasSolver ? {} : null,
        solver: hasSolver ? { tag: solver?.requires_tag || "numpy", id: solver?.id || null, params: over(sel.solver).params || {} } : null,
        run: hasSolver ? { code: code(solver, sel.solver) } : null,
    };
    const own = over(sel.visualization).code;
    if (own != null) { spec.visualization = { code: String(own), card: viz?.id || null }; }
    else if (sel.visualization && viz?.snippet) { spec.visualization = { code: cli.vizPrelude() + "\n" + snippet(viz), card: viz.id }; }
    else { spec.visualization = null; }
    return spec;
}

function fresh(solver, edits) {                    // gatherSpec, a fresh GUI with `solver` picked
    const first = (dir) => cards[dir].find((c) => !!cardCode(c, c.init || {}));
    const model = first("models"), mesh = first("meshes");
    const viz = cards.visualizations.find((c) => c.snippet);
    const mergedInit = (c) => ({ ...(c.init || {}), ...((c === solver && edits) || {}) });
    const spec = {
        meta: { title: model?.title || "Zoomy case", description: "Exported from the Zoomy model-config GUI." },
        model: { code: cardCode(model, mergedInit(model)) || "", class_path: model?.class || null, init: mergedInit(model), card: model?.id || null },
        steps: [],
        mesh: { code: cardCode(mesh, mergedInit(mesh)) || "", spec: mergedInit(mesh), card: mesh.id },
        settings: {},
        solver: { tag: solver.requires_tag || "numpy", id: solver.id || null, params: mergedInit(solver) },
    };
    const solverCode = cardCode(solver, mergedInit(solver)) || "";
    if (solverCode) { spec.run = { code: solverCode }; }
    const prelude = cli.vizPrelude();
    spec.visualization = { code: [prelude, "# --- " + (viz.title || viz.id) + " ---", snippet(viz)].join("\n"), card: viz.id };
    return spec;
}

// runAssembly: where Run sends the case. `solverLocal = !!cardCode(solver, mergedInit(solver))`
// on the PICKED catalog card (the selected one, else the first runnable): local means the
// browser's Pyodide, else a connected backend of the card's requires_tag.
function route(solverId) {
    const picked = cards.solvers.find((c) => c.id === solverId)
        || cards.solvers.find((c) => !!cardCode(c, c.init || {}));
    return picked && cardCode(picked, picked.init || {}) ? "browser" : "backend";
}

// ------------------------------------------------------------------------------------------
const items = JSON.parse(readFileSync(0, "utf8"));
const out = items.map((it) => {
    let spec;
    if (it.session) {
        const s = JSON.parse(JSON.stringify(it.session));
        if (it.timeEnd != null) {                  // the Parameters panel's time_end
            const id = (s.selections || {}).solver;
            s.cardOverrides = s.cardOverrides || {};
            s.cardOverrides[id] = s.cardOverrides[id] || {};
            s.cardOverrides[id].params = { ...(s.cardOverrides[id].params || {}), time_end: it.timeEnd };
        }
        spec = materialize(s);
    } else {
        const solver = cards.solvers.find((c) => c.id === it.solverCard);
        spec = fresh(solver, it.timeEnd != null ? { time_end: it.timeEnd } : null);
    }
    return { spec, casePy: cli.exportCase(spec, "py"), route: route(spec.solver && spec.solver.id) };
});
process.stdout.write(JSON.stringify(out));
