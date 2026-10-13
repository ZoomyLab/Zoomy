/**
 * Where the GUI's Run sends a case (src/cards.mjs, used by zoomy-theia-ext
 * model-config-widget.ts runAssembly): the browser runs only the cards of its own
 * backend (numpy); every other solver card goes to a connected backend of its tag,
 * whatever code it carries. A backend card without a template contributes no code, so a
 * fresh case's Run cell is the generated runner, not a construct of the solver class.
 */
const { test, describe, before } = require("node:test");
const assert = require("node:assert");
const path = require("path");
const fs = require("fs");
const url = require("url");

const CLI_DIR = path.resolve(__dirname, "..");
const SOLVERS = path.resolve(CLI_DIR, "..", "zoomy_gui", "cards", "solvers", "default.json");

let cards, ZoomyCLI;
before(async () => {
    const mod = await import(url.pathToFileURL(path.join(CLI_DIR, "node.mjs")).href);
    cards = mod.cards;
    ZoomyCLI = mod.ZoomyCLI;
});

const jax = { id: "solver-jax", requires_tag: "jax", class: "zoomy_jax.symsolver.solver.SymbolicHyperbolicSolverJax" };
const amrex = { id: "solver-amrex", requires_tag: "amrex", class: "zoomy_amrex.symsolver.SymbolicHyperbolicSolverAmrex" };
const foam = { id: "solver-foam", requires_tag: "OpenFOAM" };
const numpy = { id: "solver-numpy", requires_tag: "numpy", class: "zoomy_core.symsolver.numpy_backend.SymbolicHyperbolicSolverNumpy",
                template: "solver = S(time_end={time_end})", init: { time_end: 0.1 } };

describe("solver routing", () => {
    test("a card with a class but another backend goes to the backend", () => {
        assert.strictEqual(cards.solverRoute(jax), cards.Route.backend);
        assert.strictEqual(cards.solverRoute(amrex), cards.Route.backend);
        assert.strictEqual(cards.solverRoute({ ...jax, template: "x = 1" }), cards.Route.backend);
    });
    test("foam and OpenFOAM are one backend, not the browser's", () => {
        assert.strictEqual(cards.solverRoute(foam), cards.Route.backend);
        assert.strictEqual(cards.solverRoute({ requires_tag: "foam" }), cards.Route.backend);
        assert.strictEqual(cards.cardTag({ requires_tag: "foam" }), "OpenFOAM");
    });
    test("numpy cards and cards without a tag run in the browser", () => {
        assert.strictEqual(cards.solverRoute(numpy), cards.Route.browser);
        assert.strictEqual(cards.solverRoute({ template: "x = 1" }), cards.Route.browser);
        assert.strictEqual(cards.solverRoute(undefined), cards.Route.browser);
    });
    test("every catalog solver card routes by its requires_tag", () => {
        for (const c of JSON.parse(fs.readFileSync(SOLVERS, "utf8"))) {
            const want = (c.requires_tag || "numpy") === "numpy" ? cards.Route.browser : cards.Route.backend;
            assert.strictEqual(cards.solverRoute(c), want, c.id);
        }
    });
    test("the CLI exposes the same rules the GUI loads", () => {
        const cli = new ZoomyCLI({ storage: {}, pyodide: {} });
        assert.strictEqual(cli.cards.solverRoute(jax), cards.Route.backend);
    });
});

describe("card code", () => {
    test("a backend card without a template contributes no code", () => {
        assert.strictEqual(cards.cardCode(jax, { time_end: 2 }), undefined);
        assert.strictEqual(cards.cardCode(foam, {}), undefined);
    });
    test("a template is filled from the init", () => {
        assert.strictEqual(cards.cardCode(numpy, { time_end: 2 }), "solver = S(time_end=2)");
        assert.strictEqual(cards.fillTemplate("n={n} m={m}", { n: 3 }), "n=3 m={m}");
    });
    test("a browser card with only a class is constructed and described", () => {
        const code = cards.cardCode({ class: "zoomy_core.model.models.SWE" }, { dimension: 1 });
        assert.strictEqual(code, "from zoomy_core.model.models import SWE\n\nmodel = SWE(dimension=1)\ndisplay(model.describe())");
    });
});
