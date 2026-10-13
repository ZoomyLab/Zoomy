/**
 * Card rules shared by the GUI (zoomy-theia-ext model-config-widget.ts, through `cli.cards`)
 * and the backend end-to-end harness (tests/backend_e2e/compose.mjs): the code a card
 * contributes to a case, and where a case with a given solver card runs.
 *
 * A card's backend is its `requires_tag`; a card without one is the browser's (Pyodide,
 * BROWSER_TAG). The browser runs exactly the cards of its own backend: every other solver
 * card (jax, amrex, OpenFOAM, dmplex, firedrake, ...) is submitted to a connected backend of
 * its tag, whatever code the card carries.
 */

/** The backend the browser's own runtime (Pyodide) provides. */
export const BROWSER_TAG = "numpy";

/** Where Run sends a case. */
export const Route = Object.freeze({ browser: "browser", backend: "backend" });

/** One name per backend: `foam` and `OpenFOAM` are the same backend. */
export function canonTag(tag) {
    return tag === "foam" ? "OpenFOAM" : tag;
}

/** The backend a card needs: its `requires_tag`, else the browser's. */
export function cardTag(card) {
    return canonTag((card && card.requires_tag) || BROWSER_TAG);
}

/** Whether the browser runs the card itself. */
export function runsInBrowser(card) {
    return cardTag(card) === BROWSER_TAG;
}

/** Where Run sends a case whose solver card is `card`. */
export function solverRoute(card) {
    return runsInBrowser(card) ? Route.browser : Route.backend;
}

/** A `{key}` template filled from `init`; unknown keys stay as they are. */
export function fillTemplate(tpl, init) {
    if (!tpl) return "";
    return tpl.replace(/\{(\w+)\}/g, (m, k) => (init && k in init ? String(init[k]) : m));
}

/**
 * The code a card contributes for its effective init: its template, filled; for a browser
 * card with a `class` and no template, an import + construct + describe. A backend card
 * without a template contributes none: the case's Run cell is then the generated runner
 * (`ZoomyCLI._runCode`), which marches the case on that backend's symbolic solver.
 */
export function cardCode(card, init) {
    if (!card) return undefined;
    if (card.template) return fillTemplate(card.template, init);
    if (card.class && runsInBrowser(card)) {
        const dot = card.class.lastIndexOf(".");
        const mod = card.class.slice(0, dot), cls = card.class.slice(dot + 1);
        const kw = Object.entries(init || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
        return `from ${mod} import ${cls}\n\nmodel = ${cls}(${kw})\ndisplay(model.describe())`;
    }
    return undefined;
}
