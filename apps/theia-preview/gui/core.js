/**
 * Zoomy Core — pure business logic, no DOM.
 * Works in browser (via <script>) and Node.js (via require/import).
 */

(function (exports) {

/* === Card State === */

function CardState() {
    this.cards = {};
    this.defaults = {};
}

CardState.prototype.init = function (cardId, defaults, tabId, subtab) {
    // NOTE: defaults.snippet is a FILE PATH (e.g. "snippets/foo.py"), not code.
    // Only defaults.template is actual code. Leave code empty for snippet cards;
    // the refresh handler fetches the file contents on demand.
    // A _placeholder default (registered by applySaveData for an override whose
    // card's tab was never opened) is replaced by the real catalog default.
    var defCode = defaults.template || "";
    if (this.defaults[cardId] && this.defaults[cardId]._placeholder &&
            (defaults.title || defaults.template)) {
        delete this.defaults[cardId];
    }
    if (!this.defaults[cardId]) {
        this.defaults[cardId] = {
            tab: tabId || "", subtab: subtab || "",
            title: defaults.title || "", description: defaults.description || "",
            code: defCode, params: {},
            requires_tag: defaults.requires_tag || "",
            class_path: defaults["class"] || "",
            init: defaults.init || {}
        };
    }
    if (!this.cards[cardId]) {
        this.cards[cardId] = {
            tab: tabId || "", subtab: subtab || "",
            title: defaults.title || "", description: defaults.description || "",
            code: defCode, params: {},
            requires_tag: defaults.requires_tag || "",
            class_path: defaults["class"] || "",
            init: defaults.init || {}
        };
    }
    return this.cards[cardId];
};

CardState.prototype.isModified = function (cardId) {
    var cs = this.cards[cardId];
    var cd = this.defaults[cardId];
    if (!cs || !cd) return !!cs;
    return cs.title !== cd.title || cs.description !== cd.description
        || cs.code !== cd.code || JSON.stringify(cs.params) !== JSON.stringify(cd.params);
};

CardState.prototype.get = function (cardId) {
    return this.cards[cardId] || null;
};

CardState.prototype.update = function (cardId, fields) {
    var cs = this.cards[cardId];
    if (!cs) return;
    Object.keys(fields).forEach(function (k) { cs[k] = fields[k]; });
};

/* === Selection Manager === */

function SelectionManager() {
    this.selections = {};
}

SelectionManager.prototype.select = function (tab, cardId) {
    if (cardId === null || cardId === undefined) delete this.selections[tab];
    else this.selections[tab] = cardId;
};

SelectionManager.prototype.selected = function (tab) {
    return this.selections[tab] || null;
};

SelectionManager.prototype.toDict = function () {
    return JSON.parse(JSON.stringify(this.selections));
};

SelectionManager.prototype.clear = function () {
    this.selections = {};
};

/* === Session Manager (per-session selections + card overrides) === */

var _sessionCounter = 0;

function SessionManager() {
    this.sessions = [];
    this.activeId = null;
}

SessionManager.prototype.create = function (title, project) {
    /* Snapshot departing session before creating new one */
    if (project && this.activeId) this.snapshotSession(project);
    /* timestamp + counter + small random suffix. The counter alone would
       clash across tabs (each tab has its own _sessionCounter) which
       matters now that session ids are folder names in IndexedDB. */
    var id = "s-" + Date.now() + "-" + (++_sessionCounter) + "-" +
             Math.random().toString(36).slice(2, 6);
    var session = { id: id, title: title, description: "Simulation session.", selections: {}, cardOverrides: {} };
    /* Clone current selections as the new session's starting point */
    if (project) {
        session.selections = project.selections.toDict();
    }
    this.sessions.push(session);
    this.activeId = id;
    return session;
};

SessionManager.prototype.active = function () {
    for (var i = 0; i < this.sessions.length; i++) {
        if (this.sessions[i].id === this.activeId) return this.sessions[i];
    }
    return null;
};

SessionManager.prototype.get = function (id) {
    for (var i = 0; i < this.sessions.length; i++) {
        if (this.sessions[i].id === id) return this.sessions[i];
    }
    return null;
};

SessionManager.prototype.switchTo = function (id, project) {
    if (this.activeId === id) return;
    /* Snapshot departing session */
    if (project) this.snapshotSession(project);
    this.activeId = id;
    /* Restore arriving session */
    if (project) this.restoreSession(project);
};

SessionManager.prototype.snapshotSession = function (project) {
    var session = this.active();
    if (!session) return;
    session.selections = project.selections.toDict();
    /* Save only modified card state as overrides */
    var overrides = {};
    Object.keys(project.cardState.cards).forEach(function (cardId) {
        if (project.cardState.isModified(cardId)) {
            var cs = project.cardState.cards[cardId];
            overrides[cardId] = { params: JSON.parse(JSON.stringify(cs.params || {})), code: cs.code || "" };
        }
    });
    session.cardOverrides = overrides;
};

SessionManager.prototype.restoreSession = function (project) {
    var session = this.active();
    if (!session) return;
    /* Reset all cards to defaults first */
    Object.keys(project.cardState.defaults).forEach(function (cardId) {
        var def = project.cardState.defaults[cardId];
        var cs = project.cardState.cards[cardId];
        if (cs) {
            cs.params = JSON.parse(JSON.stringify(def.params || {}));
            cs.code = def.code || "";
        }
    });
    /* Apply session-specific overrides */
    var overrides = session.cardOverrides || {};
    Object.keys(overrides).forEach(function (cardId) {
        var cs = project.cardState.cards[cardId];
        if (cs) {
            if (overrides[cardId].params) cs.params = JSON.parse(JSON.stringify(overrides[cardId].params));
            if (overrides[cardId].code) cs.code = overrides[cardId].code;
        }
    });
    /* Restore selections — clear first so the departing session's
       selections don't leak into the arriving one. */
    project.selections.clear();
    var sel = session.selections || {};
    Object.keys(sel).forEach(function (tab) {
        project.selections.select(tab, sel[tab]);
    });
};

/* === Project (orchestrates everything) === */

function Project() {
    this.cardState = new CardState();
    this.selections = new SelectionManager();
    this.sessions = new SessionManager();
}

Project.prototype.status = function () {
    var result = {};
    var tabs = ["model", "mesh", "solver"];
    for (var i = 0; i < tabs.length; i++) {
        var cardId = this.selections.selected(tabs[i]);
        var card = cardId ? this.cardState.get(cardId) : null;
        result[tabs[i]] = card ? card.title : "Not selected";
    }
    var session = this.sessions.active();
    result.session = session ? session.title : "No session";
    return result;
};

Project.prototype.buildCase = function () {
    var model = this.selections.selected("model");
    var mesh = this.selections.selected("mesh");
    var solver = this.selections.selected("solver");

    if (!model || !mesh || !solver) {
        var missing = [];
        if (!model) missing.push("model");
        if (!mesh) missing.push("mesh");
        if (!solver) missing.push("solver");
        throw new Error("Missing selections: " + missing.join(", "));
    }

    var mc = this.cardState.get(model);
    var meshC = this.cardState.get(mesh);

    var meshInit = meshC.params || {};
    var meshSpec;
    if (meshInit.n_cells) {
        meshSpec = { type: "create_1d", domain: [meshInit.x_min || 0, meshInit.x_max || 1], n_cells: meshInit.n_cells };
    } else if (meshInit.nx) {
        meshSpec = { type: "create_2d", x_min: meshInit.x_min || 0, x_max: meshInit.x_max || 1,
                     y_min: meshInit.y_min || 0, y_max: meshInit.y_max || 1, nx: meshInit.nx, ny: meshInit.ny };
    } else {
        meshSpec = { type: "create_1d", domain: [0, 1], n_cells: 100 };
    }

    return {
        version: "1.0",
        model: { class_path: mc.class_path || mc.title, init: mc.init || {}, parameters: mc.params || {} },
        mesh: meshSpec,
        solver: { time_end: 0.1, cfl: 0.45, output_snapshots: 10 }
    };
};

Project.prototype.listCards = function (tab) {
    var result = [];
    var cards = this.cardState.cards;
    Object.keys(cards).forEach(function (id) {
        if (!tab || cards[id].tab === tab) result.push({ id: id, title: cards[id].title, tab: cards[id].tab });
    });
    return result;
};

/* === Safe folder name === */

function safeFolderName(str) {
    return (str || "unnamed").replace(/[^a-zA-Z0-9_-]/g, "_").replace(/_+/g, "_").replace(/^_|_$/g, "").toLowerCase();
}

/* === Project serialization (ZIP-compatible data) === */

Project.prototype.buildSaveData = function () {
    var proj = this;
    /* Snapshot current session before serializing */
    proj.sessions.snapshotSession(proj);

    /* Build per-session serializable data */
    var sessionsData = proj.sessions.sessions.map(function (s) {
        return {
            id: s.id, title: s.title, description: s.description,
            selections: s.selections || {},
            cardOverrides: s.cardOverrides || {}
        };
    });

    var meta = {
        version: "1.1",
        sessions: sessionsData,
        activeSession: proj.sessions.activeId
    };

    /* Emit card files for every session that has overrides */
    var cards = [];
    sessionsData.forEach(function (session) {
        var overrides = session.cardOverrides || {};
        Object.keys(overrides).forEach(function (cardId) {
            var cs = proj.cardState.cards[cardId];
            if (!cs || !cs.tab) return;
            var ov = overrides[cardId];

            var folder = safeFolderName(session.title) + "/" + cs.tab;
            if (cs.subtab) folder += "/" + cs.subtab;
            folder += "/" + safeFolderName(cs.title);

            cards.push({
                folder: folder,
                sessionId: session.id,
                meta: { id: cardId, title: cs.title, description: cs.description, params: ov.params || cs.params, tab: cs.tab, subtab: cs.subtab },
                code: ov.code || null
            });
        });
    });

    return { projectJson: meta, cards: cards };
};

Project.prototype.applySaveData = function (projectJson, cardEntries) {
    var proj = this;

    /* Helper: resolve card ID from meta (by id or title) */
    function resolveCardId(meta) {
        var targetId = meta.id || null;
        if (targetId && !proj.cardState.cards[targetId]) targetId = null;
        /* Title matching only with a real title — placeholder defaults may
           carry empty titles and would soak up every unresolved override. */
        if (!targetId && meta.title) {
            Object.keys(proj.cardState.defaults).forEach(function (cid) {
                if (!targetId && proj.cardState.defaults[cid].title === meta.title) targetId = cid;
            });
        }
        if (!targetId && meta.title) {
            Object.keys(proj.cardState.cards).forEach(function (cid) {
                if (!targetId && proj.cardState.cards[cid].title === meta.title) targetId = cid;
            });
        }
        return targetId;
    }

    /* Version 1.1: per-session selections + overrides */
    if (projectJson.version === "1.1" && projectJson.sessions) {
        this.sessions.sessions = projectJson.sessions.map(function (s) {
            /* Rebuild cardOverrides with resolved card IDs */
            var resolvedOverrides = {};
            var ov = s.cardOverrides || {};
            Object.keys(ov).forEach(function (cardId) {
                var resolved = cardId;
                var matching = cardEntries.filter(function (e) { return e.sessionId === s.id && e.meta.id === cardId; });
                if (!proj.cardState.cards[resolved]) {
                    /* Try to resolve by scanning card entries for this session */
                    if (matching.length > 0) resolved = resolveCardId(matching[0].meta) || cardId;
                }
                if (!proj.cardState.cards[resolved]) {
                    /* Card state not registered yet — states are init'ed
                       LAZILY when the UI first touches a card, so overrides
                       for cards in never-opened tabs (solver, visualization)
                       used to be dropped here and the compose fell back to
                       defaults. Register a placeholder from the save data;
                       CardState.init swaps in the real catalog default when
                       that card is eventually rendered. */
                    var m = (matching[0] && matching[0].meta) || {};
                    proj.cardState.init(resolved,
                        { title: m.title || "", description: m.description || "" },
                        m.tab || "", m.subtab || "");
                    proj.cardState.defaults[resolved]._placeholder = true;
                }
                resolvedOverrides[resolved] = ov[cardId];
            });
            return {
                id: s.id, title: s.title, description: s.description || "",
                selections: s.selections || {},
                cardOverrides: resolvedOverrides
            };
        });
        this.sessions.activeId = projectJson.activeSession || (this.sessions.sessions[0] && this.sessions.sessions[0].id) || "";
        /* Restore the active session's state */
        this.sessions.restoreSession(this);
        return this.sessions.sessions.length;
    }

    /* Version 1.0 (legacy): global selections, flat card list */
    if (projectJson.sessions) {
        this.sessions.sessions = projectJson.sessions.map(function (s) {
            return { id: s.id, title: s.title, description: s.description || "", selections: {}, cardOverrides: {} };
        });
        this.sessions.activeId = projectJson.activeSession || "";
    }
    if (projectJson.selections) {
        var sel = projectJson.selections;
        for (var tab in sel) this.selections.select(tab, sel[tab]);
        /* Store selections in the active session */
        var active = this.sessions.active();
        if (active) active.selections = JSON.parse(JSON.stringify(sel));
    }

    var restored = 0;
    for (var i = 0; i < cardEntries.length; i++) {
        var entry = cardEntries[i];
        var targetId = resolveCardId(entry.meta);
        if (!targetId) continue;

        var cs = this.cardState.cards[targetId];
        cs.tab = entry.meta.tab || cs.tab || "";
        cs.subtab = entry.meta.subtab || cs.subtab || "";
        cs.title = entry.meta.title;
        cs.description = entry.meta.description || "";
        cs.params = entry.meta.params || {};
        if (entry.code) cs.code = entry.code;
        restored++;
    }
    /* Snapshot restored state into active session */
    this.sessions.snapshotSession(this);
    return restored;
};

/* === Initialize from the loaded tab/card config (see _loadAllCards) === */

Project.fromConfig = function (config) {
    var proj = new Project();

    var tabs = config.tabs || [];
    for (var i = 0; i < tabs.length; i++) {
        var tab = tabs[i];
        if (tab.type !== "cards") continue;
        var tabId = tab.id;
        var cards = tab.cards || [];
        var firstId = null;

        for (var j = 0; j < cards.length; j++) {
            var entry = cards[j];
            var cardId = "card-" + entry.id;
            var subtab = entry.subtab || "";
            proj.cardState.init(cardId, entry, tabId, subtab);
            if (!firstId) firstId = cardId;
        }
        if (firstId) proj.selections.select(tabId, firstId);
    }
    /* Create default session with initial selections. The id is
       persisted in localStorage and reused on subsequent visits so
       user-authored cards written under it (keyed by session id inside
       IndexedDB) are still discoverable after a reload. SessionManager.
       create would otherwise mint a fresh random id every page load and
       strand previously-saved user cards under a dead key. */
    var persistedId = null;
    try {
        persistedId = (typeof window !== "undefined" && window.localStorage)
            ? window.localStorage.getItem("zoomy-default-session-id")
            : null;
    } catch (e) { /* localStorage disabled */ }

    if (persistedId) {
        /* Cheap inline equivalent of sessions.create that honours the
           persisted id. Matches the record shape the rest of the
           codebase expects. */
        proj.sessions.sessions.push({
            id: persistedId,
            title: "Default session",
            description: "Simulation session.",
            selections: proj.selections.toDict(),
            cardOverrides: {},
        });
        proj.sessions.activeId = persistedId;
    } else {
        var s = proj.sessions.create("Default session", proj);
        try {
            if (typeof window !== "undefined" && window.localStorage) {
                window.localStorage.setItem("zoomy-default-session-id", s.id);
            }
        } catch (e) { /* localStorage disabled */ }
    }
    return proj;
};

/* === Exports === */

exports.CardState = CardState;
exports.SelectionManager = SelectionManager;
exports.SessionManager = SessionManager;
exports.Project = Project;
exports.safeFolderName = safeFolderName;

})(typeof module !== "undefined" && module.exports ? module.exports : (window.ZoomyCore = {}));
