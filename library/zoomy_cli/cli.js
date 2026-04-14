#!/usr/bin/env node

var fs = require("fs");
var path = require("path");
var ZoomyCore = require(path.join(__dirname, "..", "zoomy_gui", "core.js"));
var JSZip = require("jszip");

var STATE_DIR = ".zoomy";
var STATE_FILE = path.join(STATE_DIR, "state.json");
var CONFIG_FILE = path.join(STATE_DIR, "cards.json");

var DEFAULT_CONFIG_PATHS = [
    path.join(__dirname, "..", "zoomy_gui", "cards.json"),
    "cards.json"
];

function findDefaultConfig() {
    for (var i = 0; i < DEFAULT_CONFIG_PATHS.length; i++) {
        if (fs.existsSync(DEFAULT_CONFIG_PATHS[i])) return DEFAULT_CONFIG_PATHS[i];
    }
    return null;
}

function _loadJsonSafe(p) { try { return JSON.parse(fs.readFileSync(p, "utf8")); } catch (e) { return null; } }

function _loadCardsFolder(baseDir) {
    /* Load tabs.json + default/generated/user per category — mirrors GUI _loadAllCards */
    var tabsPath = path.join(baseDir, "cards", "tabs.json");
    var tabsMeta = _loadJsonSafe(tabsPath);
    if (!tabsMeta) return null;

    var categories = [
        { dir: "models", tabId: "model" },
        { dir: "solvers", tabId: "solver" },
        { dir: "meshes", tabId: "mesh" },
        { dir: "visualizations", tabId: "visualization" }
    ];

    var tabs = [tabsMeta.dashboard || { id: "dashboard", title: "Dashboard", type: "dashboard" }];
    categories.forEach(function (cat) {
        var cardsDir = path.join(baseDir, "cards", cat.dir);
        var def = _loadJsonSafe(path.join(cardsDir, "default.json")) || [];
        var gen = _loadJsonSafe(path.join(cardsDir, "generated.json")) || [];
        var usr = _loadJsonSafe(path.join(cardsDir, "user.json")) || [];
        var seen = {}, merged = [];
        [def, gen, usr].forEach(function (list) {
            list.forEach(function (c) { if (!seen[c.id]) { seen[c.id] = true; merged.push(c); } });
        });
        var meta = tabsMeta[cat.tabId] || { id: cat.tabId, title: cat.dir, type: "cards" };
        meta.cards = merged;
        tabs.push(meta);
    });
    return { tabs: tabs };
}

function loadProject() {
    /* Try cards/ folder first, fall back to cards.json */
    var guiDir = path.join(__dirname, "..", "zoomy_gui");
    var config = _loadCardsFolder(guiDir);
    if (!config) {
        var configPath = fs.existsSync(CONFIG_FILE) ? CONFIG_FILE : findDefaultConfig();
        if (!configPath) { console.error("No project found. Run 'zoomy start' first."); process.exit(1); }
        config = JSON.parse(fs.readFileSync(configPath, "utf8"));
    }
    var proj = ZoomyCore.Project.fromConfig(config);
    loadState(proj);
    return proj;
}

var _backends = {};

function saveState(proj) {
    if (!fs.existsSync(STATE_DIR)) fs.mkdirSync(STATE_DIR, { recursive: true });
    fs.writeFileSync(STATE_FILE, JSON.stringify({
        selections: proj.selections.toDict(),
        sessions: proj.sessions.sessions,
        activeSession: proj.sessions.activeId,
        backends: _backends
    }, null, 2));
}

function loadState(proj) {
    if (!fs.existsSync(STATE_FILE)) return;
    var state = JSON.parse(fs.readFileSync(STATE_FILE, "utf8"));
    if (state.selections) {
        Object.keys(state.selections).forEach(function (tab) {
            proj.selections.select(tab, state.selections[tab]);
        });
    }
    if (state.sessions && state.sessions.length > 0) {
        proj.sessions.sessions = state.sessions;
        proj.sessions.activeId = state.activeSession || state.sessions[0].id;
    }
    if (state.backends) _backends = state.backends;
}

function isBackendConnected(tag) {
    if (tag === "numpy") return true;
    return !!_backends[tag];
}

function shortId(cardId) {
    return cardId.replace(/^card-/, "").replace(/^(mesh|solver|vis)-/, "");
}
function fullId(id) {
    if (id.indexOf("card-") === 0) return id;
    if (id.indexOf("mesh-") === 0 || id.indexOf("solver-") === 0 || id.indexOf("vis-") === 0) return "card-" + id;
    return "card-" + id;
}
function resolveTab(tab) {
    var aliases = { visu: "visualization", vis: "visualization" };
    return aliases[tab] || tab;
}
function resolveCardId(proj, tab, name) {
    tab = resolveTab(tab);
    var direct = "card-" + name;
    if (proj.cardState.get(direct)) return direct;
    var prefixes = { mesh: "mesh-", solver: "solver-", visualization: "vis-" };
    var prefix = prefixes[tab];
    if (prefix) {
        var prefixed = "card-" + prefix + name;
        if (proj.cardState.get(prefixed)) return prefixed;
    }
    return null;
}

/* === Commands === */

function installCompletions() {
    var shell = (process.env.SHELL || "").split("/").pop();
    var home = process.env.HOME || "";
    if (!home) return;

    var readline = require("readline");
    var rl = readline.createInterface({ input: process.stdin, output: process.stdout });

    return new Promise(function (resolve) {
        if (shell === "zsh") {
            var compDir = path.join(home, ".zfunc");
            var compFile = path.join(compDir, "_zoomy");
            var rcLine = 'fpath=(~/.zfunc $fpath)\nautoload -Uz compinit && compinit';

            if (!fs.existsSync(compDir)) fs.mkdirSync(compDir, { recursive: true });
            var script = [
                '#compdef zoomy',
                '_zoomy() {',
                '  local -a commands tabs session_cmds',
                '  commands=(start overview status list select show session connect disconnect backends run watch jobs case save load)',
                '  tabs=(model mesh solver visu)',
                '  session_cmds=(new switch rename list)',
                '  if (( CURRENT == 2 )); then',
                '    _describe "command" commands',
                '  elif (( CURRENT == 3 )); then',
                '    case "${words[2]}" in',
                '      list|overview|select|show) _describe "tab" tabs ;;',
                '      session) _describe "subcommand" session_cmds ;;',
                '      load) _files -g "*.zip" ;;',
                '    esac',
                '  elif (( CURRENT == 4 )); then',
                '    case "${words[2]}" in',
                '      select|show)',
                '        local -a cards',
                '        cards=(${(f)"$(zoomy list ${words[3]} 2>/dev/null | awk \'{print $1}\')"})',
                '        _describe "card" cards',
                '      ;;',
                '      session)',
                '        if [[ "${words[3]}" == "switch" ]]; then',
                '          local -a sessions',
                '          sessions=(${(f)"$(zoomy session list 2>/dev/null | sed \'s/ \\[active\\]//\' | sed \'s/^  //\')"})',
                '          _describe "session" sessions',
                '        fi',
                '      ;;',
                '    esac',
                '  fi',
                '}',
                'compdef _zoomy zoomy'
            ].join("\n") + "\n";
            fs.writeFileSync(compFile, script);
            console.log("  Completion script written to " + compFile);

            var zshrc = path.join(home, ".zshrc");
            var needsRc = fs.existsSync(zshrc) && fs.readFileSync(zshrc, "utf8").indexOf(".zfunc") === -1;

            if (needsRc) {
                console.log("\n  To enable tab completion, add this to your ~/.zshrc:\n");
                console.log("    " + rcLine.replace(/\n/g, "\n    "));
                console.log("");
                rl.question("  Add it automatically? [y/N] ", function (answer) {
                    if (answer.toLowerCase() === "y") {
                        fs.appendFileSync(zshrc, "\n" + rcLine + "\n");
                        console.log("  Added. Restart your shell to activate.\n");
                    } else {
                        console.log("  Skipped. Add it manually when ready.\n");
                    }
                    rl.close();
                    resolve();
                });
                return;
            }
        } else if (shell === "bash") {
            var bashCompDir = path.join(home, ".bash_completion.d");
            var compFile = path.join(bashCompDir, "zoomy");
            var rcLine = '[ -f ~/.bash_completion.d/zoomy ] && source ~/.bash_completion.d/zoomy';

            if (!fs.existsSync(bashCompDir)) fs.mkdirSync(bashCompDir, { recursive: true });
            var script = [
                '_zoomy_completions() {',
                '  local cur="${COMP_WORDS[COMP_CWORD]}" prev="${COMP_WORDS[COMP_CWORD-1]}"',
                '  local cmds="start overview status list select show session connect disconnect backends run watch jobs case save load"',
                '  local tabs="model mesh solver visu"',
                '  if [ "$COMP_CWORD" = 1 ]; then COMPREPLY=($(compgen -W "$cmds" -- "$cur"))',
                '  elif [ "$prev" = "list" ] || [ "$prev" = "overview" ] || [ "$prev" = "select" ] || [ "$prev" = "show" ]; then COMPREPLY=($(compgen -W "$tabs" -- "$cur"))',
                '  elif [ "$prev" = "session" ]; then COMPREPLY=($(compgen -W "new switch list" -- "$cur"))',
                '  elif [ "$prev" = "model" ] || [ "$prev" = "mesh" ] || [ "$prev" = "solver" ] || [ "$prev" = "visu" ]; then',
                '    COMPREPLY=($(compgen -W "$(zoomy list $prev 2>/dev/null | awk \'{print $1}\')" -- "$cur"))',
                '  elif [ "$prev" = "load" ]; then COMPREPLY=($(compgen -f -X "!*.zip" -- "$cur"))',
                '  fi',
                '}',
                'complete -F _zoomy_completions zoomy'
            ].join("\n") + "\n";
            fs.writeFileSync(compFile, script);
            console.log("  Completion script written to " + compFile);

            var bashrc = path.join(home, ".bashrc");
            var needsRc = fs.existsSync(bashrc) && fs.readFileSync(bashrc, "utf8").indexOf("bash_completion.d/zoomy") === -1;

            if (needsRc) {
                console.log("\n  To enable tab completion, add this to your ~/.bashrc:\n");
                console.log("    " + rcLine);
                console.log("");
                rl.question("  Add it automatically? [y/N] ", function (answer) {
                    if (answer.toLowerCase() === "y") {
                        fs.appendFileSync(bashrc, "\n" + rcLine + "\n");
                        console.log("  Added. Restart your shell to activate.\n");
                    } else {
                        console.log("  Skipped. Add it manually when ready.\n");
                    }
                    rl.close();
                    resolve();
                });
                return;
            }
        }
        rl.close();
        resolve();
    });
}

async function cmdStart() {
    if (fs.existsSync(STATE_DIR)) {
        console.log("Project already initialized in .zoomy/");
        return;
    }
    fs.mkdirSync(STATE_DIR, { recursive: true });
    var srcConfig = findDefaultConfig();
    if (srcConfig) fs.copyFileSync(srcConfig, CONFIG_FILE);

    var proj = loadProject();
    saveState(proj);
    ["model", "mesh", "solver", "visualization"].forEach(function (tab) {
        fs.mkdirSync(path.join(STATE_DIR, tab), { recursive: true });
    });

    console.log("\n  Project initialized in .zoomy/\n");

    await installCompletions();

    console.log("  Run 'zoomy overview' to see available cards.\n");
}

function formatCardLine(c, proj, sel, indent) {
    var sn = shortId(c.id);
    var cs = proj.cardState.get(c.id);
    var marker = (sel[c.tab] === c.id) ? " [selected]" : "";
    var mod = proj.cardState.isModified(c.id) ? " (modified)" : "";
    var conn = "";
    if (cs && cs.requires_tag) {
        conn = isBackendConnected(cs.requires_tag) ? " \u2713" : " \u2717";
    }
    return indent + sn.padEnd(22) + c.title + conn + marker + mod;
}

function cmdOverview(proj, tab) {
    if (tab) tab = resolveTab(tab);
    var allTabs = tab ? [tab] : ["model", "mesh", "solver", "visualization"];
    var sel = proj.selections.toDict();
    var config = fs.existsSync(CONFIG_FILE) ? JSON.parse(fs.readFileSync(CONFIG_FILE, "utf8")) : null;

    allTabs.forEach(function (t) {
        var cards = proj.listCards(t);
        if (cards.length === 0) return;

        var tabConfig = config ? config.tabs.find(function (tc) { return tc.id === t; }) : null;
        var subtabs = tabConfig && tabConfig.subtabs ? tabConfig.subtabs : [];

        console.log("\n  " + t.toUpperCase());
        console.log("  " + "-".repeat(50));

        if (subtabs.length > 0) {
            subtabs.forEach(function (st) {
                var stCards = cards.filter(function (c) {
                    var cs = proj.cardState.get(c.id);
                    return cs && cs.subtab === st.id;
                });
                if (stCards.length === 0) return;
                console.log("    [" + st.title + "]");
                stCards.forEach(function (c) { console.log(formatCardLine(c, proj, sel, "      ")); });
            });
            var noSub = cards.filter(function (c) { var cs = proj.cardState.get(c.id); return !cs || !cs.subtab; });
            noSub.forEach(function (c) { console.log(formatCardLine(c, proj, sel, "      ")); });
        } else {
            cards.forEach(function (c) { console.log(formatCardLine(c, proj, sel, "    ")); });
        }
    });

    var s = proj.sessions.active();
    console.log("\n  SESSION: " + (s ? s.title : "none"));
    console.log("");
}

function cmdStatus(proj) {
    var s = proj.status();
    console.log("");
    Object.keys(s).forEach(function (k) {
        console.log("  " + k.padEnd(10) + ": " + s[k]);
    });
    console.log("");
}

function cmdList(proj, tab) {
    if (!tab) {
        console.log("Usage: zoomy list <tab>");
        console.log("  Tabs: model, mesh, solver, visu");
        return;
    }
    tab = resolveTab(tab);
    var cards = proj.listCards(tab);
    if (cards.length === 0) { console.log("No cards for '" + tab + "'"); return; }
    var sel = proj.selections.toDict();
    cards.forEach(function (c) {
        var marker = (sel[c.tab] === c.id) ? " *" : "";
        console.log("  " + shortId(c.id).padEnd(22) + c.title + marker);
    });
}

function cmdSelect(proj, tab, name) {
    if (!tab || !name) {
        console.log("Usage: zoomy select <tab> <name>");
        console.log("  Tabs: model, mesh, solver, visu");
        console.log("  Example: zoomy select model smt");
        return;
    }
    tab = resolveTab(tab);
    var cardId = resolveCardId(proj, tab, name);
    if (!cardId) {
        console.log("Unknown card: " + name);
        var available = proj.listCards(tab).map(function (c) { return shortId(c.id); });
        if (available.length) console.log("Available in " + tab + ": " + available.join(", "));
        return;
    }
    var card = proj.cardState.get(cardId);
    if (card.requires_tag && !isBackendConnected(card.requires_tag)) {
        console.log("Cannot select '" + card.title + "': backend '" + card.requires_tag + "' not connected.");
        console.log("Connect it first: zoomy connect <url>");
        return;
    }
    proj.selections.select(tab, cardId);
    saveState(proj);
    console.log("Selected '" + card.title + "' in " + tab);
}

function cmdShow(proj, tab, name) {
    if (!tab || !name) {
        console.log("Usage: zoomy show <tab> <name>");
        console.log("  Example: zoomy show model smt");
        return;
    }
    tab = resolveTab(tab);
    var cardId = resolveCardId(proj, tab, name);
    if (!cardId) {
        console.log("Unknown card: " + name);
        var available = proj.listCards(tab).map(function (c) { return shortId(c.id); });
        if (available.length) console.log("Available in " + tab + ": " + available.join(", "));
        return;
    }
    var card = proj.cardState.get(cardId);
    console.log("");
    console.log("  Title:     " + card.title);
    console.log("  Name:      " + shortId(cardId));
    console.log("  Tab:       " + card.tab);
    if (card.subtab) console.log("  Subtab:    " + card.subtab);
    if (card.description) console.log("  Description: " + card.description.replace(/<[^>]*>/g, "").substring(0, 120));
    if (card.code) {
        console.log("  Code:");
        card.code.split("\n").forEach(function (l) { console.log("    " + l); });
    }
    console.log("");
}

function cmdSession(proj, subcmd, arg) {
    if (!subcmd) {
        var s = proj.sessions.active();
        console.log("Active session: " + (s ? s.title : "none"));
        console.log("\nAll sessions:");
        proj.sessions.sessions.forEach(function (s) {
            var marker = s.id === proj.sessions.activeId ? " [active]" : "";
            console.log("  " + s.title + marker);
        });
        return;
    }
    if (subcmd === "new") {
        var title = arg || ("Session " + (proj.sessions.sessions.length + 1));
        proj.sessions.create(title);
        saveState(proj);
        console.log("Created and switched to '" + title + "'");
    } else if (subcmd === "switch") {
        if (!arg) { console.log("Usage: zoomy session switch <name>"); return; }
        var found = proj.sessions.sessions.find(function (s) { return s.title === arg; });
        if (!found) {
            console.log("Unknown session: " + arg);
            proj.sessions.sessions.forEach(function (s) { console.log("  " + s.title); });
            return;
        }
        proj.sessions.switchTo(found.id);
        saveState(proj);
        console.log("Switched to '" + found.title + "'");
    } else if (subcmd === "list") {
        proj.sessions.sessions.forEach(function (s) {
            var marker = s.id === proj.sessions.activeId ? " [active]" : "";
            console.log("  " + s.title + marker);
        });
    } else if (subcmd === "rename") {
        if (!arg) { console.log("Usage: zoomy session rename <new_name>"); console.log("       zoomy session rename <old_name> <new_name>"); return; }
        var parts = arg.split(" ");
        var target, newName;
        if (parts.length >= 2) {
            var splitIdx = arg.lastIndexOf(" ");
            var possibleOld = arg.substring(0, splitIdx);
            var possibleNew = arg.substring(splitIdx + 1);
            var found = proj.sessions.sessions.find(function (s) { return s.title === possibleOld; });
            if (found) {
                target = found;
                newName = possibleNew;
            } else {
                target = proj.sessions.active();
                newName = arg;
            }
        } else {
            target = proj.sessions.active();
            newName = arg;
        }
        if (!target) { console.log("No session found"); return; }
        var old = target.title;
        target.title = newName;
        saveState(proj);
        console.log("Renamed '" + old + "' to '" + newName + "'");
    } else {
        console.log("Usage: zoomy session [new|switch|rename|list]");
    }
}

function cmdConnect(proj, url) {
    if (!url) { console.log("Usage: zoomy connect <url>"); return; }
    url = url.replace(/\/+$/, "");
    var http = url.indexOf("https") === 0 ? require("https") : require("http");
    return new Promise(function (resolve) {
        var req = http.get(url + "/api/v1/health", { timeout: 3000 }, function (res) {
            var body = "";
            res.on("data", function (d) { body += d; });
            res.on("end", function () {
                try {
                    var data = JSON.parse(body);
                    if (data.status === "ok") {
                        var tag = data.tag || "unknown";
                        _backends[tag] = url;
                        saveState(proj);
                        console.log("Connected to '" + tag + "' at " + url);
                    } else {
                        console.log("Server responded but status is not ok");
                    }
                } catch (e) { console.log("Invalid response from " + url); }
                resolve();
            });
        });
        req.on("error", function () { console.log("Cannot reach " + url); resolve(); });
        req.on("timeout", function () { req.destroy(); console.log("Timeout connecting to " + url); resolve(); });
    });
}

function cmdDisconnect(proj, tag) {
    if (!tag) { console.log("Usage: zoomy disconnect <tag>"); return; }
    if (!_backends[tag]) { console.log("Not connected to '" + tag + "'"); return; }
    delete _backends[tag];
    saveState(proj);
    console.log("Disconnected from '" + tag + "'");
}

function cmdBackends() {
    console.log("\n  BACKENDS");
    console.log("  " + "-".repeat(40));
    console.log("    numpy".padEnd(20) + "(built-in, always available)");
    Object.keys(_backends).forEach(function (tag) {
        console.log("    " + tag.padEnd(20) + _backends[tag]);
    });
    console.log("");
}

function resolveBackendUrl(proj) {
    var solverCardId = proj.selections.selected("solver");
    var solverCard = solverCardId ? proj.cardState.get(solverCardId) : null;
    var tag = solverCard && solverCard.requires_tag ? solverCard.requires_tag : "numpy";
    var url = _backends[tag];
    return { tag: tag, url: url };
}

async function pollJob(url, jobId) {
    var http = url.indexOf("https") === 0 ? require("https") : require("http");
    console.log("Watching job " + jobId + "...");
    var done = false;
    while (!done) {
        await new Promise(function (r) { setTimeout(r, 2000); });
        var status = await new Promise(function (resolve) {
            http.get(url + "/api/v1/jobs/" + jobId, function (res) {
                var data = "";
                res.on("data", function (d) { data += d; });
                res.on("end", function () { try { resolve(JSON.parse(data)); } catch (e) { resolve({}); } });
            }).on("error", function () { resolve({}); });
        });
        if (status.status === "complete") {
            process.stdout.write("\n");
            console.log("Complete!");
            done = true;
        } else if (status.status === "failed") {
            process.stdout.write("\n");
            console.log("Failed: " + (status.error || "").substring(0, 200));
            done = true;
        } else if (status.progress) {
            var p = status.progress;
            process.stdout.write("\r  t=" + (p.time || 0).toFixed(4) + " / " + (p.time_end || 0).toFixed(4) + "  iter=" + (p.iteration || 0) + "    ");
        }
    }
}

async function cmdRunLocal(proj) {
    var zcase;
    try { zcase = proj.buildCase(); }
    catch (err) { console.log("Error: " + err.message); return; }

    /* Get model code: user-edited > template > auto-generated */
    var modelCardId = proj.selections.selected("model");
    var modelState = modelCardId ? proj.cardState.get(modelCardId) : null;
    var modelCode;
    if (modelState && modelState.code) {
        modelCode = modelState.code;
    } else {
        var cp = zcase.model.class_path || "";
        var parts = cp.split(".");
        var cls = parts[parts.length - 1];
        var mod = parts.slice(0, -1).join(".");
        var initKw = zcase.model.init || {};
        var kwargs = Object.keys(initKw).map(function (k) {
            var v = initKw[k];
            return k + "=" + (typeof v === "string" ? "'" + v + "'" : v);
        }).join(", ");
        modelCode = "from " + mod + " import " + cls + "\nmodel = " + cls + "(" + kwargs + ")\n";
    }

    /* Mesh */
    var meshCode;
    var ms = zcase.mesh;
    if (ms.type === "create_2d") {
        meshCode = "from zoomy_core.mesh import BaseMesh\nmesh = BaseMesh.create_2d((" +
            ms.x_min + ", " + ms.x_max + ", " + ms.y_min + ", " + ms.y_max + "), nx=" + ms.nx + ", ny=" + ms.ny + ")\n";
    } else if (ms.type === "create_3d") {
        meshCode = "from zoomy_core.mesh import BaseMesh\nmesh = BaseMesh.create_3d((" +
            ms.x_min + ", " + ms.x_max + ", " + ms.y_min + ", " + ms.y_max + ", " +
            ms.z_min + ", " + ms.z_max + "), nx=" + ms.nx + ", ny=" + ms.ny + ", nz=" + ms.nz + ")\n";
    } else {
        var dom = ms.domain || [0, 1];
        meshCode = "from zoomy_core.mesh import BaseMesh\nmesh = BaseMesh.create_1d(domain=(" + dom[0] + ", " + dom[1] + "), n_inner_cells=" + (ms.n_cells || 100) + ")\n";
    }

    /* Solver */
    var ss = zcase.solver || {};
    var solverCode = "from zoomy_core.fvm.solver_numpy import FreeSurfaceFlowSolver, HyperbolicSolver\n" +
        "import zoomy_core.fvm.timestepping as ts\n" +
        "keys = list(model.variables.keys()) if hasattr(model.variables, 'keys') else []\n" +
        "Solver = FreeSurfaceFlowSolver if ('h' in keys and 'b' in keys) else HyperbolicSolver\n" +
        "solver = Solver(time_end=" + (ss.time_end || 0.1) + ", compute_dt=ts.adaptive(CFL=" + (ss.cfl || 0.3) + "))\n" +
        "Q, Qaux = solver.solve(mesh, model, write_output=False)\n" +
        "print(f'Done: {Q.shape[0]} variables, {mesh.n_inner_cells} cells')\n";

    var script = modelCode + "\n" + meshCode + "\n" + solverCode;

    console.log("Running locally with Python...\n");

    var child = require("child_process");
    var proc = child.spawn("python", ["-c", script], { stdio: "inherit" });
    return new Promise(function (resolve) {
        proc.on("close", function (code) {
            if (code !== 0) console.log("\nProcess exited with code " + code);
            resolve();
        });
    });
}

async function cmdRun(proj, opts) {
    var wait = opts.indexOf("--wait") !== -1;
    var local = opts.indexOf("--local") !== -1;

    if (local) return cmdRunLocal(proj);

    var backend = resolveBackendUrl(proj);

    if (!backend.url) {
        console.log("Backend '" + backend.tag + "' not connected.");
        console.log("Use: zoomy run --local    (run with local Python)");
        console.log("Or:  zoomy connect <url>  (connect to server)");
        return;
    }

    var zcase;
    try { zcase = proj.buildCase(); }
    catch (err) { console.log("Error: " + err.message); return; }

    console.log("Submitting to " + backend.tag + " at " + backend.url + "...");

    var http = backend.url.indexOf("https") === 0 ? require("https") : require("http");
    var body = JSON.stringify(zcase);

    var jobId = await new Promise(function (resolve) {
        var req = http.request(backend.url + "/api/v1/jobs", {
            method: "POST", headers: { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(body) }
        }, function (res) {
            var data = "";
            res.on("data", function (d) { data += d; });
            res.on("end", function () {
                try { resolve(JSON.parse(data).job_id); }
                catch (e) { console.log("Bad response: " + data); resolve(null); }
            });
        });
        req.on("error", function (e) { console.log("Failed: " + e.message); resolve(null); });
        req.write(body);
        req.end();
    });

    if (!jobId) return;
    console.log("Job submitted: " + jobId);

    if (!wait) {
        console.log("Check status:  zoomy jobs " + jobId);
        console.log("Watch live:    zoomy watch " + jobId);
        return;
    }

    await pollJob(backend.url, jobId);
}

async function cmdWatch(proj, jobId) {
    if (!jobId) {
        console.log("Usage: zoomy watch <job_id>");
        return;
    }
    var backend = resolveBackendUrl(proj);
    if (!backend.url) {
        console.log("Backend '" + backend.tag + "' not connected. Run: zoomy connect <url>");
        return;
    }
    await pollJob(backend.url, jobId);
}

async function cmdJobs(proj, jobId) {
    var backend = resolveBackendUrl(proj);
    if (!backend.url) { console.log("No backend connected for '" + backend.tag + "'"); return; }
    var url = backend.url;

    var http = url.indexOf("https") === 0 ? require("https") : require("http");
    var endpoint = jobId ? url + "/api/v1/jobs/" + jobId : url + "/api/v1/jobs";

    var data = await new Promise(function (resolve) {
        http.get(endpoint, function (res) {
            var body = "";
            res.on("data", function (d) { body += d; });
            res.on("end", function () { try { resolve(JSON.parse(body)); } catch (e) { resolve(null); } });
        }).on("error", function () { resolve(null); });
    });

    if (!data) { console.log("Cannot reach backend"); return; }

    if (jobId) {
        console.log("");
        console.log("  Job:     " + (data.job_id || jobId));
        console.log("  Status:  " + (data.status || "unknown"));
        if (data.progress) {
            var p = data.progress;
            console.log("  Time:    " + (p.time || 0).toFixed(4) + " / " + (p.time_end || 0).toFixed(4));
            console.log("  Iter:    " + (p.iteration || 0));
        }
        if (data.error) console.log("  Error:   " + data.error.substring(0, 200));
        console.log("");
    } else {
        if (!Array.isArray(data) || data.length === 0) { console.log("No jobs"); return; }
        console.log("");
        data.forEach(function (j) {
            var status = (j.status || "?").padEnd(10);
            var progress = j.progress ? "t=" + (j.progress.time || 0).toFixed(4) : "";
            console.log("  " + (j.job_id || "?").padEnd(12) + status + progress);
        });
        console.log("");
    }
}

function cmdCase(proj) {
    try {
        console.log(JSON.stringify(proj.buildCase(), null, 2));
    } catch (err) {
        console.log("Error: " + err.message);
    }
}

async function cmdSave(proj, filepath) {
    filepath = filepath || "zoomy-project.zip";
    var data = proj.buildSaveData();
    var zip = new JSZip();
    zip.file("project.json", JSON.stringify(data.projectJson, null, 2));
    data.cards.forEach(function (c) {
        zip.file(c.folder + "/card.json", JSON.stringify(c.meta, null, 2));
        if (c.code) zip.file(c.folder + "/code.py", c.code);
    });
    var buf = await zip.generateAsync({ type: "nodebuffer" });
    fs.writeFileSync(filepath, buf);
    console.log("Saved to " + filepath + " (" + data.cards.length + " modified cards)");
}

async function cmdLoad(proj, filepath) {
    if (!filepath) { console.log("Usage: zoomy load <path.zip>"); return; }
    if (!fs.existsSync(filepath)) { console.log("File not found: " + filepath); return; }
    var buf = fs.readFileSync(filepath);
    var zip = await JSZip.loadAsync(buf);
    var projectJson = {};
    var cardEntries = [];
    var names = Object.keys(zip.files);

    for (var i = 0; i < names.length; i++) {
        if (zip.files[names[i]].dir) continue;
        if (names[i].endsWith("project.json"))
            projectJson = JSON.parse(await zip.files[names[i]].async("string"));
    }
    var folders = {};
    for (var i = 0; i < names.length; i++) {
        if (zip.files[names[i]].dir || names[i].endsWith("project.json")) continue;
        var parts = names[i].split("/");
        var fn = parts[parts.length - 1];
        var fk = parts.slice(0, -1).join("/");
        if (!folders[fk]) folders[fk] = {};
        if (fn === "card.json") folders[fk].meta = JSON.parse(await zip.files[names[i]].async("string"));
        else if (fn === "code.py") folders[fk].code = await zip.files[names[i]].async("string");
    }
    Object.keys(folders).forEach(function (f) {
        if (folders[f].meta) cardEntries.push({ meta: folders[f].meta, code: folders[f].code || null });
    });

    var count = proj.applySaveData(projectJson, cardEntries);
    saveState(proj);
    console.log("Loaded " + count + " cards from " + filepath);
}

/* === Completion (zsh + bash) === */

function cmdCompletion(shell) {
    shell = shell || (process.env.SHELL || "").indexOf("zsh") !== -1 ? "zsh" : "bash";

    if (shell === "zsh") {
        console.log([
            '#compdef zoomy',
            '_zoomy() {',
            '  local -a commands tabs',
            '  commands=(start overview status list select show session connect disconnect backends run watch jobs case save load)',
            '  tabs=(model mesh solver visualization)',
            '  if (( CURRENT == 2 )); then',
            '    _describe "command" commands',
            '  elif (( CURRENT == 3 )); then',
            '    case "${words[2]}" in',
            '      list|overview|select) _describe "tab" tabs ;;',
            '      show) _describe "tab or card" tabs ;;',
            '      load) _files -g "*.zip" ;;',
            '    esac',
            '  elif (( CURRENT == 4 )); then',
            '    case "${words[2]}" in',
            '      select|show)',
            '        local -a cards',
            '        cards=(${(f)"$(zoomy list ${words[3]} 2>/dev/null | awk \'{print $1}\')"})',
            '        _describe "card" cards',
            '      ;;',
            '    esac',
            '  fi',
            '}',
            'compdef _zoomy zoomy'
        ].join("\n"));
    } else {
        console.log([
            '_zoomy_completions() {',
            '  local cur="${COMP_WORDS[COMP_CWORD]}"',
            '  local prev="${COMP_WORDS[COMP_CWORD-1]}"',
            '  local cmds="start overview status list select show session connect disconnect backends run watch jobs case save load"',
            '  local tabs="model mesh solver visualization"',
            '  if [ "$COMP_CWORD" = 1 ]; then',
            '    COMPREPLY=($(compgen -W "$cmds" -- "$cur"))',
            '  elif [ "$prev" = "list" ] || [ "$prev" = "overview" ] || [ "$prev" = "select" ]; then',
            '    COMPREPLY=($(compgen -W "$tabs" -- "$cur"))',
            '  elif [ "$prev" = "model" ] || [ "$prev" = "mesh" ] || [ "$prev" = "solver" ] || [ "$prev" = "visualization" ]; then',
            '    local cards=$(zoomy list "$prev" 2>/dev/null | awk \'{print $1}\')',
            '    COMPREPLY=($(compgen -W "$cards" -- "$cur"))',
            '  elif [ "$prev" = "load" ]; then',
            '    COMPREPLY=($(compgen -f -X "!*.zip" -- "$cur"))',
            '  fi',
            '}',
            'complete -F _zoomy_completions zoomy'
        ].join("\n"));
    }
}

/* === Main === */

async function main() {
    var args = process.argv.slice(2);
    var cmd = args[0];

    if (!cmd || cmd === "--help" || cmd === "-h") {
        console.log([
            "",
            "  zoomy — Simulation project manager",
            "",
            "  GETTING STARTED:",
            "    zoomy start                     Initialize project in current directory",
            "    zoomy overview                   Show all cards and selections",
            "",
            "  SELECTION:  (tabs: model, mesh, solver, visu)",
            "    zoomy select <tab> <name>        Select a card",
            "    zoomy status                     Show current selections",
            "",
            "  SESSIONS:",
            "    zoomy session                    Show active session",
            "    zoomy session new [name]         Create new session",
            "    zoomy session switch <name>      Switch session",
            "    zoomy session rename <name>      Rename active session",
            "    zoomy session list               List all sessions",
            "",
            "  BACKENDS:",
            "    zoomy connect <url>              Connect to a solver backend",
            "    zoomy disconnect <tag>           Disconnect a backend",
            "",
            "  RUN:",
            "    zoomy run                        Submit simulation to backend",
            "    zoomy run --local                Run locally with Python (no server)",
            "    zoomy run --wait                 Submit and wait for completion",
            "    zoomy watch <job_id>             Attach to job with live progress",
            "    zoomy jobs                       List all jobs",
            "    zoomy jobs <job_id>              Show job status",
            "",
            "  INSPECTION:",
            "    zoomy list <tab>                 List card names",
            "    zoomy show <tab> <name>          Show card details",
            "    zoomy case                       Print simulation case JSON",
            "",
            "  PROJECT:",
            "    zoomy save [path.zip]            Save modified cards to zip",
            "    zoomy load <path.zip>            Load project from zip",
            "",
            ""
        ].join("\n"));
        return;
    }

    if (cmd === "start") { await cmdStart(); return; }

    var proj = loadProject();

    if (cmd === "overview") cmdOverview(proj, args[1]);
    else if (cmd === "status") cmdStatus(proj);
    else if (cmd === "list") cmdList(proj, args[1]);
    else if (cmd === "select") cmdSelect(proj, args[1], args[2]);
    else if (cmd === "show") cmdShow(proj, args[1], args[2]);
    else if (cmd === "session") cmdSession(proj, args[1], args.slice(2).join(" "));
    else if (cmd === "connect") await cmdConnect(proj, args[1]);
    else if (cmd === "disconnect") cmdDisconnect(proj, args[1]);
    else if (cmd === "run") await cmdRun(proj, args.slice(1));
    else if (cmd === "watch") await cmdWatch(proj, args[1]);
    else if (cmd === "jobs") await cmdJobs(proj, args[1]);
    else if (cmd === "case") cmdCase(proj);
    else if (cmd === "save") await cmdSave(proj, args[1]);
    else if (cmd === "load") await cmdLoad(proj, args[1]);
    else console.log("Unknown command: " + cmd + ". Run 'zoomy --help'.");
}

main().catch(function (err) { console.error("Error: " + err.message); process.exit(1); });
