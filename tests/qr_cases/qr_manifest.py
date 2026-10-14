"""The thesis QR targets and the GUI sessions behind them.

The manifest is the thesis's `scripts/animations.yml` (key -> URL). `QR_MANIFEST` names it
explicitly; else `thesis/scripts/animations.yml` of this checkout; else the snapshot next to
this file (CI: the thesis repo is private). A session is composed into the cells the GUI
writes for it (`materializeSessions` + `ZoomyCLI._caseCells`): model, steps, mesh, solver
settings, run, visualization.
"""
from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlsplit

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GUI = ROOT / "library" / "zoomy_gui"
PROJECTS = GUI / "projects"
THESIS_MANIFEST = ROOT / "thesis" / "scripts" / "animations.yml"
SNAPSHOT = HERE / "animations.yml"
SITE = "https://zoomylab.github.io/Zoomy/"
#: Solver tags the browser runs itself; any other tag is submitted to a backend.
BROWSER_LOCAL = (None, "numpy")
#: Where a solving session's store must end (`session_runner.end_time`): the solver card's
#: `time_end` parameter unless listed. A string is an expression over the cells' names (the
#: settings, the model cell's constants); None: nothing is solved.
END_TIME = {("bingham-session.zip", 0): "settings['time_end'] * tscale",    # time_end is t' = t / tscale
            ("coupling-session.zip", 0): None}                              # replays a recorded run


def manifest_path() -> Path:
    env = os.environ.get("QR_MANIFEST")
    if env:
        return Path(env)
    return THESIS_MANIFEST if THESIS_MANIFEST.exists() else SNAPSHOT


def load_manifest(path=None) -> dict:
    return yaml.safe_load(Path(path or manifest_path()).read_text()) or {}


@dataclass(frozen=True)
class Target:
    """One QR code: manifest `key`, its `url`, the project zip it opens (page-relative,
    `gui/projects/x.zip`) and the file it opens inside that zip."""
    key: str
    url: str
    project: str | None
    path: str | None

    @property
    def zip_url(self) -> str | None:
        page = self.url.split("#", 1)[0].split("?", 1)[0]
        return urljoin(page, self.project) if self.project else None

    @property
    def local_zip(self) -> Path | None:
        return PROJECTS / Path(self.project).name if self.project else None


def targets(manifest=None) -> list[Target]:
    m = load_manifest() if manifest is None else manifest
    out = []
    for key, url in m.items():
        parts = urlsplit(url)
        q = parse_qs(parts.query)
        if "?" in parts.fragment:            # `#/open?project=...&path=...`
            q.update(parse_qs(parts.fragment.split("?", 1)[1]))
        out.append(Target(key, url, q.get("project", [None])[0], q.get("path", [None])[0]))
    return out


def project_json(zip_path) -> dict | None:
    with zipfile.ZipFile(zip_path) as z:
        if "project.json" not in z.namelist():
            return None
        return json.loads(z.read("project.json"))


def session_projects(manifest=None) -> list[Path]:
    """The local zips the manifest links to that carry GUI sessions (not the GIF bundle)."""
    seen = []
    for t in targets(manifest):
        z = t.local_zip
        if z is not None and z not in seen and z.exists() and (project_json(z) or {}).get("sessions"):
            seen.append(z)
    return seen


def sessions(manifest=None) -> list[tuple[str, int, dict]]:
    """`[(zip name, session index, session)]` for every linked session project."""
    return [(z.name, i, s) for z in session_projects(manifest)
            for i, s in enumerate(project_json(z)["sessions"])]


def _catalog(tab: str, card_id: str | None) -> dict | None:
    if not card_id:
        return None
    cards = json.loads((GUI / "cards" / tab / "default.json").read_text())
    return (next((c for c in cards if c["id"] == card_id), None)
            or next((c for c in cards if "card-" + c["id"] == card_id), None))


def backend(session: dict) -> str | None:
    """The solver card's backend tag (`numpy`, `jax`, `amrex`, ...); None without a solver."""
    sel, ov = session.get("selections") or {}, session.get("cardOverrides") or {}
    solver = _catalog("solvers", sel.get("solver"))
    if not solver and (ov.get(sel.get("solver")) or {}).get("code") is None:
        return None
    return (solver or {}).get("requires_tag") or "numpy"


def compose(session: dict) -> list[tuple[str, str]]:
    """`[(role, code)]`, the code cells of the case the GUI writes for `session`."""
    sel, ov = session.get("selections") or {}, session.get("cardOverrides") or {}

    def code(card_id):
        c = (ov.get(card_id) or {}).get("code")
        if c is None:
            raise NotImplementedError(f"card {card_id!r} has no code override (template cards "
                                      "are not composed by this harness)")
        return c

    cells = [("model", code(sel.get("model")).rstrip())]
    cells += [("step: " + st.get("title", "Step"), str(st.get("code", "")).rstrip())
              for st in session.get("steps") or []]
    if sel.get("mesh") and (_catalog("meshes", sel["mesh"]) or (ov.get(sel["mesh"]) or {}).get("code") is not None):
        cells.append(("mesh", code(sel["mesh"]).rstrip()))
    tag = backend(session)
    if tag is not None:
        solver = _catalog("solvers", sel.get("solver")) or {}
        # the GUI's Run and notebook (gatherSpec): the card's defaults under the session's values
        settings = {**(solver.get("init") or {}),
                    **((ov.get(sel.get("solver")) or {}).get("params") or {})}
        settings.setdefault("backend", tag)
        if solver.get("id"):
            settings.setdefault("solver_id", solver["id"])
        cells.append(("settings", "settings = " + json.dumps(settings, indent=2, ensure_ascii=False)))
        cells.append(("run", code(sel.get("solver"))))
    if sel.get("visualization"):
        cells.append(("visualization", code(sel["visualization"]).rstrip()))
    return cells


def time_end_spec(zname, idx, session):
    """The time a solving session's store must reach: `END_TIME`, else the solver card's
    `time_end` parameter (a number or an expression for `session_runner.end_time`)."""
    if (zname, idx) in END_TIME:
        return END_TIME[(zname, idx)]
    card = (session.get("cardOverrides") or {}).get(session["selections"]["solver"]) or {}
    spec = (card.get("params") or {}).get("time_end")
    assert spec is not None, (f"{zname}#{idx}: no time_end to check the run against: the solver "
                              "card's `time_end` parameter, else an END_TIME entry")
    return spec
