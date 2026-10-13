"""The GUI sessions and solver cards a backend runs, composed the way the GUI composes them.

Sessions: every packed session (`project.json` `sessions`) of the tested checkout's
library/zoomy_gui/projects, the zips a thesis QR code opens first (tests/qr_cases/animations.yml),
exact duplicates (zoomy-cases.zip repacks the others) once. Cards: every catalog solver card,
in a fresh GUI. A case goes to the backend whose adapter tag is its solver card's
`requires_tag` (the GUI's `tagMatches`).
"""
from __future__ import annotations

import json
import os
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

from harness import CHECKOUT, HERE, ROOT, sha

#: the GUI's time_end knob (Parameters panel), as a reader would set it for a quick look;
#: model, mesh and numerics unchanged
TIME_LIMITS = {"malpasset-0": 2.0,     # JAX: 2000 s -> 2 s (as tests/qr_cases)
               "malpasset-1": 2.0,     # AMReX: 100 s -> 2 s
               "bingham-threshold-0": 20.0}   # 3 x 100 s (physical) -> 3 x 20 s
#: the model card a reader picks with a solver card that names it (the Chorin split cards march
#: the `split` the VAM card binds); other cards run with the first model card, as a fresh GUI
CARD_MODELS = {"card-solver-split-numpy": "vam", "card-solver-split-amrex": "vam"}
#: cases whose time_end knob is not the store's time unit (no t_last == time_end check)
TIME_UNITS = {"bingham-0": "time_end is t' = t / tscale (the model cell's nu / (g s^2))"}
#: not run, with the reason
SKIP = {
    "coupling-live-0": "known gap: the SME-VOF live coupling is not ported to the symbolic "
                       "solvers (not QR-wired); it drives a host checkout's case tree in place "
                       "(SME_VOF_CASE, writes snap_gui_live there). E2E_LIVE_COUPLING=1 runs it",
    "coupling-triple-live-0": "known gap: as coupling-live-0. E2E_LIVE_COUPLING=1 runs it",
}
#: per case: the conserved row and the bound on its relative change (closed domains, round-off
#: as matrix C6); open domains (inflow / outflow, the replayed coupling boundary) have none
#: (Bingham threshold: open incline; Bingham analytics: a one-field summary store)
EXPECT = {"malpasset-0": {"row": "h", "tol": 1e-12},           # walled basin
          "malpasset-1": {"row": "h", "tol": 1e-12},           # walled basin
          "zoomy-codeprinter-0": {"row": "h", "tol": 1e-12},   # walls, Riemann problem
          "bingham-0": {"row": "h", "tol": 1e-12}}             # periodic


def canon(tag: str) -> str:
    return "OpenFOAM" if tag == "foam" else tag


@dataclass
class GuiCase:
    id: str                  # malpasset-0 | card-solver-jax
    title: str
    tag: str                 # the solver card's requires_tag
    qr: tuple                # thesis QR keys that open it
    item: dict               # compose.mjs input
    time_end: float | None   # the knob the harness sets, None: the case's own
    skip: str | None = None

    @property
    def qr_wired(self) -> bool:
        return bool(self.qr)


def gui_dir(checkout: Path = CHECKOUT) -> Path:
    return checkout / "library" / "zoomy_gui"


def _manifest() -> dict:
    """The thesis QR manifest (tests/qr_cases/animations.yml): `{key: url}`."""
    for p in (CHECKOUT / "tests/qr_cases/animations.yml", ROOT / "tests/qr_cases/animations.yml"):
        if p.exists():
            out = {}
            for line in p.read_text().splitlines():
                if line[:1].isalnum() and ": http" in line:
                    k, url = line.split(":", 1)
                    out[k.strip()] = url.strip()
            return out
    return {}


def _qr_keys(zip_name: str) -> tuple:
    return tuple(k for k, url in _manifest().items() if f"gui/projects/{zip_name}" in str(url))


def _catalog(gui: Path, tab: str) -> list:
    return json.loads((gui / "cards" / tab / "default.json").read_text())


def _solver_tag(gui: Path, session: dict) -> str | None:
    """The tag of the session's solver card (`materializeSessions`: catalog id or "card-" + id;
    a session without a solver runs nothing)."""
    sel, ov = session.get("selections") or {}, session.get("cardOverrides") or {}
    sid = sel.get("solver")
    if not sid:
        return None
    cards = _catalog(gui, "solvers")
    card = next((c for c in cards if c["id"] == sid), None) or \
        next((c for c in cards if "card-" + c["id"] == sid), None)
    if card is None and (ov.get(sid) or {}).get("code") is None:
        return None
    return (card or {}).get("requires_tag") or "numpy"


def gui_cases(checkout: Path = CHECKOUT) -> list[GuiCase]:
    gui = gui_dir(checkout)
    out, seen = [], set()
    zips = sorted((gui / "projects").glob("*.zip"))
    zips.sort(key=lambda z: (not _qr_keys(z.name), z.name == "zoomy-cases.zip", z.name))
    for z in zips:
        with zipfile.ZipFile(z) as f:
            if "project.json" not in f.namelist():
                continue
            proj = json.loads(f.read("project.json"))
        for i, s in enumerate(proj.get("sessions") or []):
            tag = _solver_tag(gui, s)
            key = sha(json.dumps({k: s.get(k) for k in ("selections", "cardOverrides", "steps")},
                                 sort_keys=True))
            if tag is None or key in seen:
                continue
            seen.add(key)
            cid = f"{z.name.removesuffix('-session.zip').removesuffix('.zip')}-{i}"
            skip = None if os.environ.get("E2E_LIVE_COUPLING") else SKIP.get(cid)
            out.append(GuiCase(cid, s.get("title") or cid, tag, _qr_keys(z.name),
                               {"session": s, "timeEnd": TIME_LIMITS.get(cid)},
                               TIME_LIMITS.get(cid), skip))
    for c in _catalog(gui, "solvers"):
        cid = "card-" + c["id"]
        out.append(GuiCase(cid, c.get("title") or cid, c.get("requires_tag") or "numpy", (),
                           {"solverCard": c["id"], "timeEnd": TIME_LIMITS.get(cid),
                            "model": CARD_MODELS.get(cid)},
                           TIME_LIMITS.get(cid)))
    return out


def for_backend(adapter: str, checkout: Path = CHECKOUT) -> list[GuiCase]:
    return [c for c in gui_cases(checkout) if canon(c.tag) == canon(adapter)]


def compose(cases: list[GuiCase], lib: Path) -> list[dict]:
    """`[{spec, casePy}]`, by the GUI's own code (compose.mjs) on the library `lib`."""
    p = subprocess.run(["node", str(HERE / "compose.mjs"), str(lib / "zoomy_cli/src/cli.mjs"),
                        str(lib / "zoomy_gui")], input=json.dumps([c.item for c in cases]),
                       capture_output=True, text=True, check=True)
    return json.loads(p.stdout)


def cells(case_py: str) -> list[tuple[str, str]]:
    """`[(role, code)]` of a composed case.py, the cells a notebook / the GUI worker runs."""
    out, meta, buf = [], None, []

    def flush():
        if meta is not None and meta[0] == "code":
            out.append((meta[1].get("role") or "code", "\n".join(buf).strip("\n")))

    for line in case_py.split("\n"):
        if line.startswith("# %%"):
            flush()
            md = "[markdown]" in line
            info = json.loads(line.split("zoomy=", 1)[1]) if "zoomy=" in line else {}
            meta, buf = ("md" if md else "code", info), []
        else:
            buf.append(line)
    flush()
    return out
