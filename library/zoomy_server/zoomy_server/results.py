"""Named result store — the RESULTS SHELF.

A per-server-instance registry of named HDF5 result stores. A completed
job's ``simulation.h5`` (in the ephemeral jobs dir, which is GC'd when the
job is cancelled or the server restarts) can be *saved under a name* into a
separate persistent-ish results dir; any client (a later session, another
run) can then list, download, or delete results by name.

Storage is intentionally **ephemeral-per-server**: the results dir defaults
to ``$ZOOMY_RESULTS_DIR`` or ``<tmp>/zoomy_results``. It outlives individual
jobs (a job GC never touches it) but is NOT a durable archive — it lives on
the same box as the server and disappears with the OS tmp reaper / a fresh
container. Point ``ZOOMY_RESULTS_DIR`` at persistent storage if you need it
to survive a server restart.

Mirrors the shape of ``jobs.py``: a module-level dir + plain functions,
reading the module global at call time so tests can monkeypatch it.
"""

import os
import re
import shutil
import tempfile
from datetime import datetime, timezone

RESULTS_DIR = os.environ.get(
    "ZOOMY_RESULTS_DIR",
    os.path.join(tempfile.gettempdir(), "zoomy_results"),
)


def slugify(name):
    """A filesystem-safe, collision-stable slug for a result name.

    Lower-cases, maps every run of non ``[a-z0-9]`` to a single ``-``,
    trims leading/trailing dashes, caps length. Raises ``ValueError`` for
    a name that slugs to empty (all punctuation / whitespace)."""
    s = re.sub(r"[^a-z0-9]+", "-", str(name or "").strip().lower()).strip("-")
    s = s[:80]
    if not s:
        raise ValueError(f"result name {name!r} slugs to empty")
    return s


def _path(name):
    return os.path.join(RESULTS_DIR, slugify(name) + ".h5")


def _entry(name):
    p = _path(name)
    st = os.stat(p)
    return {
        "name": slugify(name),
        "size": st.st_size,
        "created": datetime.fromtimestamp(st.st_mtime, timezone.utc)
        .isoformat(timespec="seconds"),
    }


def save(src_h5_path, name):
    """Copy an HDF5 file into the results shelf under ``name``.

    ``src_h5_path`` is the job's ``simulation.h5`` (see
    ``jobs.get_hdf5_path``). Returns the new registry entry. Overwrites a
    same-named result (last-write-wins, like a named variable)."""
    if not src_h5_path or not os.path.isfile(src_h5_path):
        raise FileNotFoundError(f"no HDF5 to save: {src_h5_path!r}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dest = _path(name)
    shutil.copyfile(src_h5_path, dest)
    return _entry(name)


def list_results():
    """All named results, newest first: ``[{name, size, created}, ...]``."""
    if not os.path.isdir(RESULTS_DIR):
        return []
    out = []
    for fn in os.listdir(RESULTS_DIR):
        if not fn.endswith(".h5"):
            continue
        try:
            out.append(_entry(fn[:-3]))
        except (OSError, ValueError):
            continue
    out.sort(key=lambda e: e["created"], reverse=True)
    return out


def get_path(name):
    """Absolute path to a named result's HDF5, or ``None`` if absent."""
    try:
        p = _path(name)
    except ValueError:
        return None
    return p if os.path.isfile(p) else None


def delete(name):
    """Remove a named result. Returns True if it existed, False otherwise."""
    p = get_path(name)
    if not p:
        return False
    os.remove(p)
    return True
