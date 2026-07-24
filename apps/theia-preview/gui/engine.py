"""Pyodide-side execution engine for the Zoomy GUI.

Design notes:

* Simulation results live in an HDF5 file on Pyodide's virtual filesystem
  (``/tmp/zoomy_sim/sim.h5`` by convention). Nothing else. The browser is a
  regular filesystem as far as Python is concerned.
* ``store`` is a :class:`zoomy_plotting.SimulationStore` built via
  ``zoomy_plotting.read_hdf5(path)``. Lazy field reads, no in-memory
  arrays outside the open h5py handle.
* No ``SimulationStore`` shim, no ``auto_save_from_scope`` sniffing,
  no ``load_server_results`` JSON path. One code path for both the local
  Pyodide solver and a remote server job: download the HDF5 → write to
  VFS → read it.
* No fallbacks. If ``store`` is unset or malformed, viz snippets raise.
"""

import base64
import io
import json
import os
import sys

import numpy as np


# --- Lazy imports — plotly init is slow in WASM, matplotlib-pyodide
#     hangs in web workers unless we go through pyplot carefully. ---
_go = None
_plt = None


def _get_go():
    global _go
    if _go is None:
        import plotly.graph_objects as go
        _go = go
    return _go


def _get_plt():
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use("agg")
        import matplotlib.pyplot as mpl_plt
        _plt = mpl_plt
    return _plt


# --- Robust JSON encoder for numpy types (int64/float32 crash the default). ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# --- Rich display funnel (Jupyter-like output cells). ---
class ZoomyDisplay:
    def __call__(self, obj=None, *, mermaid=None, latex=None, html=None):
        if mermaid is not None:
            self._emit({"mime": "text/x-mermaid", "content": str(mermaid)})
        elif latex is not None:
            self._emit({"mime": "text/x-latex", "content": str(latex)})
        elif html is not None:
            self._emit({"mime": "text/html", "content": str(html)})
        elif obj is None:
            return
        elif hasattr(obj, "to_dict"):  # plotly figure
            self._emit({"mime": "application/vnd.plotly+json",
                        "content": json.dumps(obj.to_dict(), cls=NumpyEncoder)})
        elif hasattr(obj, "savefig"):  # matplotlib figure
            buf = io.BytesIO()
            obj.savefig(buf, format="svg", bbox_inches="tight")
            buf.seek(0)
            self._emit({"mime": "image/svg+xml",
                        "content": buf.read().decode("utf-8")})
        elif hasattr(obj, "to_html"):  # pandas DataFrame
            self._emit({"mime": "text/html", "content": obj.to_html()})
        elif isinstance(obj, np.ndarray):
            self._emit({"mime": "text/plain", "content": repr(obj)})
        elif self._emit_ipython_repr(obj):
            # Handled via Jupyter-style _repr_mimebundle_ / _repr_*_.
            pass
        else:
            self._emit({"mime": "text/plain", "content": str(obj)})

    # Jupyter-style rich reprs. `display(model.describe())` returns a
    # zoomy_core.misc.description.Description whose _repr_markdown_
    # carries the rendered model docs (with embedded $$-math and
    # optionally ```mermaid blocks). Without this path the object falls
    # through to str(obj) and the frontend renders raw markdown source.
    # Priority follows IPython's default: html > markdown > latex.
    def _emit_ipython_repr(self, obj):
        # _repr_mimebundle_ is the canonical multi-mime contract; it
        # returns either a dict of mime→content or a (dict, metadata)
        # tuple. We pick the richest mime we understand.
        if hasattr(obj, "_repr_mimebundle_"):
            try:
                bundle = obj._repr_mimebundle_(include=None, exclude=None)
                data = bundle[0] if isinstance(bundle, tuple) else bundle
                if isinstance(data, dict):
                    for src_mime, out_mime in (
                        ("text/html",     "text/html"),
                        ("text/markdown", "text/markdown"),
                        ("text/latex",    "text/x-latex"),
                    ):
                        if src_mime in data and data[src_mime]:
                            self._emit({"mime": out_mime,
                                        "content": str(data[src_mime])})
                            return True
            except Exception:
                pass
        # Fall back to the individual _repr_*_ hooks. Each returns None
        # if the object can't produce that mime; skip to the next.
        for method, mime in (
            ("_repr_html_",     "text/html"),
            ("_repr_markdown_", "text/markdown"),
            ("_repr_latex_",    "text/x-latex"),
        ):
            if hasattr(obj, method):
                try:
                    content = getattr(obj, method)()
                    if content:
                        self._emit({"mime": mime, "content": str(content)})
                        return True
                except Exception:
                    continue
        return False

    def _emit(self, cell):
        if hasattr(sys, "_zoomy_display_callback"):
            sys._zoomy_display_callback(cell)
        else:
            content = cell.get("content", "")
            if cell.get("mime") == "text/x-mermaid":
                print("[mermaid]", content[:200])
            elif cell.get("mime") == "text/x-latex":
                print("[latex]", content[:200])
            else:
                print(content[:500] if len(content) > 500 else content)


display = ZoomyDisplay()


# --- Persistent exec scope. Populated lazily; ``store`` starts unset and
#     is set by the solver template to a ``zoomy_plotting.SimulationStore``. ---
if not hasattr(sys, "_shallowflow_scope"):
    sys._shallowflow_scope = {"np": np}

sys._shallowflow_scope["display"] = display
sys._shallowflow_scope.setdefault("store", None)


# --- Live stdout streaming to the GUI dashboard log. ---
class _LiveStdout(io.StringIO):
    def __init__(self):
        super().__init__()
        self._buf = ""

    def write(self, s):
        super().write(s)
        if hasattr(sys, "_zoomy_display_callback"):
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip():
                    try:
                        sys._zoomy_display_callback({
                            "mime": "text/x-log",
                            "content": line,
                        })
                    except Exception:
                        pass
        return len(s)


# --- Helper used by the solver template to load results into the store. ---
def open_hdf5(path):
    """Open an HDF5 simulation output via zoomy_plotting and install it
    as the exec-scope ``store``.

    Raises loudly on anything unexpected: missing file, missing ``/mesh``
    group, mesh/field shape mismatch. No fallback, no soft failure."""
    import zoomy_plotting as zp   # lazy; triggers PyPI micropip install

    if not os.path.isfile(path):
        raise FileNotFoundError(f"open_hdf5: no such file: {path}")

    # Installing a new store replaces the old one: release the previous
    # run's h5py handle first so re-opens never leak file handles.
    close_store()

    store = zp.read_hdf5(path)    # validates schema internally

    # Sanity: the mesh we loaded must match the fields we loaded.
    # zoomy_plotting's SimulationStore already asserts vertices.shape[1]==dim
    # in __post_init__; we add a cell-count cross-check here so mismatches
    # surface with a clear message instead of at plot time.
    if store.cells.shape[0] != store.n_cells:
        raise ValueError(
            f"open_hdf5: cells/Q mismatch in {path}: "
            f"{store.cells.shape[0]} cells from /mesh, "
            f"but fields report {store.n_cells} cells. "
            f"Check the solver's HDF5 writer."
        )

    sys._shallowflow_scope["store"] = store
    print(f"[store] opened {path}  dim={store.dim} cell_type={store.cell_type} "
          f"n_cells={store.n_cells} n_snapshots={store.n_snapshots}")
    return store


sys._shallowflow_scope["open_hdf5"] = open_hdf5


# --- Named result shelf (cross-run / cross-session comparison). ----------
# A run's HDF5 store can be saved under a NAME (by the GUI: remote job ->
# server /api/v1/results, then staged into this VFS shelf; local run ->
# staged directly). A viz card then does `ref = open_result("swe-reference")`
# and composes comparisons. `open_result` returns a fresh SimulationStore and
# does NOT touch the exec-scope ``store`` — the current run's plot stays put.
_RESULTS_DIR = "/tmp/zoomy_results"


def _result_slug(name):
    """Slug identical to zoomy_server.results.slugify so a name saved to a
    backend resolves to the same file when staged into this VFS shelf."""
    import re
    s = re.sub(r"[^a-z0-9]+", "-", str(name or "").strip().lower()).strip("-")[:80]
    if not s:
        raise ValueError(f"result name {name!r} slugs to empty")
    return s


def result_path(name):
    """VFS path for a named result: /tmp/zoomy_results/<slug>.h5."""
    return os.path.join(_RESULTS_DIR, _result_slug(name) + ".h5")


def open_result(name):
    """Open a named result from the shelf as a fresh SimulationStore.

    Unlike ``open_hdf5`` this returns the store WITHOUT installing it as the
    scope ``store``, so a viz card can reference other results alongside the
    current run:  ``ref = open_result("swe-reference")``."""
    import zoomy_plotting as zp   # lazy; triggers micropip install

    path = result_path(name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"open_result: no such result {name!r} at {path} "
            f"(available: {list_results()})")
    return zp.read_hdf5(path)


def open_results(names):
    """Open several named results at once -> ``{name: SimulationStore}``."""
    return {n: open_result(n) for n in names}


def list_results():
    """Names present in the local results shelf (sorted)."""
    if not os.path.isdir(_RESULTS_DIR):
        return []
    return sorted(fn[:-3] for fn in os.listdir(_RESULTS_DIR) if fn.endswith(".h5"))


def save_result_local(name):
    """Copy the currently-open scope ``store``'s HDF5 into the local results
    shelf under ``name`` (a local run's "Save result as…"). Returns the slug.
    Raises if no store is open."""
    import shutil
    s = sys._shallowflow_scope.get("store")
    src = getattr(s, "source_path", None) if s is not None else None
    if not src or not os.path.isfile(src):
        raise RuntimeError("save_result_local: no open store to save")
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    dest = result_path(name)
    if os.path.abspath(src) != os.path.abspath(dest):
        shutil.copyfile(src, dest)
    return _result_slug(name)


def store_source_path():
    """VFS path of the current run's open store (its HDF5 file), or None.

    The GUI reads these bytes to route the post-processing chain to a backend
    after a local (Pyodide) run — the store the run just produced is the
    chain's input."""
    s = sys._shallowflow_scope.get("store")
    return getattr(s, "source_path", None) if s is not None else None


sys._shallowflow_scope["result_path"] = result_path
sys._shallowflow_scope["open_result"] = open_result
sys._shallowflow_scope["open_results"] = open_results
sys._shallowflow_scope["list_results"] = list_results
sys._shallowflow_scope["save_result_local"] = save_result_local
sys._shallowflow_scope["store_source_path"] = store_source_path


def close_store():
    """Close any store currently installed in scope and release its file handle.

    The previous run's ``SimulationStore`` holds an open ``h5py.File`` via
    ``_resource``; that lock prevents ``mesh.write_to_hdf5(path)`` from
    truncating the same path on a subsequent run. ``process_code`` now calls
    this automatically at the start of every store-producing (solver) run, so
    card code NEVER needs it — it is kept as a public no-op-compatible shim
    for backward compatibility with older cards that still call it."""
    s = sys._shallowflow_scope.get("store")
    if s is None:
        return
    try:
        s.close()
    except Exception:
        pass
    sys._shallowflow_scope["store"] = None


sys._shallowflow_scope["close_store"] = close_store


def _is_store_producer(code):
    """A run that re-creates the results store — it truncates a fresh HDF5
    (``mesh.write_to_hdf5``) and/or re-opens it (``open_hdf5``). Pure viz
    cards do neither; they read the scope ``store`` the solver run installed,
    so we must NOT close it out from under them."""
    return ("write_to_hdf5" in code) or ("open_hdf5" in code)


# --- Autocomplete via jedi ------------------------------------------------
# jedi is installed by the worker on first 'complete_code' call via
# micropip (the worker owns async install — doing it from sync Python
# is painful in Pyodide's single-threaded event loop). engine.py just
# imports it and runs jedi.Script.complete().


def complete_code(code: str, row: int, col: int, limit: int = 80) -> dict:
    """Return jedi completions at (1-indexed row, 0-indexed col).

    We combine two jedi APIs:
      * Script.complete() — returns members / names valid at cursor.
      * Script.get_signatures() — when the cursor is inside a
        callable's parens, returns the callable's params. We splice
        any missing param names in as type='param' completions so the
        full keyword-arg list is in the initial response (otherwise
        jedi sometimes omits params it thinks are positionally filled,
        and the user has to type a prefix to discover them).
    """
    try:
        import jedi
    except ImportError:
        return {"completions": [], "error": "jedi unavailable"}
    try:
        script = jedi.Script(code)
        completions = script.complete(row, col)
    except Exception as e:
        return {"completions": [], "error": str(e)}

    out = []
    seen_names = set()
    def _key(name):
        # jedi appends '=' to kwarg names ("assumptions=") but the
        # signature-supplement below adds the bare name. Normalise on
        # the bare name so we don't end up with both forms.
        return (name or "").rstrip("=")
    for c in completions[:limit]:
        k = _key(c.name)
        if k in seen_names:
            continue
        seen_names.add(k)
        sig_str = ""
        try:
            sigs = c.get_signatures()
            if sigs:
                sig_str = sigs[0].to_string()
        except Exception:
            pass
        doc = ""
        try:
            doc = c.docstring(raw=True) or ""
            if len(doc) > 2000:
                doc = doc[:2000] + " […]"
        except Exception:
            pass
        out.append({
            "name": c.name,
            "type": c.type,
            "signature": sig_str,
            "docstring": doc,
            "module": getattr(c, "module_name", "") or "",
        })

    # --- Supplement: full param list when inside a call -------------
    # jedi.complete() at `fn(<cursor>` sometimes returns only a subset
    # of the callable's params (it scores positionally-plausible ones
    # higher and trims the rest). get_signatures() always returns the
    # full param list; we splice in any missing names so Ace's
    # initial popup is comprehensive.
    try:
        signatures = script.get_signatures(row, col)
    except Exception:
        signatures = []
    for sig in signatures:
        sig_str = sig.to_string()
        doc = ""
        try:
            doc = sig.docstring(raw=True) or ""
            if len(doc) > 2000:
                doc = doc[:2000] + " […]"
        except Exception:
            pass
        for param in sig.params:
            pname = (param.name or "").rstrip("=")
            if not pname or pname in seen_names:
                continue
            seen_names.add(pname)
            # Append '=' so Ace inserts the kwarg form ("strip_args=") and
            # the cursor lands ready for the value — same convention jedi
            # uses for the kwargs it does return from complete().
            out.append({
                "name": pname + "=",
                "type": "param",
                "signature": sig_str,
                "docstring": doc,
                "module": "",
            })

    return {"completions": out}


sys._shallowflow_scope["complete_code"] = complete_code


# --- Main entry point for run_code messages from the worker. ---
def process_code(code_string):
    new_stdout = _LiveStdout()
    old_stdout = sys.stdout
    sys.stdout = new_stdout

    # Single output convention: the only way a script produces a card-level
    # output is by calling ``display(obj)``. No more fig-sniffing from the
    # exec scope — keeps snippets uniform and makes the "one plot replaces
    # the previous one" behaviour in the GUI a simple clear-then-append.
    res = {"status": "success", "output": "", "store_meta": None}
    scope = sys._shallowflow_scope

    # Auto-close the previous run's store before a solver run truncates the
    # same HDF5 path. Card code never needs close_store(); pure viz runs (no
    # write_to_hdf5/open_hdf5) keep the installed store so plots still resolve.
    if _is_store_producer(code_string):
        close_store()

    try:
        if _plt is not None:
            _plt.close("all")   # tidy up any stray mpl figures from the prior run
        exec(code_string, scope)

    except KeyboardInterrupt:
        # Cooperative cancel: the main thread wrote SIGINT into the shared
        # interrupt buffer, Pyodide raised KeyboardInterrupt between
        # bytecodes. Close any open store so the next run's write_to_hdf5
        # doesn't collide with a half-finished handle.
        s = scope.get("store")
        if s is not None:
            try:
                s.close()
            except Exception:
                pass
            scope["store"] = None
        res["status"] = "cancelled"
        res["output"] = "Simulation cancelled by user.\n"
    except Exception:
        import traceback
        res["status"] = "error"
        res["output"] = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
        res["output"] = new_stdout.getvalue() + res["output"]

    # Store metadata for the GUI's slider / field selector. Read off the
    # zoomy_plotting.SimulationStore currently in scope, if one is installed.
    s = scope.get("store")
    if s is not None and hasattr(s, "field") and hasattr(s, "n_snapshots"):
        try:
            res["store_meta"] = {
                "fields": list(s.field.keys()),
                "n_snapshots": int(s.n_snapshots),
                "dim": int(s.dim),
                "n_cells": int(s.n_cells),
            }
        except Exception:
            res["store_meta"] = None

    return json.dumps(res, cls=NumpyEncoder)
