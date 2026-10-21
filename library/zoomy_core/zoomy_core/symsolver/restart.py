"""The HDF5 restart file of the Python backends (NumPy, JAX): the march state `(time, Q, Qaux)` that
the templates' `io.save_state` / `io.load_state` pass, global arrays in the mesh's cell order.
h5py is imported only when a file is written or read (NumPy must import under Pyodide).
"""
import os

import numpy as np

from zoomy_core.misc.options import did_you_mean

FORMAT = "zoomy-restart-1"


def restart_path(checkpoint, output):
    """Where the restart file goes: `checkpoint`, else `<output stem>.restart.h5` next to the output
    (resolved as the output is), else None (no restart file).
    """
    if checkpoint is not None:
        return os.fspath(checkpoint)
    if not output:
        return None
    from zoomy_core.misc import misc
    stem = os.path.splitext(os.fspath(output))[0]
    return os.path.join(misc.get_main_directory(), stem + ".restart.h5")


def missing(path, what="restart_from"):
    """`FileNotFoundError` for a restart file that does not exist, naming the closest sibling."""
    folder = os.path.dirname(os.path.abspath(path))
    near = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
    hint = did_you_mean(os.path.basename(path), near) if near else f" (no directory {folder})"
    return FileNotFoundError(f"{what} {os.fspath(path)!r}: no such file{hint}")


def write_state(path, time, Q, Qaux, names):
    """Write the march state to `path` (replaced atomically: a crash keeps the previous file)."""
    import h5py
    path = os.fspath(path)
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with h5py.File(tmp, "w") as f:
        f.attrs["format"] = FORMAT
        f.attrs["names"] = np.array([str(n) for n in names], dtype="S")
        f.create_dataset("time", data=np.float64(time))
        f.create_dataset("Q", data=np.asarray(Q, dtype=np.float64))
        f.create_dataset("Qaux", data=np.asarray(Qaux, dtype=np.float64))
    os.replace(tmp, path)


def check_state(path, names, n_cells):
    """Raise with a clear message unless `path` is a restart file of this state and mesh."""
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise missing(path)
    import h5py
    try:
        f = h5py.File(path, "r")
    except OSError as e:
        raise ValueError(f"restart_from {path!r}: not an HDF5 restart file ({e})") from None
    with f:
        if f.attrs.get("format") != FORMAT:
            raise ValueError(f"restart_from {path!r}: not a zoomy restart file (format "
                             f"{f.attrs.get('format')!r}, expected {FORMAT!r})")
        saved = [n.decode() for n in f.attrs["names"]]
        if saved != [str(n) for n in names]:
            raise ValueError(f"restart_from {path!r}: state {saved}, this model has "
                             f"{[str(n) for n in names]}")
        if f["Q"].shape[1] != n_cells:
            raise ValueError(f"restart_from {path!r}: {f['Q'].shape[1]} cells, this mesh has "
                             f"{n_cells}")


def read_state(path):
    """`(time, Q, Qaux)` of a restart file (global arrays)."""
    import h5py
    with h5py.File(os.fspath(path), "r") as f:
        return float(f["time"][()]), f["Q"][()], f["Qaux"][()]
