"""
Utilities for loading Basilisk benchmark output (VTK + CSV)
and comparing against Zoomy solver results.
"""
import json
import numpy as np
from pathlib import Path

BASILISK_OUTPUT_ROOT = Path(__file__).parents[2] / "artifacts" / "basilisk_benchmarks"


def load_vtk_series(case_name: str, field_name: str):
    """
    Load a time series of a scalar field from Basilisk VTK output.

    Returns
    -------
    times : list[float]
    x_coords : list[np.ndarray]
    values : list[np.ndarray]
    """
    case_dir = BASILISK_OUTPUT_ROOT / case_name
    time_map = _load_time_map(case_dir, case_name)
    series_files = sorted(case_dir.glob(f"{case_name}_*.vtk"))

    if not series_files:
        raise FileNotFoundError(
            f"No VTK files for '{case_name}' in {case_dir}. "
            "Run containers/basilisk/run_benchmarks.sh first."
        )

    times = []
    x_coords = []
    values = []

    for vtk_path in series_files:
        x, v = _parse_vtk_fields(vtk_path, field_name)
        t = time_map.get(vtk_path.name, _extract_frame_index(vtk_path))
        times.append(t)
        x_coords.append(x)
        values.append(v)

    return times, x_coords, values


def load_csv_profile(case_name: str, filename: str):
    """Load a CSV file from a Basilisk benchmark output directory."""
    path = BASILISK_OUTPUT_ROOT / case_name / filename
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return np.genfromtxt(path, delimiter=",", names=True)


def compare_fields(x_ref, v_ref, x_test, v_test, rtol=0.05, atol=1e-6):
    """
    Compare two 1D fields by interpolating test onto reference grid.

    Returns
    -------
    dict with keys: l1, l2, linf, passed
    """
    v_interp = np.interp(x_ref, x_test, v_test)
    diff = np.abs(v_interp - v_ref)
    ref_scale = max(np.max(np.abs(v_ref)), 1e-12)

    l1 = np.mean(diff)
    l2 = np.sqrt(np.mean(diff**2))
    linf = np.max(diff)

    passed = (linf / ref_scale) < rtol and linf < (atol + rtol * ref_scale)

    return {"l1": l1, "l2": l2, "linf": linf, "passed": passed}


def _load_time_map(case_dir: Path, case_name: str):
    """Load time mapping from .vtk.series JSON file if available."""
    series_path = case_dir / f"{case_name}.vtk.series"
    if not series_path.exists():
        return {}
    with open(series_path) as f:
        data = json.load(f)
    return {entry["name"]: entry["time"] for entry in data.get("files", [])}


def _parse_vtk_fields(path: Path, field_name: str):
    """
    Parse a legacy ASCII VTK unstructured grid file.
    Returns (x_coordinates, field_values).
    """
    with open(path) as f:
        lines = f.readlines()

    n_points = 0
    x_coords = []
    field_values = []
    idx = 0

    while idx < len(lines):
        line = lines[idx].strip()

        if line.startswith("POINTS"):
            n_points = int(line.split()[1])
            for j in range(1, n_points + 1):
                parts = lines[idx + j].split()
                x_coords.append(float(parts[0]))
            idx += n_points + 1
            continue

        if line.startswith(f"SCALARS {field_name}"):
            idx += 2  # skip LOOKUP_TABLE line
            for j in range(n_points):
                field_values.append(float(lines[idx + j].strip()))
            idx += n_points
            continue

        idx += 1

    if not field_values:
        available = _list_vtk_fields(lines)
        raise KeyError(
            f"Field '{field_name}' not found in {path.name}. "
            f"Available: {available}"
        )

    return np.array(x_coords), np.array(field_values)


def _extract_frame_index(path: Path):
    """Extract frame index from filename like 'swe_bump1d_0005.vtk'."""
    stem = path.stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return 0


def _list_vtk_fields(lines):
    """List all SCALARS fields found in VTK file lines."""
    fields = []
    for line in lines:
        if line.strip().startswith("SCALARS"):
            fields.append(line.strip().split()[1])
    return fields


def available_cases():
    """List available benchmark cases (directories with VTK output)."""
    if not BASILISK_OUTPUT_ROOT.exists():
        return []
    return [d.name for d in BASILISK_OUTPUT_ROOT.iterdir() if d.is_dir()]
