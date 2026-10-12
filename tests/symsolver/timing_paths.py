"""Paths of the symsolver timing tests; importing it puts the matrix and
gates directories on sys.path (their modules import each other by bare name)."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MATRIX = ROOT / "docs" / "dev" / "symbolic_solver" / "matrix"
GATES = ROOT / "docs" / "dev" / "symbolic_solver" / "gates"

for _p in (MATRIX, GATES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def results_dir():
    """Directory of timings.jsonl / timings_latest.json (ZOOMY_TIMING_RESULTS)."""
    return Path(os.environ.get("ZOOMY_TIMING_RESULTS", ROOT / "tests" / "results" / "symsolver"))


def work_dir():
    """Scratch directory: case directories, the JAX reference cache (ZOOMY_TIMING_WORK)."""
    return Path(os.environ.get("ZOOMY_TIMING_WORK", HERE / "work"))
