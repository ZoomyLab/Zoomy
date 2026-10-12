"""The cross-backend matrix as tests: every row of ``matrix.measure()`` on the
store (MATRIX_STORE, default docs/dev/symbolic_solver/matrix/store) is one test,
checked against the threshold the matrix itself attaches to it.

    MATRIX_STORE=/path/to/store pytest tests/symsolver/test_matrix.py -m medium
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing_paths  # noqa: E402,F401

pytestmark = pytest.mark.medium

try:
    import matrix
except ImportError as e:
    pytest.skip(f"the matrix is not importable ({e}); source docs/dev/symbolic_solver/matrix/env.sh",
                allow_module_level=True)

if not matrix.STORE.is_dir():
    pytest.skip(f"no matrix store at {matrix.STORE} (set MATRIX_STORE)", allow_module_level=True)


def _rows():
    """[(case, check, backend, config, value, threshold, note)], in the order of the table."""
    try:
        rows = matrix.measure()
    except Exception as e:                  # a store the matrix cannot read is one failing test
        return [("store", f"measure() raised {type(e).__name__}: {str(e)[:200]}", "-", "-", e, None, "")]
    return sorted(rows, key=lambda r: (int(r[0][1:]), r[2], r[1], r[3]))


ROWS = _rows()


def _passes(value, th):
    """The table's verdict (``matrix.table``): ``value`` against threshold ``th`` = (op, bound, source)."""
    return bool(value >= th[1] if th[0] == ">=" else value <= th[1])


@pytest.mark.parametrize("row", [pytest.param(r, id="|".join((r[0], r[2], r[3], r[1]))) for r in ROWS])
def test_matrix_row(row):
    case, check, backend, config, value, th, note = row
    if isinstance(value, Exception):
        raise value
    if isinstance(value, str):
        pytest.skip(f"{case} {check}, {backend} {config}: {value}")
    if th is None:
        pytest.skip(f"{case} {check}, {backend} {config}: information only ({value:.3e})")
    assert _passes(value, th), (f"{case} {check}, {backend} {config}: {value:.3e} (need {th[0]} {th[1]:g}; "
                                f"{th[2]}) {note}")
