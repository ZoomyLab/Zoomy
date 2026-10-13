"""Every DOI / URL cited in the sources is in docs/book/references/literature.bib.

Offline: ``collect_references.py --check`` scans the library sources and the
docs pages and compares with the committed bib file (no network).
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docs" / "scripts" / "collect_references.py"


@pytest.mark.small
@pytest.mark.unittest
def test_literature_bib_covers_all_cited_references():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stdout + r.stderr
