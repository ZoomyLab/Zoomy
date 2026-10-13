"""Build the pure-Python wheels the GUI's Pyodide workers install, from this checkout.

`python docs/scripts/build_gui_wheels.py OUT` writes `OUT/<wheel>.whl` for zoomy_core and
zoomy_plotting plus `OUT/index.json`:
`{"zoomy-core": {"wheel": "zoomy_core-0.2.14-py3-none-any.whl", "commit": .., "sha256": ..}, ...}`.

- The deploy (`render-webpage.yml`) serves `OUT` as `gui/gui/wheels/`; the workers install
  what `index.json` names and fall back to PyPI only when it is absent (local dev).
- The QR CI (`qr-cases.yml`) installs the same files in CPython and Pyodide.
- Zip timestamps are the source commit's time (`SOURCE_DATE_EPOCH`), so one commit builds
  the same wheel in both places. A wheel that is not `py3-none-any` is an error.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = {"zoomy-core": ROOT / "library" / "zoomy_core",
            "zoomy-plotting": ROOT / "library" / "zoomy_plotting"}


def _git(src, *args):
    return subprocess.check_output(["git", "-C", str(src), *args], text=True).strip()


def build(src, out):
    """Build one wheel from a clean copy of `src` into `out`; return its file name."""
    env = dict(os.environ, SOURCE_DATE_EPOCH=_git(src, "log", "-1", "--format=%ct", "HEAD"))
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / src.name
        shutil.copytree(src, copy, ignore=shutil.ignore_patterns(
            ".git", "tests", "examples", "__pycache__", "build", "dist", "*.egg-info"))
        subprocess.run([sys.executable, "-m", "pip", "wheel", "-q", "--no-deps", "-w",
                        str(Path(tmp) / "dist"), str(copy)], check=True, env=env)
        (whl,) = (Path(tmp) / "dist").glob("*.whl")
        if not whl.name.endswith("-py3-none-any.whl"):
            raise SystemExit(f"{whl.name}: not a pure-Python wheel")
        shutil.copy(whl, out / whl.name)
        return whl.name


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for name, src in PACKAGES.items():
        whl = build(src, out)
        index[name] = {"wheel": whl, "commit": _git(src, "rev-parse", "HEAD"),
                       "sha256": hashlib.sha256((out / whl).read_bytes()).hexdigest()}
    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(json.dumps(index, indent=1))
    return index


if __name__ == "__main__":
    main(sys.argv[1])
