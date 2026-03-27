import csv
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple


CASES = [
    {
        "label": "swashes_dam_break_dry_ritter",
        "args": ["1", "3", "1", "2", "200"],  # dim,type,domain,choice,ncellx
    },
    {
        "label": "swashes_dam_break_step",
        "args": ["1", "7", "1", "1", "200"],
    },
]


HEADER_RE = re.compile(r"^#\s*(.+?):\s*(.+)\s*$")


def _parse_swashes_output(stdout: str) -> Tuple[Dict[str, str], List[Dict[str, float]]]:
    lines = stdout.splitlines()
    meta: Dict[str, str] = {}
    rows: List[Dict[str, float]] = []
    in_table = False
    table_cols = ["x", "h", "u", "b", "q", "eta", "froude", "crit_head"]

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if "(i-0.5)*dx" in line:
                in_table = True
                continue
            m = HEADER_RE.match(line)
            if m:
                key = m.group(1).strip().lower().replace(" ", "_")
                meta[key] = m.group(2).strip()
            continue
        if not in_table:
            continue

        # Split by arbitrary whitespace. NaN values are supported by float().
        vals = line.split()
        if len(vals) < 8:
            continue
        row = {
            "x": float(vals[0]),
            "h": float(vals[1]),
            "u": float(vals[2]),
            "b": float(vals[3]),
            "q": float(vals[4]),
            "eta": float(vals[5]),
            "froude": float(vals[6]),
            "crit_head": float(vals[7]),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No SWASHES data rows parsed from output.")
    return meta, rows


def _run_swashes(swashes_bin: str, args: List[str]) -> Tuple[Dict[str, str], List[Dict[str, float]]]:
    cmd = [swashes_bin] + args
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return _parse_swashes_output(result.stdout)


def _write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_meta(path: str, meta: Dict[str, str], args: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("swashes_args: " + " ".join(args) + "\n")
        for k in sorted(meta.keys()):
            f.write(f"{k}: {meta[k]}\n")


def run():
    swashes_bin = os.environ.get("SWASHES_BIN", "")
    if not swashes_bin:
        # Prefer the executable in the active Python environment.
        candidate = os.path.join(os.path.dirname(sys.executable), "swashes")
        swashes_bin = candidate if os.path.exists(candidate) else (shutil.which("swashes") or "swashes")
    out_dir = "outputs/swashes_reference_v2"
    os.makedirs(out_dir, exist_ok=True)

    for case in CASES:
        label = case["label"]
        args = case["args"]
        meta, rows = _run_swashes(swashes_bin, args)
        csv_path = os.path.join(out_dir, f"{label}.csv")
        meta_path = os.path.join(out_dir, f"{label}.meta.txt")
        _write_csv(csv_path, rows)
        _write_meta(meta_path, meta, args)
        print(f"saved: {csv_path} ({len(rows)} rows)")
        print(f"saved: {meta_path}")


if __name__ == "__main__":
    run()
