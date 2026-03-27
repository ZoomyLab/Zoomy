import json
import os
from pathlib import Path
from typing import Any

import numpy as np


_BASE_DIR = Path(__file__).resolve().parents[1] / "results" / "baselines"


def _create_mode_enabled() -> bool:
    val = os.environ.get("ZOOMY_CREATE_BASELINES", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            out[key] = value
        elif np.isscalar(value):
            out[key] = np.asarray(value)
        else:
            out[key] = np.asarray(value)
    return out


def assert_or_create_array_baseline(
    name: str,
    payload: dict[str, Any],
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> None:
    """
    Store arrays into baseline file if missing (or explicitly requested),
    otherwise compare current payload against stored baseline.
    """
    _BASE_DIR.mkdir(parents=True, exist_ok=True)
    baseline_file = _BASE_DIR / f"{name}.npz"
    normalized = _normalize_payload(payload)

    if _create_mode_enabled() or (not baseline_file.exists()):
        np.savez(baseline_file, **normalized)
        return

    baseline = np.load(baseline_file)
    for key, value in normalized.items():
        if key not in baseline:
            raise AssertionError(f"Missing key '{key}' in baseline '{baseline_file}'.")
        np.testing.assert_allclose(value, baseline[key], rtol=rtol, atol=atol)


def assert_or_create_json_baseline(
    name: str,
    payload: dict[str, Any],
) -> None:
    """
    Store scalar metrics in JSON if missing (or explicitly requested),
    otherwise compare exact values. Use this for categorical metadata.
    """
    _BASE_DIR.mkdir(parents=True, exist_ok=True)
    baseline_file = _BASE_DIR / f"{name}.json"

    if _create_mode_enabled() or (not baseline_file.exists()):
        with open(baseline_file, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        return

    with open(baseline_file, "r", encoding="utf-8") as fh:
        baseline = json.load(fh)

    if payload != baseline:
        raise AssertionError(
            f"JSON baseline mismatch for '{name}'.\n"
            f"Current: {payload}\nBaseline: {baseline}"
        )

