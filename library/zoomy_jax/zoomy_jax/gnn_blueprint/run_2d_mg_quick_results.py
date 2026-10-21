#!/usr/bin/env python3
"""Quick end-to-end numbers for multibranch + z on all dataset_2d_mg NPZ files (~3–8 min total on CPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from zoomy_jax.gnn_blueprint.train_2d_mg_benchmark import (
    gmres_mean_matvec,
    gmres_mean_matvec_multibranch,
    train_multibranch,
    train_z,
)


def main():
    data_dir = Path.cwd() / "outputs/gnn_blueprint/dataset_2d_mg"
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        print("no npz", data_dir)
        return 1
    rows = []
    for path in files:
        if path.name == "benchmark_results.json" or "benchmark" in path.name:
            continue
        d = np.load(path)
        data = {k: d[k] for k in d.files}
        n = int(np.asarray(d["n_nodes"]).reshape(-1)[0])
        mesh = str(np.asarray(d["mesh_kind"]).reshape(-1)[0])
        if isinstance(mesh, bytes):
            mesh = mesh.decode()
        print(f"\n>>> {path.name} n={n} {mesh}")
        mb_p, mb_m = train_multibranch(
            data, epochs=42, batch=48, lr=0.018, lambda_branch=0.42, n_mp=3, hid=28, seed=13
        )
        z_p, z_m = train_z(
            data, epochs=42, batch=48, lr=0.02, beta_star=0.55, n_mp=3, hid=28, seed=13
        )
        edges = jnp.asarray(d["edges"], dtype=jnp.int32)
        a = np.asarray(d["A"], dtype=np.float64)
        gm_mb, gm0 = gmres_mean_matvec_multibranch(
            a, mb_p, d["f"], edges, 3, 28, 18, 1e-8, min(600, 10 * n)
        )
        gm_z, _ = gmres_mean_matvec(a, z_p, d["f"], edges, 3, 28, 18, 1e-8, min(600, 10 * n))
        row = {
            "file": path.name,
            "mesh": mesh,
            "n": n,
            "mb_mse_ut": mb_m["mse_ut_test"],
            "z_mse": z_m["mse_z_test"],
            "gmres_baseline": gm0,
            "gmres_multibranch": gm_mb,
            "gmres_z": gm_z,
        }
        rows.append(row)
        print(json.dumps(row, indent=2))
    out = data_dir / "benchmark_quick_results.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
