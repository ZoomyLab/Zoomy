import argparse
from pathlib import Path

import numpy as np


def _load_collect(path: Path):
    d = np.load(path)
    q = d["Q"]
    qaux = d["Qaux"]
    n_steps = int(d["n_steps"][0]) if "n_steps" in d.files else 0
    return {
        "path": str(path),
        "Q": q,
        "Qaux": qaux,
        "n_steps": n_steps,
    }


def _estimate_scale(records):
    # heuristic: use state amplitude and aux variability to produce stable guess_scale
    amps = []
    for r in records:
        q = r["Q"]
        qaux = r["Qaux"]
        amp_q = float(np.mean(np.abs(q)))
        amp_aux = float(np.mean(np.abs(qaux))) if qaux.size else 0.0
        amps.append(amp_q + 0.5 * amp_aux)
    a = float(np.mean(amps)) if amps else 1.0
    # map amplitude to conservative scale in [0.2, 1.8]
    scale = 1.0 / (1.0 + 0.5 * a)
    return float(np.clip(scale * 2.0, 0.2, 1.8))


def customize(collect_files, out_file: Path, base_guess_mode: str):
    records = [_load_collect(Path(f)) for f in collect_files]
    scale = _estimate_scale(records)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        guess_mode=np.asarray([base_guess_mode]),
        guess_scale=np.asarray([scale], dtype=float),
        n_records=np.asarray([len(records)], dtype=int),
        sources=np.asarray([str(Path(f)) for f in collect_files]),
    )
    print(f"Wrote preconditioner config: {out_file}")
    print(f"guess_mode={base_guess_mode} guess_scale={scale:.4f} from {len(records)} record(s)")


def main():
    parser = argparse.ArgumentParser(description="Customize GNN preconditioner config from collected solver runs")
    parser.add_argument("--collect", nargs="+", required=True, help="One or more collect .npz files")
    parser.add_argument("--out", type=Path, default=Path("outputs/gnn_blueprint/precond/customized_precond.npz"))
    parser.add_argument("--guess-mode", type=str, default="residual", choices=["zero", "explicit", "residual", "learned_deltaq"])
    args = parser.parse_args()

    out = args.out if args.out.is_absolute() else (Path.cwd() / args.out)
    customize(args.collect, out, args.guess_mode)


if __name__ == "__main__":
    main()
