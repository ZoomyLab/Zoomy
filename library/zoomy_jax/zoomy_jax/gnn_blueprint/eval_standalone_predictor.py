import argparse
from pathlib import Path
import numpy as np


def evaluate(dataset_path: Path, weights_path: Path):
    d = np.load(dataset_path)
    q = d['q']
    dt = d['dt']
    dq_true = d['delta_q']
    cls = d['class_id'] if 'class_id' in d.files else np.zeros((q.shape[0], q.shape[2]), dtype=float)

    w = np.load(weights_path)
    w_self = w['w_self']
    b = w['b']
    variant = str(w['variant'][0]) if 'variant' in w.files else 'linear'

    n_samples, n_fields, n_cells = q.shape
    dq_pred = np.zeros_like(dq_true)

    for i in range(n_samples):
        cls_scale = 1.0 / (1.0 + 0.2 * cls[i])
        cls_scale = np.clip(cls_scale, 0.5, 1.0)
        for f in range(n_fields):
            rhs = w_self[min(f, len(w_self)-1)] * q[i, f] + b[min(f, len(b)-1)]
            dq_pred[i, f] = dt[i] * rhs * cls_scale

    mse = float(np.mean((dq_pred - dq_true)**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(dq_pred - dq_true)))

    print(f"variant={variant}")
    print(f"standalone_rmse={rmse:.6e}")
    print(f"standalone_mae={mae:.6e}")
    return rmse, mae


def main():
    parser = argparse.ArgumentParser(description='Standalone predictor evaluation on dataset')
    parser.add_argument('--dataset', type=Path, default=Path('outputs/gnn_blueprint/dataset_deltaq.npz'))
    parser.add_argument('--weights', type=Path, default=Path('outputs/gnn_blueprint/model_deltaq/weights_deltaq.npz'))
    args = parser.parse_args()

    dataset = args.dataset if args.dataset.is_absolute() else (Path.cwd() / args.dataset)
    weights = args.weights if args.weights.is_absolute() else (Path.cwd() / args.weights)
    evaluate(dataset, weights)


if __name__ == '__main__':
    main()
