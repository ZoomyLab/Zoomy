import argparse
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import gmres


def _laplacian_1d(n):
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, i] = 2.0
        if i > 0:
            A[i, i-1] = -1.0
        if i < n - 1:
            A[i, i+1] = -1.0
    return A


def _run_gmres(A, b, x0):
    it_counter = {"n": 0}
    def cb(_):
        it_counter["n"] += 1
    x, info = gmres(A, b, x0=x0, callback=cb, callback_type="legacy", atol=0.0, rtol=1e-8, restart=50, maxiter=200)
    return x, info, it_counter['n']


def benchmark(model_dir: Path, n: int = 200):
    w = np.load(model_dir / 'weights_deltaq.npz')
    bvec = w['b']
    wself = w['w_self']

    A = _laplacian_1d(n)
    rhs = np.sin(np.linspace(0, 2 * np.pi, n))

    # Baseline: zero guess.
    _, info0, it0 = _run_gmres(A, rhs, np.zeros(n))

    # GNN-like guess (deltaQ style proxy): use rhs as forcing feature plus learned bias term.
    q_old = np.zeros(n)
    delta_guess = 0.5 * rhs + bvec[0]
    x0 = q_old + delta_guess
    _, info1, it1 = _run_gmres(A, rhs, x0)

    print(f"GMRES baseline: info={info0}, iterations={it0}")
    print(f"GMRES gnn_guess: info={info1}, iterations={it1}")
    print(f"Iteration improvement: {it0 - it1}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark GMRES with deltaQ initial guess')
    parser.add_argument('--model-dir', type=Path, default=Path('outputs/gnn_blueprint/model_deltaq'))
    parser.add_argument('--n', type=int, default=200)
    args = parser.parse_args()

    model_dir = args.model_dir if args.model_dir.is_absolute() else (Path.cwd() / args.model_dir)
    benchmark(model_dir, args.n)


if __name__ == '__main__':
    main()
