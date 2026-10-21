import argparse
from pathlib import Path

import numpy as np


def coarsen_indices(n):
    # simple pairwise coarsening for 1D-like ordering
    idx = np.arange(n)
    coarse = idx[::2]
    fine_to_coarse = np.minimum(np.arange(n)//2, len(coarse)-1)
    return coarse, fine_to_coarse


def restrict_field(q_fine, fine_to_coarse, n_coarse):
    q_coarse = np.zeros((q_fine.shape[0], n_coarse), dtype=float)
    count = np.zeros((n_coarse,), dtype=float)
    for i in range(q_fine.shape[1]):
        c = fine_to_coarse[i]
        q_coarse[:, c] += q_fine[:, i]
        count[c] += 1
    q_coarse /= np.maximum(count[None,:], 1.0)
    return q_coarse


def prolong_field(q_coarse, fine_to_coarse):
    q_fine = np.zeros((q_coarse.shape[0], fine_to_coarse.shape[0]), dtype=float)
    for i,c in enumerate(fine_to_coarse):
        q_fine[:, i] = q_coarse[:, c]
    return q_fine


def smooth(q, omega=0.6):
    left = np.pad(q[:, :-1], ((0,0),(1,0)), mode='edge')
    right = np.pad(q[:, 1:], ((0,0),(0,1)), mode='edge')
    return (1-omega)*q + 0.5*omega*(left+right)


def multigrid_step(q, levels=3):
    pyr=[q]
    maps=[]
    # restrict
    for _ in range(levels-1):
        n=pyr[-1].shape[1]
        if n<=4:
            break
        coarse_idx, f2c = coarsen_indices(n)
        qc = restrict_field(pyr[-1], f2c, len(coarse_idx))
        pyr.append(qc)
        maps.append(f2c)
    # coarse smoothing
    pyr[-1] = smooth(pyr[-1], omega=0.9)
    # prolong + smooth
    for li in range(len(pyr)-2, -1, -1):
        up = prolong_field(pyr[li+1], maps[li])
        pyr[li] = smooth(0.5*pyr[li] + 0.5*up, omega=0.7)
    return pyr[0]


def demo(n_cells=80, n_fields=2, levels=3):
    x = np.linspace(0, 10, n_cells)
    q = np.zeros((n_fields, n_cells), dtype=float)
    q[0] = np.sin(2*np.pi*x/10) + 0.4*np.sin(8*np.pi*x/10)
    q[1] = np.cos(2*np.pi*x/10) + 0.2*np.cos(10*np.pi*x/10)
    q1 = multigrid_step(q, levels=levels)
    rmse = float(np.sqrt(np.mean((q1 - q)**2)))
    print(f"multigrid toy: n_cells={n_cells} levels={levels} rmse_change={rmse:.6e}")


def main():
    p=argparse.ArgumentParser(description='Toy multigrid GNN-style propagation demo')
    p.add_argument('--n-cells',type=int,default=80)
    p.add_argument('--n-fields',type=int,default=2)
    p.add_argument('--levels',type=int,default=3)
    a=p.parse_args()
    demo(a.n_cells,a.n_fields,a.levels)

if __name__ == '__main__':
    main()
