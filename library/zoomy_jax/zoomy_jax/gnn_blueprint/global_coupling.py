"""Global coupling modes for learned preconditioners (pressure / elliptic context).

The blueprint supports three architectural choices for long-range information flow:

- `MULTIGRID` (0): Multilevel restriction / prolongation + local smoothing (default).
- `FFT_1D` (1): After multilevel propagation, apply a per-field real-FFT linear
  spectral mix on uniform 1D cell ordering (prototype for GN 1D line meshes).
- `NUFFT_STUB` (2): Legacy placeholder; runtime still maps to `MULTIGRID`.
- `NUDFT_1D` (3): Dense low-mode non-uniform DFT (NUFFT-style) on 1D coordinates.
- `RFF_KERNEL_1D` (4): Random Fourier feature global mix on 1D coordinates.
- `NUDFT_2D` (5): Same as `NUDFT_1D` but with 2D wavevectors on `(x,y)`.
- `RFF_KERNEL_2D` (6): RFF global mix on 2D coordinates.
- `GRAPH_POLY_LAPL` (7): Learned polynomial `sum_k c_k L_sym^k` of the **mesh graph**
  Laplacian (cheap: `K` matmuls per field; connectivity-respecting).
- `GRAPH_EIGEN_LOW` (8): Diagonal filter in the first low **eigenvectors** of `L_sym`
  (graph Fourier prototype; `eigh` once per mesh offline, forward `O(nK)`).

See `README.md` section on global coupling for rationale.

**Note:** `MULTIGRID` here is **not** classical AMG — it is learned local smoothing plus
heuristic 1D coarsening along a cell ordering. `GRAPH_*` modes use the **same adjacency**
as the mesh graph (suitable for holes / irregular boundaries when `A` reflects them).
"""

from __future__ import annotations

MULTIGRID = 0
FFT_1D = 1
NUFFT_STUB = 2
NUDFT_1D = 3
RFF_KERNEL_1D = 4
NUDFT_2D = 5
RFF_KERNEL_2D = 6
GRAPH_POLY_LAPL = 7
GRAPH_EIGEN_LOW = 8

_MODE_NAMES = {
    MULTIGRID: "multigrid",
    FFT_1D: "fft_1d",
    NUFFT_STUB: "nufft_stub",
    NUDFT_1D: "nudft_1d",
    RFF_KERNEL_1D: "rff_kernel_1d",
    NUDFT_2D: "nudft_2d",
    RFF_KERNEL_2D: "rff_kernel_2d",
    GRAPH_POLY_LAPL: "graph_poly_lapl",
    GRAPH_EIGEN_LOW: "graph_eigen_low",
}


def mode_name(mode: int) -> str:
    return _MODE_NAMES.get(int(mode), "unknown")


def is_nufft_stub(mode: int) -> bool:
    return int(mode) == NUFFT_STUB
