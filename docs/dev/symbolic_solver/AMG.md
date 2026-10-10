# Optional: AmgX pressure solves (JAX-AMG)

CPU: `pressure_linear_solver="pyamg"` -- PyAMG (Bell, Olson, Schroder 2022,
doi:10.21105/joss.04142; `pip install pyamg`) smoothed aggregation as the
preconditioner of GMRES, on the host; default on the CPU when installed
(`"auto"`).

GPU: `SymbolicChorinSolverJax(pressure_linear_solver="amgx")` solves the Chorin
pressure system with NVIDIA AmgX through JAX-AMG (Liu, Fan, Wang 2026,
doi:10.1016/j.softx.2026.102966; AmgX: Naumov et al. 2015,
doi:10.1137/140980260).  CUDA GPU only; optional (imported only when
selected, clear error otherwise).  Default stays `"gmres"`.

How it is used

- The pressure operator stays matrix-free in the template (`LinSolve`).
- Pattern: every pressure row of a cell couples to the cells of its LSQ
  stencil (static mesh topology), cell-major, `block_dim = #pressure rows`.
- The matrix is assembled in JAX by colouring (one vmapped operator
  evaluation per colour) and handed to AmgX / PyAMG as a sparse matrix.
- Default config: FGMRES(30) + AMG; with `block_dim` JAX-AMG uses
  aggregation AMG + block-Jacobi.  The 2x2 VAM block is indefinite
  (sym(-M0) of the derived operator: eigenvalues -0.10, 1.10); scalar AMG stalls
  (relative residual 0.17), block AMG converges -- no row rotation needed.

Install (gaia, L40S sm_89, CUDA 12.9; `zoomy-cuda` left otherwise unchanged)

```bash
micromamba create -n amgx-build -c conda-forge cuda-version=12.9 cuda-nvcc \
  cuda-cudart-dev libcublas-dev libcusparse-dev libcusolver-dev libcurand-dev \
  cuda-nvtx-dev cmake make git python=3.12
git clone --recursive https://github.com/NVIDIA/AMGX.git ~/opt/src/AMGX
cd ~/opt/src/AMGX && git checkout v2.5.0
P=$(micromamba run -n amgx-build bash -c 'echo $CONDA_PREFIX')
micromamba run -n amgx-build cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_NO_MPI=True -DCUDA_ARCH=89 -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER=$P/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++
micromamba run -n amgx-build cmake --build build -j 24
# a conventional CUDA_HOME over the conda toolkit
C=~/opt/cuda-12.9; mkdir -p $C
ln -s $P/targets/x86_64-linux/include $C/include
ln -s $P/targets/x86_64-linux/lib $C/lib64; ln -s $P/bin $C/bin
# build JAX-AMG against the env's jaxlib (0.9.2), no dependency changes
printf "jax==0.9.2\njaxlib==0.9.2\n" > pin.txt
PIP_CONSTRAINT=pin.txt CUDA_HOME=$C AMGX_ROOT=~/opt/src/AMGX PATH=$C/bin:$PATH \
  micromamba run -n zoomy-cuda python -m pip install --no-deps "jaxamg==0.1.4"
# at run time
export LD_LIBRARY_PATH=~/opt/src/AMGX/build:$C/lib64:$LD_LIBRARY_PATH
```

Pixi: AmgX is not on conda-forge; Pixi could run the AmgX CMake build as a
task (or a rattler-build recipe) and keep JAX-AMG an optional feature.

Check: `gates/gate14_amgx.py` (GMRES vs AmgX, same template).
