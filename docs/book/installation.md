# Install

Zoomy is a set of Python packages on PyPI. It runs on Linux, Windows and macOS. The JAX backend
needs Linux, Windows or a Mac with Apple silicon (JAX provides no packages for Intel Macs).

## Python environment

Python 3.11 or newer ([python.org](https://www.python.org/downloads/), or the package manager of
your system). Install Zoomy into its own virtual environment.

Linux and macOS:

```bash
python3 -m venv zoomy-env
source zoomy-env/bin/activate
```

Windows (PowerShell):

```powershell
py -m venv zoomy-env
zoomy-env\Scripts\Activate.ps1
```

If PowerShell refuses to run the script, allow it once with
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. A conda, mamba or pixi environment works
the same way: activate it and continue with `pip`.

## Minimal

```bash
pip install zoomy-core zoomy-plotting jupyterlab
curl -LO https://raw.githubusercontent.com/ZoomyLab/Zoomy/main/tutorials/swe/simple_numpy.ipynb
jupyter lab simple_numpy.ipynb
```

On Windows write `curl.exe` instead of `curl`. Every tutorial page also has a download button
(*.ipynb*) at the top.

- More notebooks: [Tutorials](tutorials/index.md)
- The GUI: [Using the GUI](guide/gui.md)
- Nothing to install: [In the browser](browser.md)

## JAX

```bash
pip install zoomy-jax
curl -LO https://raw.githubusercontent.com/ZoomyLab/Zoomy/main/tutorials/swe/bump_2d.ipynb
jupyter lab bump_2d.ipynb
```

First JAX notebook: {doc}`2-D bump, NumPy and JAX <tutorials/ipynb/swe/bump_2d>`. Sensitivities
by automatic differentiation: the tutorials tagged
<a href="tutorials/index.html#feature=sensitivity">sensitivity</a>.

JAX on an NVIDIA GPU (Linux, on Windows inside WSL2):

```bash
pip install "zoomy-jax[cuda]"
```

## VS Code instead of JupyterLab

Install the extensions *Python* and *Jupyter*, open the notebook, then *Select Kernel* (top
right) → *Python Environments* → `zoomy-env`. If it is not listed, *Python: Select Interpreter*
→ *Enter interpreter path* → `zoomy-env/bin/python` (Windows `zoomy-env\Scripts\python.exe`).

## Other backends

OpenFOAM, AMReX, MPI and GPU images: [Containers](containers.md). Which backend for what:
[Backends](guide/backends.md).
