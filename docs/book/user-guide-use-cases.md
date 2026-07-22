# Use Cases

## Teaching and Tutorials

**Problem:** Distribute pre-configured simulations to students that they can run and explore immediately.

**Solution:** Create a Zoomy project with multiple sessions (one per exercise), save as ZIP, host alongside the GUI. Share a single URL:

```
https://mbd-rwth.github.io/Zoomy/gui/?project=tutorials/course-exercises.zip
```

Students open the link, see all sessions in the sidebar, select one, and click Run. No installation required -- everything runs in the browser via Pyodide.

For permanent archival (e.g., linked from a publication), upload the ZIP to Zenodo:

```
https://mbd-rwth.github.io/Zoomy/gui/?project=zenodo:12345/exercises.zip
```

## Reproducible Research

**Problem:** A publication requires that simulation results are reproducible from a single link.

**Solution:** Save the exact model + mesh + solver configuration as a Zoomy project ZIP. Upload to Zenodo (gets a DOI). Include the link in the paper.

The Zenodo record preserves the configuration permanently. The GUI loads it and reproduces the simulation in-browser.

## Model Exploration and Debugging

**Problem:** Understand how a model behaves before committing to a large simulation run.

**Solution:** Open the model's code editor in the GUI, use `display()` calls to inspect intermediate state:

```python
model = SME(level=2)
display(model.describe())                    # show model equations
display(mermaid="graph LR; h-->hu-->hv")     # visualize structure
display(latex=model.flux_expression())       # render flux
```

Enable **auto** mode. As you add `display()` calls, the script re-runs and output cells update live.

## Backend Comparison

**Problem:** Compare the same simulation across different solver backends (NumPy vs JAX vs AMReX).

**Solution:** Create one session per backend:

| Session | Model | Mesh | Solver |
|---------|-------|------|--------|
| NumPy baseline | SWE L0 | 1D (200 cells) | NumPy |
| JAX comparison | SWE L0 | 1D (200 cells) | JAX |
| AMReX high-res | SWE L0 | 2D (100x100) | AMReX |

Connect the required backends, run each session, compare results on the Dashboard.

## Batch Processing (CLI)

**Problem:** Run many simulations non-interactively, e.g., for convergence studies or CI pipelines.

**Solution:** Use the CLI in a shell script:

```bash
for N in 50 100 200 400; do
  zoomy start
  zoomy select model swe
  zoomy select mesh create-1d
  zoomy select solver numpy
  zoomy run --local
done
```

Or load a pre-configured project and run:

```bash
zoomy load convergence-study.zip
zoomy session switch "Fine mesh"
zoomy run --local
```

## Sharing Configurations

**Problem:** A colleague needs your exact simulation setup.

**Solution:**
- **GUI:** Save project → share the ZIP file (or host it and share the URL)
- **CLI:** `zoomy save my-setup.zip` → send the file

The recipient loads it (`zoomy load` or `?project=` URL) and gets the exact same card selections, parameters, and code.
