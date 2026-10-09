"""Gate 3b: preCICE with ADAPTIVE dt and a window LONGER than the CFL step, so
each participant subcycles (different step counts per side).  Checks mass."""
import os, re, shutil, subprocess, sys
import numpy as np
import common as C
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW, T = float(os.environ.get("WINDOW", "4e-3")), 0.05
for order in (1, 2):
    mesh, nsm = C.dambreak_1d(0, order, nx=200)
    s = SymbolicHyperbolicSolverJax(time_end=T, cfl=0.9)
    Q0, A0 = s.setup(mesh, nsm)
    _, Qm, _ = s.run(Q0, A0)
    run = os.path.join(HERE, f"run_subcycle_o{order}")
    shutil.rmtree(run, ignore_errors=True); os.makedirs(run)
    cfg = open(os.path.join(HERE, "precice-config.xml")).read()
    cfg = re.sub(r'time-window-size value="[^"]*"', f'time-window-size value="{WINDOW}"', cfg)
    open(os.path.join(run, "precice-config.xml"), "w").write(cfg)
    env = {**os.environ, "PYTHONPATH": HERE + os.pathsep + os.environ["PYTHONPATH"]}
    procs = [subprocess.Popen([sys.executable, os.path.join(HERE, "participant.py"), n,
                               str(order), f"{n}.npy", "adaptive"], cwd=run, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for n in ("Left", "Right")]
    outs = [p.communicate(timeout=1200)[0] for p in procs]
    if any(p.returncode for p in procs):
        print("\n".join(o[-3000:] for o in outs)); raise SystemExit("failed")
    Qc = np.concatenate([np.load(f"{run}/Left.npy"), np.load(f"{run}/Right.npy")], 1)
    h, dx = nsm.sm.roles.depth, 1 / 200
    m0 = float(np.asarray(Q0)[h].sum()) * dx
    print(f"order {order}, window {WINDOW}, adaptive dt: mass drift "
          f"{abs(Qc[h].sum() * dx - m0):.2e} (single domain {abs(float(np.asarray(Qm)[h].sum()) * dx - m0):.1e}), "
          f"max|coupled-single| {np.abs(Qc - np.asarray(Qm)).max():.2e}; "
          + "; ".join(l for o in outs for l in o.splitlines() if " t = " in l))
