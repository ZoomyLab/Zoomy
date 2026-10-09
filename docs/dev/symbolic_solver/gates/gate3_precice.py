"""Gate 3: hyperbolic_precice, two JAX participants (preCICE parallel
explicit) vs the single-domain march.
  A  subcycling=False, window 1e-3 (<= CFL step): O1 == single domain (same dt),
     mass exact at O1 and O2.
  B  subcycling=False, window 4e-3 (> CFL step): warns, then goes unstable.
  C  subcycling=True,  window 4e-3: runs, mass drift reported."""
import os, re, shutil, subprocess, sys
import numpy as np
import common as C
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

HERE = os.path.dirname(os.path.abspath(__file__))
T, h, dx = 0.05, 1, 1 / 200


def couple(order, window, mode):
    run = os.path.join(HERE, f"run_{mode}_{window}_o{order}")
    shutil.rmtree(run, ignore_errors=True); os.makedirs(run)
    cfg = open(os.path.join(HERE, "precice-config.xml")).read()
    cfg = re.sub(r'time-window-size value="[^"]*"', f'time-window-size value="{window}"', cfg)
    cfg = re.sub(r'max-time value="[^"]*"', f'max-time value="{T}"', cfg)
    open(os.path.join(run, "precice-config.xml"), "w").write(cfg)
    env = {**os.environ, "PYTHONPATH": HERE + os.pathsep + os.environ["PYTHONPATH"]}
    procs = [subprocess.Popen([sys.executable, os.path.join(HERE, "participant.py"), n,
                               str(order), f"{n}.npy", mode], cwd=run, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for n in ("Left", "Right")]
    outs = [p.communicate(timeout=1200)[0] for p in procs]
    if any(p.returncode for p in procs):
        msg = [l for o in outs for l in o.splitlines() if "exceeds the CFL" in l]
        return None, (msg[0].strip()[-110:] if msg else "no CFL warning") + " / run failed"
    return np.concatenate([np.load(f"{run}/Left.npy"), np.load(f"{run}/Right.npy")], 1), ""


def single(order, fixed_dt=None):
    mesh, nsm = C.dambreak_1d(0, order, nx=200)
    s = SymbolicHyperbolicSolverJax(time_end=T, cfl=0.9, dt=fixed_dt)
    Q0, A0 = s.setup(mesh, nsm)
    return np.asarray(Q0), np.asarray(s.run(Q0, A0)[1])


if __name__ == "__main__":
    for order in (1, 2):
        Q0, Qs = single(order, fixed_dt=1e-3)
        m0 = Q0[h].sum() * dx
        Qc, _ = couple(order, 1e-3, "window")
        print(f"A  O{order} window 1e-3, no subcycling: max|coupled-single| "
              f"{np.abs(Qc - Qs).max():.2e}, mass drift {abs(Qc[h].sum() * dx - m0):.1e}")
    Qc, msg = couple(1, 4e-3, "window")
    print(f"B  O1 window 4e-3, no subcycling: {msg if Qc is None else 'ran'}")
    for order in (1, 2):
        Q0, Qs = single(order)
        Qc, _ = couple(order, 4e-3, "subcycle")
        print(f"C  O{order} window 4e-3, subcycling: max|coupled-single| "
              f"{np.abs(Qc - Qs).max():.2e}, mass drift {abs(Qc[h].sum() * dx - Q0[h].sum() * dx):.1e}")
