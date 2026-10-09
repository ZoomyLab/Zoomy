"""Animations: (1) 1-D dam break, single domain vs preCICE halves vs exact
Stoker solution; (2) 2-D radial dry dam break with MOOD."""
import os, re, shutil, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import brentq
from zoomy_core.postprocessing import style
import common as C
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
T, DT_W, WINDOW, G = 0.1, 0.004, 1e-3, 9.81
style.use("screen")


def stoker(x, t, hl=1.0, hr=0.5, x0=0.5, g=G):
    """Exact wet-bed dam break (rarefaction + shock)."""
    cl = np.sqrt(g * hl)
    f = lambda hm: (2 * (cl - np.sqrt(g * hm))
                    - (hm - hr) * np.sqrt(g * (hm + hr) / (2 * hm * hr)))
    hm = brentq(f, hr, hl)
    um = 2 * (cl - np.sqrt(g * hm))
    s = hm * um / (hm - hr)
    if t == 0:
        return np.where(x < x0, hl, hr)
    xi = (x - x0) / t
    return np.where(xi < -cl, hl, np.where(xi < um - np.sqrt(g * hm),
                    (2 * cl - xi) ** 2 / (9 * g), np.where(xi < s, hm, hr)))


def frames_of(solver, Q, A):
    out = []
    solver.on_write = lambda t, Q, A: out.append((float(t), np.asarray(Q)))
    return out


# ── single domain ──────────────────────────────────────────────────────────
single = []
mesh, nsm = C.dambreak_1d(0, 2, nx=200)
s = SymbolicHyperbolicSolverJax(time_end=T, cfl=0.9, dt_write=DT_W,
                                on_write=lambda t, Q, A: single.append((float(t), np.asarray(Q))))
Q0, A0 = s.setup(mesh, nsm)
s.run(Q0, A0)
import jax; jax.effects_barrier()

# ── preCICE halves ─────────────────────────────────────────────────────────
run = os.path.join(HERE, "run_animation")
shutil.rmtree(run, ignore_errors=True); os.makedirs(run)
cfg = open(os.path.join(HERE, "precice-config.xml")).read()
cfg = re.sub(r'time-window-size value="[^"]*"', f'time-window-size value="{WINDOW}"', cfg)
cfg = re.sub(r'max-time value="[^"]*"', f'max-time value="{T}"', cfg)
open(os.path.join(run, "precice-config.xml"), "w").write(cfg)
env = {**os.environ, "PYTHONPATH": HERE + os.pathsep + os.environ["PYTHONPATH"]}
procs = [subprocess.Popen([sys.executable, os.path.join(HERE, "participant.py"), n, "2",
                           f"{n}.npy", "window", str(DT_W)], cwd=run, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
         for n in ("Left", "Right")]
outs = [p.communicate(timeout=1800)[0] for p in procs]
assert not any(p.returncode for p in procs), "\n".join(o[-2000:] for o in outs)
L = np.load(f"{run}/Left_frames.npz"); R = np.load(f"{run}/Right_frames.npz")
assert np.allclose(L["t"], R["t"]) and len(L["t"]) == len(single), (L["t"], R["t"], len(single))

x = (np.arange(200) + 0.5) / 200
xf = np.linspace(0, 1, 2000)
h = nsm.sm.roles.depth
m0 = single[0][1][h].sum()
drift = max(abs(np.concatenate([L["Q"][k][h], R["Q"][k][h]]).sum() - m0) / 200
            for k in range(len(single)))
print(f"{len(single)} frames; coupled mass drift max {drift:.1e}; final "
      f"max|coupled-single| {np.abs(np.concatenate([L['Q'][-1], R['Q'][-1]], 1) - single[-1][1]).max():.2e}; "
      f"final max|single-exact| {np.abs(single[-1][1][h] - stoker(x, single[-1][0])).max():.2e}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ex, = ax.plot([], [], color="0.4", lw=1.5, marker="", label="exact (Stoker)")
sd, = ax.plot([], [], color=style.CYCLE[0], lw=2.5, marker="", label="single domain")
cp, = ax.plot([], [], color=style.CYCLE[1], lw=1.5, ls="--", marker="", label="preCICE halves")
ax.axvline(0.5, color="0.6", lw=1, ls=":")
ax.set(xlim=(0, 1), ylim=(0.45, 1.05), xlabel="x", ylabel="h")
ax.legend(loc="upper right")
ttl = ax.set_title("")


def draw(k):
    t, Q = single[k]
    ex.set_data(xf, stoker(xf, t))
    sd.set_data(x, Q[h])
    cp.set_data(x, np.concatenate([L["Q"][k][h], R["Q"][k][h]]))
    ttl.set_text(f"Dam break, order 2, t = {t:.3f} s")
    return ex, sd, cp, ttl


FuncAnimation(fig, draw, frames=len(single), blit=False).save(
    os.path.join(RES, "dambreak_precice.gif"), writer=PillowWriter(fps=6))
plt.close(fig)

# ── 2-D radial dry dam break ───────────────────────────────────────────────
frames2 = []
mesh, nsm = C.dry_dambreak_2d()
s2 = SymbolicHyperbolicSolverJax(time_end=0.06, cfl=0.45, reconstruction_variables="eta", mood=True,
                                 dt_write=0.003,
                                 on_write=lambda t, Q, A: frames2.append((float(t), np.asarray(Q))))
Q0, A0 = s2.setup(mesh, nsm)
s2.run(Q0, A0)
jax.effects_barrier()
cc = np.asarray(mesh.cell_centers)[:, :48 * 48]
xs, ys = np.unique(np.round(cc[0], 10)), np.unique(np.round(cc[1], 10))
ix, iy = np.searchsorted(xs, np.round(cc[0], 10)), np.searchsorted(ys, np.round(cc[1], 10))
fig, ax = plt.subplots(figsize=(6, 5))
H = np.zeros((48, 48))
im = ax.pcolormesh(xs, ys, H, cmap=style.CMAP_CONTINUOUS, vmin=0, vmax=2.0, shading="nearest")
fig.colorbar(im, ax=ax, label="h")
ax.set(aspect="equal", xlabel="x", ylabel="y")
ttl2 = ax.set_title("")
m2 = frames2[0][1][1].sum()


def draw2(k):
    t, Q = frames2[k]
    H[iy, ix] = Q[1]
    im.set_array(H.ravel())
    ttl2.set_text(f"Radial dry dam break, MOOD, t = {t:.3f} s")
    return im, ttl2


FuncAnimation(fig, draw2, frames=len(frames2), blit=False).save(
    os.path.join(RES, "radial_dry_dambreak.gif"), writer=PillowWriter(fps=6))
print(f"2-D: {len(frames2)} frames, h_min {min(f[1][1].min() for f in frames2):.2e}, "
      f"mass drift {max(abs(f[1][1].sum() - m2) for f in frames2) / m2:.1e} (relative)")
