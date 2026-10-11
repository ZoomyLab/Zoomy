"""Gate 3's coupled run with the JAX Left participant on R ranks (Right on
RR ranks) vs the 1+1 rank coupled run and the single domain.
    python gate28_jax_couple_parallel.py CASE ORDER WINDOW MODE R RR"""
import os, re, shutil, signal, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = HERE
case, order, window, mode, R, RR = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
T = 0.05


def config(run, n_state):
    cfg = open(os.path.join(G, "precice-config.xml")).read()
    cfg = re.sub(r'time-window-size value="[^"]*"', f'time-window-size value="{window}"', cfg)
    cfg = re.sub(r'max-time value="[^"]*"', f'max-time value="{T}"', cfg)
    if case == "2d":
        cfg = cfg.replace('dimensions="2"', 'dimensions="2"')
        for side in "LR":
            cfg = cfg.replace(f'<data:scalar name="{side}2" />',
                              f'<data:scalar name="{side}2" /><data:scalar name="{side}3" />')
            cfg = cfg.replace(f'<use-data name="{side}2" />',
                              f'<use-data name="{side}2" /><use-data name="{side}3" />')
        cfg = re.sub(r'(<write-data name="(\w)2" mesh="([\w-]+)" />)',
                     r'\1<write-data name="\g<2>3" mesh="\3" />', cfg)
        cfg = re.sub(r'(<read-data name="(\w)2" mesh="([\w-]+)" />)',
                     r'\1<read-data name="\g<2>3" mesh="\3" />', cfg)
        cfg = re.sub(r'(<exchange data="(\w)2" mesh="([\w-]+)" from="(\w+)" to="(\w+)" initialize="true" />)',
                     r'\1<exchange data="\g<2>3" mesh="\3" from="\4" to="\5" initialize="true" />', cfg)
    open(os.path.join(run, "precice-config.xml"), "w").write(cfg)


run = os.path.join(HERE, f"run_{case}_o{order}_{window}_{mode}_L{R}_R{RR}")
shutil.rmtree(run, ignore_errors=True); os.makedirs(run)
config(run, None)
env = {**os.environ, "PYTHONPATH": G + os.pathsep + os.environ["PYTHONPATH"]}
procs = []
for name, n in (("Left", R), ("Right", RR)):
    cmd = ["timeout", "-k", "5", "280"] + (["mpiexec", "-n", str(n)] if n > 1 else []) + \
          [sys.executable, os.path.join(HERE, "participant_mpi.py"), name, str(order), f"{name}.npy", mode, case]
    procs.append(subprocess.Popen(cmd, cwd=run, env=env, start_new_session=True,
                                  stdout=open(f"{run}/{name}.log", "w"), stderr=subprocess.STDOUT))
t0 = time.time()
try:
    while any(p.poll() is None for p in procs) and time.time() - t0 < 285:
        if any(p.poll() not in (None, 0) for p in procs):
            break
        time.sleep(0.5)
finally:
    for p in procs:                     # never leave a participant waiting
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
rc = [p.returncode for p in procs]
verts = [l for n in ("Left", "Right") for l in open(f"{run}/{n}.log") if "coupling vertices" in l]
print("".join(sorted(verts)), end="")
if any(rc):
    print(f"FAILED rc={rc}"); sys.exit(1)
Q = np.concatenate([np.load(f"{run}/Left.npy"), np.load(f"{run}/Right.npy")], 1)
np.save(os.path.join(HERE, f"Q_{case}_o{order}_{window}_{mode}_L{R}_R{RR}.npy"), Q)
print(f"{case} O{order} window {window} {mode} Left x{R} Right x{RR}: done [{time.time() - t0:.0f}s]")
