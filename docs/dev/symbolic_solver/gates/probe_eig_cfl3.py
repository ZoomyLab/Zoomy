"""Where the time goes in the CFL wave speed vs the Riemann solver (VAM bump,
t = 10 s state): profiler split into eigensolve vs the rest."""
import glob, gzip, json, collections, shutil
import numpy as np, jax, jax.numpy as jnp
import gate7_chorin as g7
TR = "/tmp/claude-765404697/-mnt-userdrive-Users-home-adam-obbpb5az1dhsjzf-git-zotero-rag/b5667895-053e-413b-8d69-8137a92fc71c/scratchpad/trace_eig"
s, Q0, A0 = g7.new_solver(1)
fr = np.load("../results/escalante_bump_symbolic_frames.npz")
m, g, rt, p = s.module, s.mesh, s.rt, s.parameters
Q = jnp.asarray(fr["Q"][20]); A = s.initial_aux(Q, A0)
AL, AR = m["face_aux"](A, g)
fc0 = np.asarray(s.mesh_np.face_cells[0]); fc1 = np.minimum(np.asarray(s.mesh_np.face_cells[1]), Q.shape[1] - 1)
QL, QR = Q[:, fc0], Q[:, fc1]
fs = {"cfl": jax.jit(lambda Q, A: m["face_wavespeed"](rt.eigenvalues_cfl, Q, A, p, g)),
      "flux": jax.jit(lambda Q, A: m["faces"](rt.numerical_face_sides, QL, QR, AL, AR, p, g))}
for name, f in fs.items():
    jax.block_until_ready(f(Q, A))
    d = f"{TR}_{name}"; shutil.rmtree(d, ignore_errors=True)
    with jax.profiler.trace(d):
        for _ in range(50):
            out = f(Q, A)
        jax.block_until_ready(out)
    ev = json.load(gzip.open(glob.glob(f"{d}/**/*.trace.json.gz", recursive=True)[0]))["traceEvents"]
    agg = collections.Counter(); cnt = collections.Counter()
    for e in ev:
        if e.get("ph") == "X" and "dur" in e:
            agg[e["name"]] += e["dur"]; cnt[e["name"]] += 1
    eig = sum(v for k, v in agg.items() if k.startswith("eig"))
    neig = sum(v for k, v in cnt.items() if k.startswith("eig"))
    tot = agg.get("ThunkExecutor::Execute", 0)
    top = [(k[:28], round(v / 50, 1)) for k, v in agg.most_common(30) if not k.startswith(("eig", "Thunk", "$"))][:6]
    print(f"{name}: per call {tot/50:.0f} us total, eig {eig/50:.0f} us in {neig//50} eig ops, rest {(tot-eig)/50:.0f} us; top rest {top}")
