import time as wall, numpy as np, jax, jax.numpy as jnp
import gate7_chorin as g7
s, Q0, A0 = g7.new_solver(1)
fr = np.load("../results/escalante_bump_symbolic_frames.npz")
m, g, rt, p = s.module, s.mesh, s.rt, s.parameters
for t_idx in (0, 20, -1):
    Q = jnp.asarray(fr["Q"][t_idx]); A = s.initial_aux(Q, A0)
    AL, AR = m["face_aux"](A, g)
    nf = g.face_normals.shape[1]
    fc0 = np.asarray(s.mesh_np.face_cells[0]); fc1 = np.minimum(np.asarray(s.mesh_np.face_cells[1]), Q.shape[1] - 1)
    QL, QR = Q[:, fc0], Q[:, fc1]
    fs = {"CFL wave speed": jax.jit(lambda Q, A: m["face_wavespeed"](rt.eigenvalues_cfl, Q, A, p, g)),
          "Riemann solver": jax.jit(lambda Q, A: m["faces"](rt.numerical_face_sides, QL, QR, AL, AR, p, g))}
    out = []
    for name, f in fs.items():
        jax.block_until_ready(f(Q, A)); t0 = wall.perf_counter()
        for _ in range(100): r = f(Q, A)
        jax.block_until_ready(r); out.append(f"{name} {1e3 * (wall.perf_counter() - t0) / 100:.2f} ms")
    print(f"t = {float(fr['t'][t_idx]):5.1f} s:", " | ".join(out))
