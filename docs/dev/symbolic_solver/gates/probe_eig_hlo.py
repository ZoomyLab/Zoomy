import re, numpy as np, jax, jax.numpy as jnp
import gate7_chorin as g7
s, Q0, A0 = g7.new_solver(1)
m, g, rt, p = s.module, s.mesh, s.rt, s.parameters
Q, A = Q0, A0
AL, AR = m["face_aux"](A, g)
fc0 = np.asarray(s.mesh_np.face_cells[0]); fc1 = np.minimum(np.asarray(s.mesh_np.face_cells[1]), Q.shape[1] - 1)
fs = {"cfl": lambda Q, A: m["face_wavespeed"](rt.eigenvalues_cfl, Q, A, p, g),
      "flux": lambda Q, A: m["faces"](rt.numerical_face_sides, Q[:, fc0], Q[:, fc1], AL, AR, p, g)}
for name, f in fs.items():
    txt = jax.jit(f).lower(Q, A).compile().as_text()
    for ln in txt.splitlines():
        if "custom_call_target" in ln and ("eig" in ln or "geev" in ln):
            tgt = re.search(r'custom_call_target="([^"]+)"', ln).group(1)
            shapes = re.findall(r"(?:f64|c128|f32|s32)\[[0-9,]*\]", ln)
            print(f"{name}: {tgt}  shapes {shapes[:6]}")
