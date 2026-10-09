import re, time as wall, jax, jax.numpy as jnp
import gate7_chorin as g7
s, Q, A = g7.new_solver(1)
m, g, rt, p = s.module, s.mesh, s.rt, s.parameters
fc0 = None
def cfl(Q, A):
    return m["face_wavespeed"](rt.eigenvalues_cfl, Q, A, p, g)
def flux(Q, A):
    AL, AR = m["face_aux"](A, g)
    return m["faces"](rt.numerical_face_sides, Q[:, :1].repeat(g.face_normals.shape[1], 1), Q[:, :1].repeat(g.face_normals.shape[1], 1), AL, AR, p, g)
for name, f in (("CFL wave speed (eigenvalues_cfl)", cfl), ("Riemann solver (numerical_face_sides)", flux)):
    jf = jax.jit(f); txt = jf.lower(Q, A).compile().as_text()
    n_eig = len(re.findall(r'custom_call_target="[^"]*(?:geev|eig)[^"]*"', txt))
    jax.block_until_ready(jf(Q, A)); t0 = wall.perf_counter()
    for _ in range(200): out = jf(Q, A)
    jax.block_until_ready(out)
    print(f"{name}: {n_eig} eig calls, {1e3 * (wall.perf_counter() - t0) / 200:.3f} ms per call")
