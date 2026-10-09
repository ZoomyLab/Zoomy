"""Gate: one solver source -> JAX (fwd + rev AD) and C++; preCICE JAX<->C++."""
import os
import shutil
import subprocess
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import sympy as sp

import build as B
import kernels_swe as K
import plain_jax
import rt_jax

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.dirname(os.path.dirname(sys.executable))
CXX = ["g++", "-std=c++17", "-O2", f"-I{HERE}", f"-I{ENV}/include"]
LINK = [f"-L{ENV}/lib", "-lprecice", f"-Wl,-rpath,{ENV}/lib"]
N, DX = 200, 1.0 / 200
X = (jnp.arange(N) + 0.5) * DX
Q0 = jnp.stack([jnp.where(X < 0.5, 1.0, 0.5), 0.0 * X])
P = dict(g=9.81, cf=0.01, nu=1e-3, cfl=0.45, dx=DX)


def jax_hyp(mode="fwd", **kw):
    return rt_jax.build(B.hyperbolic_src("py", **kw), "hyperbolic",
                        rt_jax.runtime(K.jax_kernels(), mode=mode))


def compile_cpp(src, main, exe, precice=False):
    path = os.path.join(HERE, exe + ".cpp")
    open(path, "w").write(B.cpp_program(src, main, precice))
    subprocess.run(CXX + ["-o", os.path.join(HERE, exe), path] +
                   (LINK if precice else []), check=True)
    return os.path.join(HERE, exe)


def args(hb, T=0.1, **o):
    p = {**P, **o}
    return (Q0, 0.0, T, p["g"], p["cf"], p["nu"], p["cfl"], p["dx"], hb, jnp.int64(0))


print("== 1. hyperbolic: symbolic JAX vs hand-written JAX vs C++ ==")
MAIN = (f" int N = {N}; double dx = 1.0 / N; Field Q(2, N);\n"
        " for (int c = 0; c < N; ++c) Q(0, c) = (c + 0.5) * dx < 0.5 ? 1.0 : 0.5;\n"
        " auto [tf, Qf, tk] = hyperbolic(Q, 0.0, 0.1, {g}, {cf}, {nu}, {cfl}, dx,"
        " std::atof(v[1]), 0L);\n".format(**P) +
        ' std::printf("%ld %.17g\\n", tk, tf);\n'
        ' for (int c = 0; c < N; ++c) std::printf("%.17g %.17g\\n", Qf(0, c), Qf(1, c));\n')
exe = compile_cpp(B.hyperbolic_src("cpp"), MAIN, "gen_hyperbolic")
for hb in (0.0, 0.75):
    ts, Qs, ks = jax.jit(jax_hyp())(*args(hb))
    Qp, tp, kp = jax.jit(plain_jax.make_march(rt_jax.runtime(K.jax_kernels())))(*args(hb)[:-1])
    out = subprocess.run([exe, str(hb)], check=True, capture_output=True, text=True).stdout.split("\n")
    kc, tc = out[0].split()
    Qc = jnp.array([[float(x) for x in l.split()] for l in out[1:] if l]).T
    print(f" hb={hb}: steps {int(ks)}/{int(kp)}/{kc}, max|sym-hand| "
          f"{float(jnp.max(abs(Qs - Qp))):.1e}, max|sym-C++| {float(jnp.max(abs(Qs - Qc))):.1e}")

print("== 2. AD: jvp vs FD, reverse (checkpointed) vs jvp ==")


def J(fn):
    return jax.jit(lambda p: (lambda Q: jnp.sum(Q[0] ** 2 + Q[1] ** 2) * DX)(
        fn(*args(0.0, T=0.05, g=p[0], cf=p[1], nu=p[2]))[1]))


FDT = dict(fixed_dt=sp.Float(1e-3))
p0 = jnp.array([9.81, 0.01, 1e-3])
Jf, Jr = J(jax_hyp("fwd", **FDT)), J(jax_hyp("rev", **FDT))
grad = jax.grad(Jr)(p0)
for i, nm in enumerate(("g", "cf", "nu")):
    e = jnp.zeros(3).at[i].set(1.0)
    d = float(jax.jvp(Jf, (p0,), (e,))[1])
    fd = min(abs(float((Jf(p0 + r * p0[i] * e) - Jf(p0 - r * p0[i] * e)) / (2 * r * p0[i])) - d)
             for r in (1e-3, 1e-4, 1e-5, 1e-6)) / abs(d)
    print(f" dJ/d{nm} = {d: .10e}: |FD-jvp| {fd:.1e}, |rev-jvp| {abs(float(grad[i]) - d) / abs(d):.1e}")

print("== 3. preCICE: monolith vs coupled halves (order 1, dt = window) ==")
mono = jax.jit(jax_hyp(order=1, fixed_dt=sp.Float(1e-3)))(*args(0.0, T=0.05, nu=0.0))[1]
run = os.path.join(HERE, "run_precice")
shutil.rmtree(run, ignore_errors=True)
os.makedirs(run)
shutil.copy(os.path.join(HERE, "precice-config.xml"), run)
PMAIN = (' std::string name = v[1]; std::string me = name == "Left" ? "L" : "R",'
         ' pe = name == "Left" ? "R" : "L";\n'
         ' cpl_setup(name, "precice-config.xml", name + "-Mesh", {me + "0", me + "1"},'
         ' {pe + "0", pe + "1"}, 0.5);\n'
         ' int N = 100; Field Q(2, N); for (int c = 0; c < N; ++c) Q(0, c) = std::atof(v[2]);\n'
         ' auto [tf, Qf, tk] = hyperbolic_precice(Q, 0.0, 9.81, 0.01, 0.0, 0.45, 1.0 / 200, 0.0, 0L);\n'
         ' cpl().p->finalize();\n FILE* f = std::fopen(v[3], "w");\n'
         ' for (int c = 0; c < N; ++c) std::fprintf(f, "%.17g %.17g\\n", Qf(0, c), Qf(1, c));\n'
         ' std::fclose(f); std::printf("%s: t=%.15g effects=%ld\\n", v[1], tf, tk);\n')
cpp_right = compile_cpp(B.coupled_src("cpp", 0, order=1, fixed_dt=sp.Float(1e-3)),
                        PMAIN, "gen_precice_right", precice=True)
py = [sys.executable, os.path.join(HERE, "participant_jax.py")]
env = {**os.environ, "PYTHONPATH": HERE}


def couple(right_cmd, label):
    t0 = time.time()
    pl = subprocess.Popen(py + ["Left", "1", "1.0", "L.npy"], cwd=run, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pr = subprocess.Popen(right_cmd, cwd=run, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ol, _ = pl.communicate(timeout=600)
    orr, _ = pr.communicate(timeout=600)
    if pl.returncode or pr.returncode:
        print(ol[-2000:], orr[-2000:])
        raise SystemExit(f"{label}: participant failed")
    QL = np.load(os.path.join(run, "L.npy"))
    QR = (np.loadtxt(os.path.join(run, "R.txt")).T if right_cmd[0] == cpp_right
          else np.load(os.path.join(run, "R.npy")))
    Qc = np.concatenate([QL, QR], axis=1)
    m0 = float(jnp.sum(Q0[0]) * DX)
    print(f" {label}: max|coupled-mono| {np.max(abs(Qc - np.asarray(mono))):.1e}, "
          f"mass drift {abs(Qc[0].sum() * DX - m0):.1e}, {time.time() - t0:.0f}s")
    for o in (ol, orr):
        print("   " + [l for l in o.splitlines() if "effects=" in l][-1])


couple([cpp_right, "Right", "0.5", "R.txt"], "JAX(Left) <-> C++(Right)")
couple(py + ["Right", "0", "0.5", "R.npy"], "JAX(Left) <-> JAX(Right)")

print("== 4. lines (non-blank, non-comment) ==")
for f in ("solvers.py", "plain_jax.py", "vocabulary.py", "symsolve.py",
          "rt_jax.py", "rt.hpp", "rt_precice.hpp"):
    n = sum(1 for l in open(os.path.join(HERE, f))
            if l.strip() and not l.strip().startswith(("#", "//", '"""')))
    print(f" {f}: {n}")
