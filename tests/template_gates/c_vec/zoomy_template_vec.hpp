// Plain-array ("dmplex-like") C++ backend for the template C-walker gate.
// Every handle kind is a DISTINCT std::vector wrapper, so a kind mix-up in the
// emitted orchestration is a C++ type error, not silent aliasing.  Layout:
// row-major (rows, cells|faces), exactly as numpy.
#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

template <class Tag, class T>
struct Handle : std::vector<T> {
    using std::vector<T>::vector;
};
#define ZOOMY_TAG(name) struct name##_tag {};
ZOOMY_TAG(cell_state) ZOOMY_TAG(cell_aux) ZOOMY_TAG(params) ZOOMY_TAG(face_state)
ZOOMY_TAG(face_aux) ZOOMY_TAG(face_scalar) ZOOMY_TAG(face_mask) ZOOMY_TAG(cell_scalar)
ZOOMY_TAG(cell_mask) ZOOMY_TAG(cell_grad) ZOOMY_TAG(bc_state) ZOOMY_TAG(bc_aux)
ZOOMY_TAG(array) ZOOMY_TAG(mask)
using CellState = Handle<cell_state_tag, double>;
using CellAux = Handle<cell_aux_tag, double>;
using Params = Handle<params_tag, double>;
using FaceState = Handle<face_state_tag, double>;
using FaceAux = Handle<face_aux_tag, double>;
using FaceScalar = Handle<face_scalar_tag, double>;
using FaceMask = Handle<face_mask_tag, int>;
using CellScalar = Handle<cell_scalar_tag, double>;
using CellMask = Handle<cell_mask_tag, int>;
using CellGrad = Handle<cell_grad_tag, double>;
using BCState = Handle<bc_state_tag, double>;
using BCAux = Handle<bc_aux_tag, double>;
using Array = Handle<array_tag, double>;
using Mask = Handle<mask_tag, int>;
struct Geom { int n_cells = 0; double dx = 0.0; };
using ObsAcc = double;
struct PluginState {};
struct AnyHandle {};

// ── F2: elementwise hook ──────────────────────────────────────────────────
template <class F, class T, class... Ops>
inline void zoomy_elementwise(F&& f, T& t, const Ops&... ops) {
    assert(((ops.size() == t.size()) && ...));
    for (std::size_t i = 0; i < t.size(); ++i) f(t[i], ops[i]...);
}
template <class T>
inline void zoomy_copy(T& dst, const T& src) { dst = src; }

// ── reductions (global; single process here) ──────────────────────────────
template <class C, class F, class A, class... Ops>
inline bool zoomy_reduce_any(C&, F&& f, const A& a, const Ops&... ops) {
    for (std::size_t i = 0; i < a.size(); ++i) if (f(a[i], ops[i]...)) return true;
    return false;
}
template <class C, class F, class A, class... Ops>
inline bool zoomy_reduce_all(C&, F&& f, const A& a, const Ops&... ops) {
    for (std::size_t i = 0; i < a.size(); ++i) if (!f(a[i], ops[i]...)) return false;
    return true;
}
template <class C, class F, class A, class... Ops>
inline double zoomy_reduce_min(C&, F&& f, const A& a, const Ops&... ops) {
    double m = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < a.size(); ++i) m = std::min(m, (double)f(a[i], ops[i]...));
    return m;
}
template <class C, class F, class A, class... Ops>
inline double zoomy_reduce_max(C&, F&& f, const A& a, const Ops&... ops) {
    double m = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, (double)f(a[i], ops[i]...));
    return m;
}
template <class C, class F, class A, class... Ops>
inline double zoomy_reduce_sum(C&, F&& f, const A& a, const Ops&... ops) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += (double)f(a[i], ops[i]...);
    return s;
}
inline double zoomy_safe_div(double a, double b, double fill) {
    return b != 0.0 ? a / b : fill;
}
template <class H>
inline auto& zoomy_at(H& h, int i, int j) { return h[(std::size_t)i * 1 + j]; }

// ── linear algebra for Root / LinSolve (dense; small problems) ────────────
inline bool zoomy_dense_solve(std::vector<double> A, std::vector<double>& b, int n) {
    for (int k = 0; k < n; ++k) {
        int piv = k;
        for (int i = k + 1; i < n; ++i)
            if (std::fabs(A[i * n + k]) > std::fabs(A[piv * n + k])) piv = i;
        if (A[piv * n + k] == 0.0) return false;
        if (piv != k) {
            for (int j = 0; j < n; ++j) std::swap(A[k * n + j], A[piv * n + j]);
            std::swap(b[k], b[piv]);
        }
        for (int i = k + 1; i < n; ++i) {
            const double f = A[i * n + k] / A[k * n + k];
            for (int j = k; j < n; ++j) A[i * n + j] -= f * A[k * n + j];
            b[i] -= f * b[k];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < n; ++j) s -= A[i * n + j] * b[j];
        b[i] = s / A[i * n + i];
    }
    return true;
}
inline double zoomy_norm(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}
template <class T>
inline void zoomy_project(T& v, const char* nullspace) {
    if (!nullspace) return;
    double m = 0.0;
    for (double x : v) m += x;
    m /= (double)v.size();
    for (auto& x : v) x -= m;
}
// matvec(x, y) -> dense A (column probing)
template <class T, class MV>
inline std::vector<double> zoomy_probe(MV&& mv, const T& like) {
    const int n = (int)like.size();
    std::vector<double> A((std::size_t)n * n);
    T e = like, y = like;
    for (int j = 0; j < n; ++j) {
        std::fill(e.begin(), e.end(), 0.0);
        e[j] = 1.0;
        mv(e, y);
        for (int i = 0; i < n; ++i) A[(std::size_t)i * n + j] = y[i];
    }
    return A;
}
template <class C, class MV, class P, class T>
inline void zoomy_linsolve(C&, MV&& mv, const T& rhs, T& x, P&&, const char*, double tol,
                           int, int, const char* nullspace, int, bool, bool& ok,
                           double& rel_resid, int& iters) {
    const int n = (int)rhs.size();
    T b = rhs;
    zoomy_project(b, nullspace);
    auto A = zoomy_probe(mv, rhs);
    std::vector<double> s(b.begin(), b.end());
    ok = zoomy_dense_solve(A, s, n);
    x = rhs;
    std::copy(s.begin(), s.end(), x.begin());
    zoomy_project(x, nullspace);
    T Ax = rhs;
    mv(x, Ax);
    std::vector<double> r(n);
    for (int i = 0; i < n; ++i) r[i] = b[i] - Ax[i];
    const double bn = zoomy_norm(std::vector<double>(b.begin(), b.end()));
    rel_resid = zoomy_norm(r) / (bn > 0 ? bn : 1.0);
    ok = ok && rel_resid <= tol;
    iters = 1;
}
template <class C, class R, class J, class T, class P>
inline void zoomy_root(C& ctx, R&& residual, J&& jac, T& x, double tol, int maxiter,
                       const char* jac_kind, P&& pc, const char* lin_kind, double lin_tol,
                       int restart, int lin_maxiter, const char* nullspace, int bw,
                       bool sym, bool& ok, double& rnorm, int& iters) {
    (void)jac_kind; (void)pc;
    const int n = (int)x.size();
    T r = x, xp = x, xm = x, rp = x, rm = x;
    residual(x, r);
    rnorm = zoomy_norm(std::vector<double>(r.begin(), r.end()));
    iters = 0;
    while (iters < maxiter && rnorm > tol) {
        std::vector<double> A((std::size_t)n * n);
        double xmax = 0.0;
        for (double v : x) xmax = std::max(xmax, std::fabs(v));
        const double h = std::cbrt(std::numeric_limits<double>::epsilon()) * (1.0 + xmax);
        for (int j = 0; j < n; ++j) {
            if constexpr (!std::is_same_v<std::decay_t<J>, std::nullptr_t>) {
                T e = x, jv = x;
                std::fill(e.begin(), e.end(), 0.0);
                e[j] = 1.0;
                jac(x, e, jv);
                for (int i = 0; i < n; ++i) A[(std::size_t)i * n + j] = jv[i];
            } else {
                xp = x; xm = x;
                xp[j] += h; xm[j] -= h;
                residual(xp, rp);
                residual(xm, rm);
                for (int i = 0; i < n; ++i) A[(std::size_t)i * n + j] = (rp[i] - rm[i]) / (2 * h);
            }
        }
        std::vector<double> d(r.begin(), r.end());
        for (auto& v : d) v = -v;
        if (!zoomy_dense_solve(A, d, n)) break;
        for (int i = 0; i < n; ++i) x[i] += d[i];
        residual(x, r);
        rnorm = zoomy_norm(std::vector<double>(r.begin(), r.end()));
        ++iters;
    }
    ok = rnorm <= tol;
    (void)ctx; (void)lin_kind; (void)lin_tol; (void)restart; (void)lin_maxiter;
    (void)nullspace; (void)bw; (void)sym;
}

struct Ctx;
#include "template_decls.hpp"   // EMITTED: constants, workspace, declarations

struct Ctx {
    int nv = 1, naux = 1, n_cells = 0, n_faces = 0, nb = 2, dim = 1;
    TemplateWorkspace ws;
    std::FILE* out = nullptr;
    long stage_calls = 0;
};

inline void zoomy_guard_value(Ctx&, const char* n, double v) { std::fprintf(stderr, "  %s = %.17g\n", n, v); }
inline void zoomy_guard_value(Ctx&, const char* n, int v) { std::fprintf(stderr, "  %s = %d\n", n, v); }
inline void zoomy_guard_value(Ctx&, const char* n, bool v) { std::fprintf(stderr, "  %s = %d\n", n, (int)v); }
template <class H>
inline void zoomy_guard_value(Ctx&, const char* n, const H&) { std::fprintf(stderr, "  %s = <array>\n", n); }
inline void zoomy_guard_fail(Ctx&, const char* msg) {
    std::fprintf(stderr, "GUARD FAILED: %s\n", msg);
    std::exit(3);
}
inline void zoomy_warn(Ctx&, const char* msg) { std::fprintf(stderr, "WARNING: %s\n", msg); }

inline void zoomy_alloc_cell_state(Ctx& c, CellState& a) { a.assign((std::size_t)c.nv * c.n_cells, 0.0); }
inline void zoomy_alloc_cell_aux(Ctx& c, CellAux& a) { a.assign((std::size_t)c.naux * c.n_cells, 0.0); }
inline void zoomy_alloc_face_state(Ctx& c, FaceState& a) { a.assign((std::size_t)c.nv * c.n_faces, 0.0); }
inline void zoomy_alloc_face_aux(Ctx& c, FaceAux& a) { a.assign((std::size_t)c.naux * c.n_faces, 0.0); }
inline void zoomy_alloc_face_scalar(Ctx& c, FaceScalar& a) { a.assign(c.n_faces, 0.0); }
inline void zoomy_alloc_face_mask(Ctx& c, FaceMask& a) { a.assign(c.n_faces, 0); }
inline void zoomy_alloc_cell_scalar(Ctx& c, CellScalar& a) { a.assign(c.n_cells, 0.0); }
inline void zoomy_alloc_cell_mask(Ctx& c, CellMask& a) { a.assign(c.n_cells, 0); }
inline void zoomy_alloc_cell_grad(Ctx& c, CellGrad& a) { a.assign((std::size_t)c.nv * c.dim * c.n_cells, 0.0); }
inline void zoomy_alloc_bc_state(Ctx& c, BCState& a) { a.assign((std::size_t)c.nv * c.nb, 0.0); }
inline void zoomy_alloc_bc_aux(Ctx& c, BCAux& a) { a.assign((std::size_t)c.naux * c.nb, 0.0); }
inline void zoomy_alloc_array(Ctx& c, Array& a) { a.assign(c.n_cells, 0.0); }
inline void zoomy_alloc_mask(Ctx& c, Mask& a) { a.assign(c.n_cells, 0); }
inline void zoomy_alloc_obs(Ctx&, ObsAcc& a) { a = 0.0; }
inline void zoomy_alloc_geom(Ctx&, Geom&) {}
inline void zoomy_alloc_plugin(Ctx&, PluginState&) {}
