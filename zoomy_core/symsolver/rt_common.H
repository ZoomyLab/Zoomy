#pragma once
/*---------------------------------------------------------------------------*\
    rt_common.H -- the backend-neutral part of the C++ runtimes of the
    symbolic solvers (foam rt.H, AMReX rt_amrex.H; both include it first):

      - the elementwise functors, NaN-propagating minimum / maximum;
      - Settings / Implicit: what the printed ``configure()`` sets
        (zoomy_core.symsolver.backend_cxx);
      - the effects io_write (``writer()``) and check_cfl;
      - Newton per cell (its skeleton and the small pivoted solve) and
        global (Jacobian-free Newton-Krylov);
      - the preCICE call sequence and its members (ZS_PRECICE builds).

    The runtime supplies, in namespace zs: ``struct Field`` (rows ``nq``),
    ``warn(message)``, and for the Newton skeletons (found when they are
    instantiated) ``norm(F)`` = ||.||_2 over the own cells of all ranks,
    ``perturb(x, k)`` -> (x + h e_k, h), ``newton_update(x, r, cols, hs)``
    and ``linsolve(op, b)``.  ``ZS_HD`` marks what a device loop calls
    (AMReX: AMREX_GPU_HOST_DEVICE).
\*---------------------------------------------------------------------------*/
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#ifdef ZS_PRECICE
#include <precice/precice.hpp>
#endif

#ifndef ZS_HD
#define ZS_HD
#endif

namespace zs
{
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr double SQRT_EPS = 1.4901161193847656e-08;    // sqrt(DBL_EPSILON) = 2^-26

// ── elementwise arithmetic ──────────────────────────────────────────────────
struct Add { ZS_HD double operator()(double x, double y) const { return x + y; } };
struct Sub { ZS_HD double operator()(double x, double y) const { return x - y; } };
struct Mul { ZS_HD double operator()(double x, double y) const { return x*y; } };
struct Div { ZS_HD double operator()(double x, double y) const { return x/y; } };
struct Pow { ZS_HD double operator()(double x, double y) const { return std::pow(x, y); } };
struct Abs { ZS_HD double operator()(double x, double) const { return std::abs(x); } };
// NaN propagates (std::min / std::max would swallow it)
struct Min
{
    ZS_HD double operator()(double x, double y) const
    { return (x != x || y != y) ? NaN : (x < y ? x : y); }
};
struct Max
{
    ZS_HD double operator()(double x, double y) const
    { return (x != x || y != y) ? NaN : (x > y ? x : y); }
};
inline double minimum(double a, double b) { return Min{}(a, b); }
inline double maximum(double a, double b) { return Max{}(a, b); }

// ── settings (the printed configure()) ──────────────────────────────────────
struct Settings
{
    int lsq_degree = 1;            // highest derivative order the model asks for
    double tpfa_cos_floor = NaN;   // two-point fluxes: |n.e| raised to this
};
inline Settings& settings() { static Settings s; return s; }

struct Implicit
{
    double tol = 1e-8;        // Newton: ||R||_2 <= tol
    int maxiter = 6;
    double ls_tol = 1e-7;     // GMRES: ||A x - b|| <= ls_tol ||b||
    int ls_restart = 20, ls_maxiter = 30;
    double ls_accept = -1;    // > 0: abort above this relative residual
};
inline Implicit& implicit() { static Implicit s; return s; }

// the runtime's warning (rank 0)
inline void warn(const std::string& message);

template<class... A> std::string text(const A&... a)
{
    std::ostringstream s;
    (s << ... << a);
    return s.str();
}

// ── effects ─────────────────────────────────────────────────────────────────
struct Field;
inline std::function<void(double, const Field&, const Field&)>& writer()
{
    static std::function<void(double, const Field&, const Field&)> w;
    return w;
}

inline std::tuple<long> io_write(long tok, double time, const Field& Q, const Field& Qaux)
{
    if (writer()) writer()(time, Q, Qaux);
    return {tok + 1};
}

inline std::tuple<long> check_cfl(long tok, double dt_cfl, double dt)
{
    static bool warned = false;
    if (dt > dt_cfl*(1 + 1e-12) && !warned)
    {
        warned = true;
        warn(text("time step ", dt, " exceeds the CFL step ", dt_cfl, " (warned once)"));
    }
    return {tok + 1};
}

// ── Newton ──────────────────────────────────────────────────────────────────
// x with A x = b, n x n row-major, partial pivoting; A is overwritten, x in b
ZS_HD inline void solve_small(double* A, double* b, int n)
{
    for (int k = 0; k < n; ++k)
    {
        int piv = k;
        for (int i = k + 1; i < n; ++i)
            if (std::abs(A[i*n + k]) > std::abs(A[piv*n + k])) piv = i;
        if (piv != k)
        {
            for (int j = 0; j < n; ++j)
            {
                const double t = A[k*n + j]; A[k*n + j] = A[piv*n + j]; A[piv*n + j] = t;
            }
            const double t = b[k]; b[k] = b[piv]; b[piv] = t;
        }
        for (int i = k + 1; i < n; ++i)
        {
            const double m = A[i*n + k]/A[k*n + k];
            for (int j = k; j < n; ++j) A[i*n + j] -= m*A[k*n + j];
            b[i] -= m*b[k];
        }
    }
    for (int i = n - 1; i >= 0; --i)
    {
        for (int j = i + 1; j < n; ++j) b[i] -= A[i*n + j]*b[j];
        b[i] /= A[i*n + i];
    }
}

// the one-sided difference step of a variable of value v
ZS_HD inline double fd_step(double v)
{
    const double a = std::abs(v);
    return SQRT_EPS*(a > 1.0 ? a : 1.0);
}

inline void warn_unconverged(bool& warned, const char* what, int it, double r)
{
    if (warned) return;
    warned = true;
    warn(text(what, ": not converged after ", it, " Newton updates, ||R|| = ", r,
              " (warned once)"));
}

// Newton on a cell-local residual f (x -> R, same shape): each cell's n x n
// Jacobian by one-sided differences, one variable at a time in all cells
// (``perturb``), each cell's update by solve_small (``newton_update``);
// ``implicit()`` tol / maxiter on ||R||_2.  (JAX: the exact jvp, the same
// block structure.)
template<class F, class X>
X newton_cellwise(F f, X x)
{
    const auto& o = implicit();
    X r = f(x);
    int it = 0;
    for (; it < o.maxiter && norm(r) > o.tol; ++it)
    {
        std::vector<X> cols(x.nq), hs(x.nq);
        for (int k = 0; k < x.nq; ++k)
        {
            auto ph = perturb(x, k);
            cols[k] = f(ph.first);
            hs[k] = ph.second;
        }
        x = newton_update(x, r, cols, hs);
        r = f(x);
    }
    static bool warned = false;
    if (norm(r) > o.tol) warn_unconverged(warned, "root", it, norm(r));
    return x;
}

// Newton on a global residual R (x -> R, same shape), ``implicit()`` tol /
// maxiter on ||R||_2; each update a linsolve on the Jacobian action by a
// directional difference (Jacobian-free Newton-Krylov; JAX: the exact jvp)
template<class F, class X>
X newton_krylov(F R, X x)
{
    const auto& o = implicit();
    X r = R(x);
    int it = 0;
    for (; it < o.maxiter && norm(r) > o.tol; ++it)
    {
        const X x0 = x, r0 = r;
        const double nx = norm(x0);
        auto J = [&R, x0, r0, nx](const X& v) -> X
        {
            const double nv = norm(v);
            if (!(nv > 0.0)) return X(0.0*v);
            const double h = SQRT_EPS*(1.0 + nx)/nv;
            return X((R(X(x0 + h*v)) - r0)/h);
        };
        x = X(x0 + linsolve(J, X(-r0)));
        r = R(x);
    }
    static bool warned = false;
    if (norm(r) > o.tol) warn_unconverged(warned, "newton_krylov", it, norm(r));
    return x;
}

#ifdef ZS_PRECICE
// ── preCICE ─────────────────────────────────────────────────────────────────
// The app fills ``coupling()`` (participant, data names, vertices); the
// runtime keeps where the interface lies.  Interface values live on the host
// as rows x vertices; the effects only make preCICE API calls.
struct Vertices { int nq = 0, n = 0; std::vector<double> v; };   // v[q n + j]
struct Coupling
{
    std::unique_ptr<precice::Participant> participant;
    std::string mesh;
    std::vector<std::string> write, read;
    std::vector<precice::VertexID> ids;
};
inline Coupling& coupling() { static Coupling c; return c; }

inline void couple_write_data(const Vertices& s)
{
    auto& c = coupling();
    for (size_t i = 0; i < c.write.size(); ++i)
        c.participant->writeData(c.mesh, c.write[i], c.ids,
            std::vector<double>(s.v.begin() + long(i)*s.n, s.v.begin() + long(i + 1)*s.n));
}
inline std::tuple<long, bool> couple_init(long tok, const Vertices& s)
{
    auto& c = coupling();
    if (c.participant->requiresInitialData()) couple_write_data(s);
    c.participant->initialize();
    return {tok + 1, c.participant->isCouplingOngoing()};
}
inline std::tuple<long, double> couple_max_dt(long tok)
{
    return {tok + 1, coupling().participant->getMaxTimeStepSize()};
}
inline std::tuple<long, Vertices> couple_read(long tok)
{
    auto& c = coupling();
    const int n = int(c.ids.size());
    Vertices peer{int(c.read.size()), n, std::vector<double>(c.read.size()*n)};
    std::vector<double> v(n);
    for (size_t i = 0; i < c.read.size(); ++i)
    {
        c.participant->readData(c.mesh, c.read[i], c.ids, 0.0, v);
        std::copy(v.begin(), v.end(), peer.v.begin() + long(i)*n);
    }
    return {tok + 1, peer};
}
inline std::tuple<long> couple_write(long tok, const Vertices& s)
{
    couple_write_data(s);
    return {tok + 1};
}
inline std::tuple<long, bool> couple_advance(long tok, double dt)
{
    auto& c = coupling();
    c.participant->advance(dt);
    return {tok + 1, c.participant->isCouplingOngoing()};
}
#endif

} // namespace zs
