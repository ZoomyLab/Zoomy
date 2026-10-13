#pragma once
/// \file rt_common.H
/// Backend-neutral part of the C++ runtimes of the symbolic solvers (foam rt.H, AMReX
/// rt_amrex.H, both include it first): functors, settings, effects, Newton skeletons
/// and the preCICE call sequence.
/// The runtime supplies in namespace zs `Field` (rows `nq`), `warn`, and for the Newton
/// skeletons `norm`, `perturb`, `newton_update` and `linsolve`; `ZS_HD` marks code a
/// device loop calls (AMReX: AMREX_GPU_HOST_DEVICE).
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
/// Quiet NaN.
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
/// sqrt(DBL_EPSILON), the finite-difference scale.
constexpr double SQRT_EPS = 1.4901161193847656e-08;    // sqrt(DBL_EPSILON) = 2^-26

// ── elementwise arithmetic ──────────────────────────────────────────────────
/// Elementwise x + y.
struct Add { ZS_HD double operator()(double x, double y) const { return x + y; } };
/// Elementwise x - y.
struct Sub { ZS_HD double operator()(double x, double y) const { return x - y; } };
/// Elementwise x * y.
struct Mul { ZS_HD double operator()(double x, double y) const { return x*y; } };
/// Elementwise x / y.
struct Div { ZS_HD double operator()(double x, double y) const { return x/y; } };
/// Elementwise x^y.
struct Pow { ZS_HD double operator()(double x, double y) const { return std::pow(x, y); } };
/// Elementwise |x| (second argument ignored).
struct Abs { ZS_HD double operator()(double x, double) const { return std::abs(x); } };
/// Elementwise minimum, NaN propagating (std::min would swallow a NaN).
struct Min
{
    ZS_HD double operator()(double x, double y) const
    { return (x != x || y != y) ? NaN : (x < y ? x : y); }
};
/// Elementwise maximum, NaN propagating.
struct Max
{
    ZS_HD double operator()(double x, double y) const
    { return (x != x || y != y) ? NaN : (x > y ? x : y); }
};
/// Host-callable NaN-propagating minimum, the name the printed code uses.
inline double minimum(double a, double b) { return Min{}(a, b); }
/// Host-callable NaN-propagating maximum, the name the printed code uses.
inline double maximum(double a, double b) { return Max{}(a, b); }

// ── settings (the printed configure()) ──────────────────────────────────────
/// What the printed `configure()` sets: the LSQ degree and the two-point flux floor.
struct Settings
{
    int lsq_degree = 1;            // highest derivative order the model asks for
    double tpfa_cos_floor = NaN;   // two-point fluxes: |n.e| raised to this
};
/// The run's Settings.
inline Settings& settings() { static Settings s; return s; }

/// Tolerances of the implicit solves: Newton (tol, maxiter) and GMRES (tol, restart,
/// maxiter, accept gate).
struct Implicit
{
    double tol = 1e-8;        // Newton: ||R||_2 <= tol
    int maxiter = 6;
    double ls_tol = 1e-7;     // GMRES: ||A x - b|| <= ls_tol ||b||
    int ls_restart = 20, ls_maxiter = 30;
    double ls_accept = -1;    // > 0: abort above this relative residual
};
/// The run's Implicit settings.
inline Implicit& implicit() { static Implicit s; return s; }

/// The runtime's warning (rank 0), defined by the runtime.
inline void warn(const std::string& message);

/// Concatenate the arguments into a string.
template<class... A> std::string text(const A&... a)
{
    std::ostringstream s;
    (s << ... << a);
    return s.str();
}

// ── effects ─────────────────────────────────────────────────────────────────
/// Runtime field type (rows `nq`), defined by the runtime.
struct Field;
/// The output callback set by the driver.
inline std::function<void(double, const Field&, const Field&)>& writer()
{
    static std::function<void(double, const Field&, const Field&)> w;
    return w;
}

/// Effect `io.write`: call the output callback; returns the order token.
inline std::tuple<long> io_write(long tok, double time, const Field& Q, const Field& Qaux)
{
    if (writer()) writer()(time, Q, Qaux);
    return {tok + 1};
}

/// The restart writer set by the driver: save the march state (time, Q, Qaux).
inline std::function<void(double, const Field&, const Field&)>& saver()
{
    static std::function<void(double, const Field&, const Field&)> s;
    return s;
}
/// The restart reader set by the driver: overwrite (time, Q, Qaux) with the saved state.
inline std::function<void(double&, Field&, Field&)>& loader()
{
    static std::function<void(double&, Field&, Field&)> l;
    return l;
}

/// Effect `io.save_state`: call the restart writer; returns the order token.
inline std::tuple<long> io_save_state(long tok, double time, const Field& Q, const Field& Qaux)
{
    if (saver()) saver()(time, Q, Qaux);
    return {tok + 1};
}

/// Effect `io.load_state`: the saved march state when restarting, else the arguments.
template<class F>
std::tuple<long, double, F, F> io_load_state(long tok, double time, F Q, F Qaux)
{
    if (loader()) loader()(time, Q, Qaux);
    return {tok + 1, time, Q, Qaux};
}

/// Effect `check.cfl`: warn once if `dt` exceeds the CFL step `dt_cfl`.
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
/// Solve A x = b for n x n row-major A by elimination with partial pivoting; A is
/// overwritten and x returned in b.
/// Ref: Golub, Van Loan 2013, https://doi.org/10.56021/9781421407944
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

/// One-sided finite-difference step of a variable of value v.
ZS_HD inline double fd_step(double v)
{
    const double a = std::abs(v);
    return SQRT_EPS*(a > 1.0 ? a : 1.0);
}

/// Warn once that a Newton iteration did not converge.
inline void warn_unconverged(bool& warned, const char* what, int it, double r)
{
    if (warned) return;
    warned = true;
    warn(text(what, ": not converged after ", it, " Newton updates, ||R|| = ", r,
              " (warned once)"));
}

/// Newton on a cell-local residual f (x -> R, same shape): each cell's Jacobian by
/// one-sided differences, its update by solve_small; tol / maxiter from `implicit()`.
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

/// Jacobian-free Newton-Krylov on a global residual R (x -> R, same shape): each update
/// a linsolve on the Jacobian action by a directional difference.
/// Ref: Knoll, Keyes 2004, https://doi.org/10.1016/j.jcp.2003.08.010
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
/// Interface values on the host, rows x vertices (`v[q n + j]`).
struct Vertices { int nq = 0, n = 0; std::vector<double> v; };   // v[q n + j]
/// The preCICE participant, mesh, data names and vertex ids; the app fills `coupling()`.
/// Ref: Chourdakis et al. 2022, https://doi.org/10.12688/openreseurope.14445.2
struct Coupling
{
    std::unique_ptr<precice::Participant> participant;
    std::string mesh;
    std::vector<std::string> write, read;
    std::vector<precice::VertexID> ids;
};
/// The run's Coupling.
inline Coupling& coupling() { static Coupling c; return c; }

/// Write the interface values `s` to preCICE, one data field per row.
inline void couple_write_data(const Vertices& s)
{
    auto& c = coupling();
    for (size_t i = 0; i < c.write.size(); ++i)
        c.participant->writeData(c.mesh, c.write[i], c.ids,
            std::vector<double>(s.v.begin() + long(i)*s.n, s.v.begin() + long(i + 1)*s.n));
}
/// Effect `couple.init`: write initial data if required, initialize; returns whether
/// coupling is ongoing.
inline std::tuple<long, bool> couple_init(long tok, const Vertices& s)
{
    auto& c = coupling();
    if (c.participant->requiresInitialData()) couple_write_data(s);
    c.participant->initialize();
    return {tok + 1, c.participant->isCouplingOngoing()};
}
/// Effect `couple.max_dt`: the time left in the coupling window.
inline std::tuple<long, double> couple_max_dt(long tok)
{
    return {tok + 1, coupling().participant->getMaxTimeStepSize()};
}
/// Effect `couple.read`: the peer's interface values.
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
/// Effect `couple.write`: send the interface values.
inline std::tuple<long> couple_write(long tok, const Vertices& s)
{
    couple_write_data(s);
    return {tok + 1};
}
/// Effect `couple.advance`: advance preCICE by `dt`; returns whether coupling is ongoing.
inline std::tuple<long, bool> couple_advance(long tok, double dt)
{
    auto& c = coupling();
    c.participant->advance(dt);
    return {tok + 1, c.participant->isCouplingOngoing()};
}
#endif

} // namespace zs
