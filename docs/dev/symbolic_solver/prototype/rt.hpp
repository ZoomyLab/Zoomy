// C++ runtime: what a backend implementer writes (blocks + mesh ops).
#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <tuple>
#include <vector>

namespace zs {
using std::abs; using std::pow; using std::sqrt;

struct Field {
    int nq = 0, n = 0;
    std::vector<double> d;
    Field() = default;
    Field(int nq_, int n_, double v = 0) : nq(nq_), n(n_), d(size_t(nq_) * n_, v) {}
    double& operator()(int i, int c) { return d[size_t(i) * n + c]; }
    double operator()(int i, int c) const { return d[size_t(i) * n + c]; }
};

template <class F> Field zip(const Field& a, const Field& b, F f) {
    assert(a.nq == b.nq && a.n == b.n);
    Field r(a.nq, a.n);
    for (size_t k = 0; k < r.d.size(); ++k) r.d[k] = f(a.d[k], b.d[k]);
    return r;
}
template <class F> Field map(const Field& a, F f) {
    Field r(a.nq, a.n);
    for (size_t k = 0; k < r.d.size(); ++k) r.d[k] = f(a.d[k]);
    return r;
}
inline Field operator+(const Field& a, const Field& b) { return zip(a, b, [](double x, double y) { return x + y; }); }
inline Field operator-(const Field& a, const Field& b) { return zip(a, b, [](double x, double y) { return x - y; }); }
inline Field operator*(const Field& a, const Field& b) { return zip(a, b, [](double x, double y) { return x * y; }); }
inline Field operator/(const Field& a, const Field& b) { return zip(a, b, [](double x, double y) { return x / y; }); }
inline Field operator*(const Field& a, double s) { return map(a, [s](double x) { return x * s; }); }
inline Field operator*(double s, const Field& a) { return a * s; }
inline Field operator/(const Field& a, double s) { return map(a, [s](double x) { return x / s; }); }
inline Field operator-(const Field& a) { return map(a, [](double x) { return -x; }); }
inline Field abs(const Field& a) { return map(a, [](double x) { return std::abs(x); }); }
inline Field pow(const Field& a, double e) { return map(a, [e](double x) { return std::pow(x, e); }); }
inline double minimum(double a, double b) { return std::min(a, b); }
inline double maxabs(const Field& a) { double m = 0; for (double x : a.d) m = std::max(m, std::abs(x)); return m; }

// ── mesh ops (1-D uniform, transmissive) ─────────────────────────────────
inline Field ghost(const Field& Q) {
    Field g(Q.nq, Q.n + 4);
    for (int i = 0; i < Q.nq; ++i)
        for (int c = 0; c < Q.n + 4; ++c) g(i, c) = Q(i, std::clamp(c - 2, 0, Q.n - 1));
    return g;
}
inline double minmod(double a, double b) { return a * b > 0 ? (a > 0 ? 1 : -1) * std::min(std::abs(a), std::abs(b)) : 0.0; }
inline std::tuple<Field, Field> reconstruct(const Field& g, int order) {
    int nf = g.n - 3;
    Field L(g.nq, nf), R(g.nq, nf);
    for (int i = 0; i < g.nq; ++i)
        for (int f = 0; f < nf; ++f) {
            double sl = 0, sr = 0;
            if (order == 2) {
                sl = minmod(g(i, f + 1) - g(i, f), g(i, f + 2) - g(i, f + 1));
                sr = minmod(g(i, f + 2) - g(i, f + 1), g(i, f + 3) - g(i, f + 2));
            }
            L(i, f) = g(i, f + 1) + 0.5 * sl;
            R(i, f) = g(i, f + 2) - 0.5 * sr;
        }
    return {L, R};
}
template <class K> Field face_map(K k, const Field& L, const Field& R, double par) {
    Field F(L.nq, L.n);
    for (int f = 0; f < L.n; ++f) {
        double ql[2] = {L(0, f), L(1, f)}, qr[2] = {R(0, f), R(1, f)};
        auto r = k(ql, qr, par);
        for (size_t i = 0; i < r.size(); ++i) F(i, f) = r[i];
    }
    return F;
}
template <class K> Field cell_map(K k, const Field& Q, double par) {
    Field F(1, Q.n);
    for (int c = 0; c < Q.n; ++c) { double q[2] = {Q(0, c), Q(1, c)}; F(0, c) = k(q, par)[0]; }
    return F;
}
inline Field div(const Field& F, double dx) {
    Field r(F.nq, F.n - 1);
    for (int i = 0; i < F.nq; ++i)
        for (int c = 0; c < r.n; ++c) r(i, c) = (F(i, c + 1) - F(i, c)) / dx;
    return r;
}
inline Field lap(const Field& x, double dx) {
    Field r(1, x.n);
    for (int c = 0; c < x.n; ++c)
        r(0, c) = (x(0, std::max(c - 1, 0)) - 2 * x(0, c) + x(0, std::min(c + 1, x.n - 1))) / (dx * dx);
    return r;
}
inline double reduce_max(const Field& a) { return *std::max_element(a.d.begin(), a.d.end()); }
inline bool reduce_any(const Field& a) { for (double x : a.d) if (x > 0.5) return true; return false; }
inline Field select(const Field& b, const Field& A, const Field& B) {
    Field r = B;
    for (int i = 0; i < A.nq; ++i) for (int c = 0; c < A.n; ++c) if (b(0, c) > 0.5) r(i, c) = A(i, c);
    return r;
}
inline Field comp(const Field& Q, int i) {
    Field r(1, Q.n);
    for (int c = 0; c < Q.n; ++c) r(0, c) = Q(i, c);
    return r;
}
inline Field assemble(const Field& a, const Field& b) {
    Field r(2, a.n);
    for (int c = 0; c < a.n; ++c) { r(0, c) = a(0, c); r(1, c) = b(0, c); }
    return r;
}

// ── control blocks ───────────────────────────────────────────────────────
template <class C, class B, class S> S while_loop(C c, B b, S s) {
    while (std::apply(c, s)) s = std::apply(b, s);
    return s;
}
template <class A, class B> auto cond(bool p, A a, B b) { return p ? a() : b(); }
template <class F> Field root(F f, Field x) {             // Newton, FD derivative
    for (int it = 0; it < 50; ++it) {
        Field r = f(x);
        if (maxabs(r) <= 1e-14 * (1 + maxabs(x))) break;
        Field h = map(x, [](double v) { return 1e-7 * (std::abs(v) + 1e-7); });
        x = x - r * h / (f(x + h) - r);
    }
    return x;
}
inline double dot(const Field& a, const Field& b) { double s = 0; for (size_t k = 0; k < a.d.size(); ++k) s += a.d[k] * b.d[k]; return s; }
template <class A> Field linsolve(A Aop, const Field& b) {   // CG
    Field x(b.nq, b.n), r = b, p = b;
    double rr = dot(r, r), bb = rr;
    for (int i = 0; i < 500 && rr > 1e-28 * bb; ++i) {
        Field Ap = Aop(p);
        double a = rr / dot(p, Ap);
        x = x + a * p; r = r - a * Ap;
        double rr2 = dot(r, r);
        p = r + (rr2 / rr) * p; rr = rr2;
    }
    return x;
}
inline std::tuple<long> io_write(long tok, double, const Field&) { return {tok + 1}; }
inline Field set_ghost(Field g, int side, const Field& q) {
    for (int i = 0; i < g.nq; ++i)
        for (int k = 0; k < 2; ++k) g(i, side == 1 ? g.n - 1 - k : k) = q(i, 0);
    return g;
}
inline Field edge(const Field& Q, int side) {
    Field r(Q.nq, 1);
    for (int i = 0; i < Q.nq; ++i) r(i, 0) = Q(i, side == 1 ? Q.n - 1 : 0);
    return r;
}
}  // namespace zs
