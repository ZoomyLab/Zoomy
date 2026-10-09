#include <cstdio>
#include <cstdlib>
#include <string>
#include "rt.hpp"
namespace zs {
inline std::array<double,2> rusanov(const double* qL, const double* qR, double par) {
  double hL = qL[0], huL = qL[1], hR = qR[0], huR = qR[1], g = par;
  double x0 = 1.0/hL;
  double x1 = 1.0/hR;
  double x2 = (1.0/2.0)*std::max(std::sqrt(g*hL) + std::fabs(huL*x0), std::sqrt(g*hR) + std::fabs(huR*x1));
  double x3 = (1.0/4.0)*g;
  return {(1.0/2.0)*huL + (1.0/2.0)*huR - x2*(-hL + hR), std::pow(hL, 2)*x3 + std::pow(hR, 2)*x3 + (1.0/2.0)*std::pow(huL, 2)*x0 + (1.0/2.0)*std::pow(huR, 2)*x1 - x2*(-huL + huR)};
}
inline std::array<double,1> wavespeed(const double* q, double par) {
  double h = q[0], hu = q[1], g = par;
  return {std::sqrt(g*h) + std::fabs(hu/h)};
}
inline std::array<double,1> troubled(const double* q, double par) {
  double h = q[0], hu = q[1], hb = par;
  return {((h < hb) ? (
   1.0
)
: (
   0.0
))};
}
inline auto hyperbolic = [](auto Q, auto t, auto T, auto g, auto cf, auto nu, auto cfl, auto dx, auto hb, auto tok) {
    auto f1 = [&](auto t_2, auto Q_3, auto tok_4) {
        return t_2 < T;
    };
    auto f5 = [&](auto t_6, auto Q_7, auto tok_8) {
        auto x9 = cell_map(wavespeed, Q_7, g);
        auto x10 = reduce_max(x9);
        auto x11 = minimum(-t_6 + T, cfl*dx/x10);
        auto x12 = t_6 + x11;
        auto x13 = ghost(Q_7);
        auto x14 = reconstruct(x13, 2);
        auto x15 = face_map(rusanov, std::get<0>(x14), std::get<1>(x14), g);
        auto x16 = div(x15, dx);
        auto x17 = ghost(Q_7 - x16*x11);
        auto x18 = reconstruct(x17, 2);
        auto x19 = face_map(rusanov, std::get<0>(x18), std::get<1>(x18), g);
        auto x20 = div(x19, dx);
        auto x21 = Q_7 - 1.0/2.0*x16*x11 - 1.0/2.0*x20*x11;
        auto x22 = cell_map(troubled, x21, hb);
        auto x23 = reduce_any(x22);
        auto f24 = [&]() {
            auto x25 = reconstruct(x13, 1);
            auto x26 = face_map(rusanov, std::get<0>(x25), std::get<1>(x25), g);
            auto x27 = div(x26, dx);
            auto x28 = select(x22, Q_7 - x27*x11, x21);
            return std::make_tuple(x28, tok_8);
        };
        auto f29 = [&]() {
            return std::make_tuple(x21, tok_8);
        };
        auto x30 = cond(x23, f24, f29);
        auto x31 = comp(std::get<0>(x30), 0);
        auto f32 = [&](auto x_33) {
            auto x34 = lap(x_33, dx);
            return x_33 - nu*x34*x11;
        };
        auto x35 = comp(std::get<0>(x30), 1);
        auto f36 = [&](auto x_37) {
            return x_37*cf*abs(x_37)*x11*pow(x31, -2) + x_37 - x35;
        };
        auto x38 = root(f36, x35);
        auto x39 = linsolve(f32, x38);
        auto x40 = assemble(x31, x39);
        auto x41 = io_write(tok_8, x12, x40);
        return std::make_tuple(x12, x40, std::get<0>(x41));
    };
    auto x42 = while_loop(f1, f5, std::make_tuple(t, Q, tok));
    return std::make_tuple(std::get<0>(x42), std::get<1>(x42), std::get<2>(x42));
};
}
int main(int argc, char** v) {
 using namespace zs;
 int N = 200; double dx = 1.0 / N; Field Q(2, N);
 for (int c = 0; c < N; ++c) Q(0, c) = (c + 0.5) * dx < 0.5 ? 1.0 : 0.5;
 auto [tf, Qf, tk] = hyperbolic(Q, 0.0, 0.1, 9.81, 0.01, 0.001, 0.45, dx, std::atof(v[1]), 0L);
 std::printf("%ld %.17g\n", tk, tf);
 for (int c = 0; c < N; ++c) std::printf("%.17g %.17g\n", Qf(0, c), Qf(1, c));
}
