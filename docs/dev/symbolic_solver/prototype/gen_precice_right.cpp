#include <cstdio>
#include <cstdlib>
#include <string>
#include "rt_precice.hpp"
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
inline auto hyperbolic_precice = [](auto Q, auto t, auto g, auto cf, auto nu, auto cfl, auto dx, auto hb, auto tok) {
    auto f1 = [&](auto t_2, auto Q_3, auto go_4, auto tok_5) {
        return go_4;
    };
    auto f6 = [&](auto t_7, auto Q_8, auto go_9, auto tok_10) {
        auto x11 = couple_max_dt(tok_10);
        auto x12 = minimum(0.001, std::get<1>(x11));
        auto x13 = t_7 + x12;
        auto x14 = ghost(Q_8);
        auto x15 = couple_read(std::get<0>(x11), x12);
        auto x16 = set_ghost(x14, 0, std::get<1>(x15));
        auto x17 = reconstruct(x16, 1);
        auto x18 = face_map(rusanov, std::get<0>(x17), std::get<1>(x17), g);
        auto x19 = div(x18, dx);
        auto x20 = Q_8 - x19*x12;
        auto x21 = comp(x20, 0);
        auto f22 = [&](auto x_23) {
            auto x24 = lap(x_23, dx);
            return x_23 - nu*x24*x12;
        };
        auto x25 = comp(x20, 1);
        auto f26 = [&](auto x_27) {
            return x_27*cf*abs(x_27)*x12/pow(x21, 2) + x_27 - x25;
        };
        auto x28 = root(f26, x25);
        auto x29 = linsolve(f22, x28);
        auto x30 = assemble(x21, x29);
        auto x31 = edge(x30, 0);
        auto x32 = couple_write(std::get<0>(x15), x31);
        auto x33 = couple_advance(std::get<0>(x32), x12);
        auto x34 = io_write(std::get<0>(x33), x13, x30);
        return std::make_tuple(x13, x30, std::get<1>(x33), std::get<0>(x34));
    };
    auto x35 = edge(Q, 0);
    auto x36 = couple_init(tok, x35);
    auto x37 = while_loop(f1, f6, std::make_tuple(t, Q, std::get<1>(x36), std::get<0>(x36)));
    return std::make_tuple(std::get<0>(x37), std::get<1>(x37), std::get<3>(x37));
};
}
int main(int argc, char** v) {
 using namespace zs;
 std::string name = v[1]; std::string me = name == "Left" ? "L" : "R", pe = name == "Left" ? "R" : "L";
 cpl_setup(name, "precice-config.xml", name + "-Mesh", {me + "0", me + "1"}, {pe + "0", pe + "1"}, 0.5);
 int N = 100; Field Q(2, N); for (int c = 0; c < N; ++c) Q(0, c) = std::atof(v[2]);
 auto [tf, Qf, tk] = hyperbolic_precice(Q, 0.0, 9.81, 0.01, 0.0, 0.45, 1.0 / 200, 0.0, 0L);
 cpl().p->finalize();
 FILE* f = std::fopen(v[3], "w");
 for (int c = 0; c < N; ++c) std::fprintf(f, "%.17g %.17g\n", Qf(0, c), Qf(1, c));
 std::fclose(f); std::printf("%s: t=%.15g effects=%ld\n", v[1], tf, tk);
}
