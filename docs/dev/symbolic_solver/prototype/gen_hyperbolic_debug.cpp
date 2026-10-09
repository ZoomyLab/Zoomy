inline auto hyperbolic = [](auto Q, auto t, auto T, auto g, auto cf, auto nu, auto cfl, auto dx, auto hb, auto tok) {
    auto f1 = [&](auto t_2, auto Q_2, auto tok_2) {
        return t_2 < T;
    };
    auto f2 = [&](auto t_3, auto Q_3, auto tok_3) {
        auto x3 = cell_map(wavespeed, Q_3, g);
        auto x4 = reduce_max(x3);
        auto dt = minimum(-t_3 + T, cfl*dx/x4);
        auto x5 = t_3 + dt;
        auto Qg = ghost(Q_3);
        auto x6 = reconstruct(Qg, 2);
        auto QL = std::get<0>(x6);
        auto QR = std::get<1>(x6);
        auto x7 = face_map(rusanov, QL, QR, g);
        auto x8 = div(x7, dx);
        auto Q_4 = Q_3 - x8*dt;
        auto Qg_2 = ghost(Q_4);
        auto x9 = reconstruct(Qg_2, 2);
        auto QL_2 = std::get<0>(x9);
        auto QR_2 = std::get<1>(x9);
        auto x10 = face_map(rusanov, QL_2, QR_2, g);
        auto x11 = div(x10, dx);
        auto Q_5 = Q_3 - 1.0/2.0*x8*dt - 1.0/2.0*x11*dt;
        auto bad = cell_map(troubled, Q_5, hb);
        auto x12 = reduce_any(bad);
        auto f13 = [&]() {
            auto x14 = reconstruct(Qg, 1);
            auto QL_3 = std::get<0>(x14);
            auto QR_3 = std::get<1>(x14);
            auto x15 = face_map(rusanov, QL_3, QR_3, g);
            auto x16 = div(x15, dx);
            auto x17 = select(bad, Q_3 - x16*dt, Q_5);
            return std::make_tuple(x17, tok_3);
        };
        auto f18 = [&]() {
            return std::make_tuple(Q_5, tok_3);
        };
        auto x19 = cond(x12, f13, f18);
        auto Q_6 = std::get<0>(x19);
        auto h = comp(Q_6, 0);
        auto f20 = [&](auto x) {
            auto x21 = lap(x, dx);
            return x - nu*x21*dt;
        };
        auto hu = comp(Q_6, 1);
        auto f22 = [&](auto x_2) {
            return x_2*cf*abs(x_2)*dt*pow(h, -2) + x_2 - hu;
        };
        auto hu_2 = root(f22, hu);
        auto hu_3 = linsolve(f20, hu_2);
        auto Q_7 = assemble(h, hu_3);
        auto x23 = io_write(tok_3, x5, Q_7);
        return std::make_tuple(x5, Q_7, std::get<0>(x23));
    };
    auto x24 = while_loop(f1, f2, std::make_tuple(t, Q, tok));
    return std::make_tuple(std::get<0>(x24), std::get<1>(x24), std::get<2>(x24));
};
