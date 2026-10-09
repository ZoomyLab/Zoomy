inline auto run = [](auto Q, auto Qaux, auto p, auto time, auto time_end, auto mesh, auto tok) {
    auto f1 = [&](auto time_2, auto Q_2, auto Qaux_2, auto n_write, auto tok_2) {
        return time_2 < time_end;
    };
    auto f2 = [&](auto time_3, auto Q_3, auto Qaux_3, auto n_write_2, auto tok_3) {
        auto lam = face_wavespeed(nsm.eigenvalues_cfl, Q_3, Qaux_3, p, mesh);
        auto x3 = face_inradius(mesh);
        auto dt = reduce_min(1.8*x3/lam);
        auto dt_2 = minimum(minimum(5.0, -time_3 + time_end), dt);
        auto time_4 = time_3 + dt_2;
        auto x4 = boundary_states<2>(nsm.boundary_conditions, Q_3, Qaux_3, p, time_3, mesh);
        auto Qb = std::get<0>(x4);
        auto Qauxb = std::get<1>(x4);
        auto x5 = reconstruct(Q_3, Qaux_3, Qb, Qauxb, p, 1, false, mesh);
        auto QL = std::get<0>(x5);
        auto QR = std::get<1>(x5);
        auto QR_2 = boundary_traces(nsm.boundary_conditions, Q_3, QL, QR, Qaux_3, p, time_3, mesh);
        auto x6 = face_aux(Qaux_3, mesh);
        auto AL = std::get<0>(x6);
        auto AR = std::get<1>(x6);
        auto F = faces(nsm.numerical_flux, QL, QR_2, AL, AR, p, mesh);
        auto x7 = faces<2>(nsm.numerical_fluctuations, QL, QR_2, AL, AR, p, mesh);
        auto Dp = std::get<0>(x7);
        auto Dm = std::get<1>(x7);
        auto x8 = divergence(F, Dp, Dm, mesh);
        auto x9 = cells(nsm.source, Q_3, Qaux_3, p, time_3, dt_2, mesh);
        auto Q_4 = Q_3 + dt_2*(x9 - x8);
        auto Qaux_4 = cells(nsm.update_aux_variables, Q_4, Qaux_3, p, time_4, dt_2, mesh);
        auto Qaux_5 = derivative_aux(nsm.aux_derivatives, Q_4, Qaux_4, mesh);
        return std::make_tuple(time_4, Q_4, Qaux_5, n_write_2, tok_3);
    };
    auto x10 = while_loop(f1, f2, std::make_tuple(time, Q, Qaux, 1, tok));
    auto time_5 = std::get<0>(x10);
    auto Q_5 = std::get<1>(x10);
    auto Qaux_6 = std::get<2>(x10);
    return std::make_tuple(time_5, Q_5, Qaux_6, std::get<4>(x10));
};

inline auto march = [](auto Q, auto Qaux, auto p, auto time_end, auto mesh) {
    return run(Q, Qaux, p, 0.0, time_end, mesh, 0L);
};

inline auto initial_aux = [](auto Q, auto Qaux, auto p, auto mesh, auto tok) {
    auto Qaux_2 = cells(nsm.update_aux_variables, Q, Qaux, p, 0, 1, mesh);
    auto Qaux_3 = derivative_aux(nsm.aux_derivatives, Q, Qaux_2, mesh);
    return std::make_tuple(Qaux_3, tok);
};

inline void configure()
{
    settings().limiter = "minmod";
    settings().lsq_degree = 1;
    implicit().tol = 1e-08;
    implicit().maxiter = 6;
    implicit().ls_tol = 1e-07;
    implicit().ls_restart = 20;
    implicit().ls_maxiter = 30;
    implicit().ls_accept = -1.0;
}
