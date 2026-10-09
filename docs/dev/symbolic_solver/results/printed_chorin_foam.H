inline auto run = [](auto Q, auto Qaux, auto p, auto time, auto time_end, auto mesh, auto tok) {
    auto f1 = [&](auto time_2, auto Q_2, auto Qaux_2, auto n_write, auto tok_2) {
        return time_2 < time_end;
    };
    auto f2 = [&](auto time_3, auto Q_3, auto Qaux_3, auto n_write_2, auto tok_3) {
        auto dt = minimum(0.0078125, -time_3 + time_end);
        auto time_4 = time_3 + dt;
        auto Qaux_4 = cells(pred.update_aux_variables, Q_3, Qaux_3, p, time_3, dt, mesh);
        auto Qaux_5 = derivative_aux(pred.aux_derivatives, Q_3, Qaux_4, mesh);
        auto x3 = boundary_states<2>(pred.boundary_conditions, Q_3, Qaux_5, p, time_3, mesh);
        auto Qb = std::get<0>(x3);
        auto Qauxb = std::get<1>(x3);
        auto x4 = reconstruct(Q_3, Qaux_5, Qb, Qauxb, p, 1, false, mesh);
        auto QL = std::get<0>(x4);
        auto QR = std::get<1>(x4);
        auto QR_2 = boundary_traces(pred.boundary_conditions, Q_3, QL, QR, Qaux_5, p, time_3, mesh);
        auto x5 = face_aux(Qaux_5, mesh);
        auto AL = std::get<0>(x5);
        auto AR = std::get<1>(x5);
        auto F = faces(pred.numerical_flux, QL, QR_2, AL, AR, p, mesh);
        auto x6 = faces<2>(pred.numerical_fluctuations, QL, QR_2, AL, AR, p, mesh);
        auto Dp = std::get<0>(x6);
        auto Dm = std::get<1>(x6);
        auto x7 = divergence(F, Dp, Dm, mesh);
        auto x8 = -dt*x7;
        auto x9 = cells(pred.source, Q_3 + x8, Qaux_5, p, time_3, dt, mesh);
        auto Q_4 = Q_3 + dt*x9 + x8;
        auto Y = put(Q_4, std::make_tuple(6, 7), 0);
        auto Qaux_6 = zeros(11, mesh);
        auto Qaux_7 = derivative_aux(press.aux_derivatives, Y, Qaux_6, mesh);
        auto b = cells(press.source, Y, Qaux_7, p, time_3, dt, mesh);
        auto x10 = -b;
        auto f11 = [&](auto P) {
            auto Y_2 = put(Q_4, std::make_tuple(6, 7), P);
            auto Qaux_8 = derivative_aux(press.aux_derivatives, Y_2, Qaux_6, mesh);
            auto x12 = cells(press.source, Y_2, Qaux_8, p, time_3, dt, mesh);
            return x10 + x12;
        };
        auto x13 = linsolve(f11, x10);
        auto Q_5 = put(Q_4, std::make_tuple(6, 7), x13);
        auto Qaux_9 = zeros(6, mesh);
        auto Qaux_10 = derivative_aux(corr.aux_derivatives, Q_5, Qaux_9, mesh);
        auto x14 = cells(corr.update_variables, Q_5, Qaux_10, p, time_3, dt, mesh);
        auto Q_6 = put(Q_5, std::make_tuple(2, 3, 4, 5), x14);
        auto Qaux_11 = cells(pred.update_aux_variables, Q_6, Qaux_3, p, time_4, dt, mesh);
        auto Qaux_12 = derivative_aux(pred.aux_derivatives, Q_6, Qaux_11, mesh);
        auto lam = face_wavespeed(pred.eigenvalues_cfl, Q_3, Qaux_3, p, mesh);
        auto x15 = face_inradius(mesh);
        auto dt_2 = reduce_min(1.8*x15/lam);
        auto x16 = check_cfl(tok_3, minimum(5.0, dt_2), 0.0078125);
        return std::make_tuple(time_4, Q_6, Qaux_12, n_write_2, std::get<0>(x16));
    };
    auto x17 = while_loop(f1, f2, std::make_tuple(time, Q, Qaux, 1, tok));
    auto time_5 = std::get<0>(x17);
    auto Q_7 = std::get<1>(x17);
    auto Qaux_13 = std::get<2>(x17);
    return std::make_tuple(time_5, Q_7, Qaux_13, std::get<4>(x17));
};

inline auto march = [](auto Q, auto Qaux, auto p, auto time_end, auto mesh) {
    return run(Q, Qaux, p, 0.0, time_end, mesh, 0L);
};

inline auto initial_aux = [](auto Q, auto Qaux, auto p, auto mesh, auto tok) {
    auto Qaux_2 = cells(pred.update_aux_variables, Q, Qaux, p, 0, 1, mesh);
    auto Qaux_3 = derivative_aux(pred.aux_derivatives, Q, Qaux_2, mesh);
    return std::make_tuple(Qaux_3, tok);
};

inline void configure()
{
    settings().limiter = "venkatakrishnan";
    settings().lsq_degree = 2;
    implicit().tol = 1e-08;
    implicit().maxiter = 6;
    implicit().ls_tol = 1e-10;
    implicit().ls_restart = 120;
    implicit().ls_maxiter = 100;
    implicit().ls_accept = 1e-06;
}
