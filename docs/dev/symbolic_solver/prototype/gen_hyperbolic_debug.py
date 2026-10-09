def hyperbolic(Q, t, T, g, cf, nu, cfl, dx, hb, tok):
    def f1(t_2, Q_2, tok_2):
        return (t_2 < T)
    def f2(t_3, Q_3, tok_3):
        x3 = cell_map(wavespeed, Q_3, g)
        x4 = reduce_max(x3)
        dt = minimum(-t_3 + T, cfl*dx/x4)
        x5 = t_3 + dt
        Qg = ghost(Q_3)
        x6 = reconstruct(Qg, 2)
        QL = x6[0]
        QR = x6[1]
        x7 = face_map(rusanov, QL, QR, g)
        x8 = div(x7, dx)
        Q_4 = Q_3 - x8*dt
        Qg_2 = ghost(Q_4)
        x9 = reconstruct(Qg_2, 2)
        QL_2 = x9[0]
        QR_2 = x9[1]
        x10 = face_map(rusanov, QL_2, QR_2, g)
        x11 = div(x10, dx)
        Q_5 = Q_3 - 1/2*x8*dt - 1/2*x11*dt
        bad = cell_map(troubled, Q_5, hb)
        x12 = reduce_any(bad)
        def f13():
            x14 = reconstruct(Qg, 1)
            QL_3 = x14[0]
            QR_3 = x14[1]
            x15 = face_map(rusanov, QL_3, QR_3, g)
            x16 = div(x15, dx)
            x17 = select(bad, Q_3 - x16*dt, Q_5)
            return (x17, tok_3, )
        def f18():
            return (Q_5, tok_3, )
        x19 = cond(x12, f13, f18)
        Q_6 = x19[0]
        h = comp(Q_6, 0)
        def f20(x):
            x21 = lap(x, dx)
            return x - nu*x21*dt
        hu = comp(Q_6, 1)
        def f22(x_2):
            return x_2*cf*abs(x_2)*dt*h**(-2) + x_2 - hu
        hu_2 = root(f22, hu)
        hu_3 = linsolve(f20, hu_2)
        Q_7 = assemble(h, hu_3)
        x23 = io_write(tok_3, x5, Q_7)
        return (x5, Q_7, x23[0], )
    x24 = while_loop(f1, f2, (t, Q, tok, ))
    return (x24[0], x24[1], x24[2], )
