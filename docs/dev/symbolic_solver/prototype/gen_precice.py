def hyperbolic_precice(Q, t, g, cf, nu, cfl, dx, hb, tok):
    def f1(t_2, Q_3, go_4, tok_5):
        return go_4
    def f6(t_7, Q_8, go_9, tok_10):
        x11 = couple_max_dt(tok_10)
        x12 = minimum(0.001, x11[1])
        x13 = t_7 + x12
        x14 = ghost(Q_8)
        x15 = couple_read(x11[0], x12)
        x16 = set_ghost(x14, 1, x15[1])
        x17 = reconstruct(x16, 1)
        x18 = face_map(rusanov, x17[0], x17[1], g)
        x19 = div(x18, dx)
        x20 = Q_8 - x19*x12
        x21 = comp(x20, 0)
        def f22(x_23):
            x24 = lap(x_23, dx)
            return x_23 - nu*x24*x12
        x25 = comp(x20, 1)
        def f26(x_27):
            return x_27*cf*abs(x_27)*x12/x21**2 + x_27 - x25
        x28 = root(f26, x25)
        x29 = linsolve(f22, x28)
        x30 = assemble(x21, x29)
        x31 = edge(x30, 1)
        x32 = couple_write(x15[0], x31)
        x33 = couple_advance(x32[0], x12)
        x34 = io_write(x33[0], x13, x30)
        return (x13, x30, x33[1], x34[0], )
    x35 = edge(Q, 1)
    x36 = couple_init(tok, x35)
    x37 = while_loop(f1, f6, (t, Q, x36[1], x36[0], ))
    return (x37[0], x37[1], x37[3], )
