def hyperbolic(Q, t, T, g, cf, nu, cfl, dx, hb, tok):
    def f1(t_2, Q_3, tok_4):
        return (t_2 < T)
    def f5(t_6, Q_7, tok_8):
        x9 = cell_map(wavespeed, Q_7, g)
        x10 = reduce_max(x9)
        x11 = minimum(-t_6 + T, cfl*dx/x10)
        x12 = t_6 + x11
        x13 = ghost(Q_7)
        x14 = reconstruct(x13, 2)
        x15 = face_map(rusanov, x14[0], x14[1], g)
        x16 = div(x15, dx)
        x17 = ghost(Q_7 - x16*x11)
        x18 = reconstruct(x17, 2)
        x19 = face_map(rusanov, x18[0], x18[1], g)
        x20 = div(x19, dx)
        x21 = Q_7 - 1/2*x16*x11 - 1/2*x20*x11
        x22 = cell_map(troubled, x21, hb)
        x23 = reduce_any(x22)
        def f24():
            x25 = reconstruct(x13, 1)
            x26 = face_map(rusanov, x25[0], x25[1], g)
            x27 = div(x26, dx)
            x28 = select(x22, Q_7 - x27*x11, x21)
            return (x28, tok_8, )
        def f29():
            return (x21, tok_8, )
        x30 = cond(x23, f24, f29)
        x31 = comp(x30[0], 0)
        def f32(x_33):
            x34 = lap(x_33, dx)
            return x_33 - nu*x34*x11
        x35 = comp(x30[0], 1)
        def f36(x_37):
            return x_37*cf*abs(x_37)*x11*x31**(-2) + x_37 - x35
        x38 = root(f36, x35)
        x39 = linsolve(f32, x38)
        x40 = assemble(x31, x39)
        x41 = io_write(tok_8, x12, x40)
        return (x12, x40, x41[0], )
    x42 = while_loop(f1, f5, (t, Q, tok, ))
    return (x42[0], x42[1], x42[2], )
