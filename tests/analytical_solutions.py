import matplotlib.pyplot as plt
import numpy as np
from sympy import *

z = Symbol("z", real=True)
basis = [1, 1 - 2 * z, 1 - 6 * z + 6 * z**2]


def GP(f, k):
    assert k < len(basis)
    return integrate(f * basis[k], (z, 0, 1))


def solution_finder():
    x = Symbol("x", real=True)
    t = Symbol("t", real=True)
    z = Symbol("z", real=True)
    v = Function("v")(z)
    force = Symbol("force")
    a1, a2, a3, a4, a5, a6, a7 = symbols("a1, a2, a3, a4, a5, a6 ,a7", real=True)
    g, ex, ez, rho, nu = symbols("g ex ez rho nu", real=True, constant=True)
    # TC 2
    # h = sin(x - t )
    # b = -sin(x - t)
    # u_mean = 1.
    # u_deviation = a1 * ln(z)
    # # u_deviation = 0
    # # u_deviation = a1*z + a2*z**2 + a3*z**3
    # u = u_mean + u_deviation - integrate(u_deviation, (z, 0, 1))

    # TC wired
    h = sin(x) + 2
    b = 0
    u_mean = 1.0
    u_deviation = a1 * ln(z)
    # u_deviation = 0
    # u_deviation = a1*z + a2*z**2 + a3*z**3
    u = u_mean + u_deviation - integrate(u_deviation, (z, 0, 1))

    # TC3
    # h = a1
    # b = 0
    # u_mean = 0.
    # u_deviation = a2*(1-2*z) + a3*(1-6*z+6*z**2)
    # # u_deviation = 0
    # # u_deviation = a1*z + a2*z**2 + a3*z**3
    # u = u_mean + u_deviation - integrate(u_deviation, (z, 0, 1))
    # TEST
    # h = Function('h')(t, x)
    # b = Function('b')(t, x)

    mass = diff(h, t) + diff(h * u_mean, x)
    print("residual mass balance", mass)

    phi_bar = lambda f: integrate(integrate(f, (z, 0, 1)) - f, (z, 0, z))
    w = 1 / h * phi_bar(diff(h * u, x))

    print("h \t \t \t", h)
    print("u \t \t \t", u)
    print("w \t \t \t", w)
    print("d/dt hu \t \t", simplify(diff(h * u, t)))
    print("d/dx hu^2 \t \t", simplify(diff(h * u * u, x)))
    print("huw \t \t \t", simplify(h * u * w))
    print("d/dz huw \t \t", simplify(diff(h * u * w, z)))
    # eq = diff(h*u, t) + diff(h*u**2, x)  + diff(h*u*w, z)
    # sol = solve(eq, a1, dict=True)
    # for s in sol:
    #     print(s)
    # return

    # sigma = rho * nu * (z * (ln(z) - 1)) * diff(h, t)
    # sigma = -rho * nu * (z * (ln(z) - 1)) * diff(h, x)
    # sigma = 0
    sigma = a7 * diff(u, z)

    momentum = (
        diff(h * u, t)
        + diff(h * u**2 + g / 2 * h**2, x)
        + diff(h * u * w - 1 / rho * sigma, z)
        - g * h * ex
        + g * h * ez * diff(b, x)
    )
    momentum = momentum.subs({ez: 1, ex: force, nu: 1})
    print("residual momentum", simplify(momentum))

    if simplify(momentum) != 0:
        print("")
        print("Try to solve for the momentum")
        # u = u + v
        # momentum = diff(h*u, t) + diff(h*u**2 + g/2*h**2, x) + diff(h*u*w -1/rho * sigma, z) - g*h*ex  + g*h*ez*diff(b, x)
        # momentum = simplify(momentum.subs({ez: 1, ex: 0, nu:1}))
        sol = solve(momentum, (a1, a2, a3, a4, a5, a6, a7, force), dict=True)
        # sol = solve(momentum, v, dict=True)
        # sol = solve(momentum, (h,b, v), dict=True)
        print("momentum balance")
        print(simplify(momentum))
        if sol is not None:
            print("number of solutions ", len(sol))
            for s in sol:
                print(simplify(s))
            # print(simplify(sol[0]))
            # print(simplify(sol[0].subs(v, 1)))
        else:
            print("no solution found")
    else:
        print("")
        print("PROJECTIONS")
        for i in range(len(basis)):
            print(f"momentum_{i}", GP(momentum, i))
        for i in range(len(basis)):
            print(f"sigma_{i}", GP(sigma, i))
        for i in range(len(basis)):
            print(f"u{i}", GP(u, i))

    plot = False
    if plot:
        X = np.linspace(0, 2 * np.pi, 100)
        t0 = 0.0
        t1 = 1.0
        print(u)
        l_h = lambda time, place: h.subs(t, time).subs(x, place)
        l_u_mean = lambda time, place: u_mean.subs(t, time).subs(x, place)
        l_b = lambda time, place: b.subs(t, time).subs(x, place)
        fig, ax = plt.subplots(2)
        H = [l_h(t0, x) for x in X]
        U_mean = [l_u_mean(t0, x) for x in X]
        B = [l_b(t0, x) for x in X]
        ax[0].plot(X, H, label="h")
        ax[0].plot(X, U_mean, label="u")
        ax[0].plot(X, B, label="b")

        Z = np.linspace(0.00001, 1, 100)
        l_u_dev = lambda place: u_deviation.subs(z, place)
        U_dev = [l_u_dev(z) for z in Z]
        ax[1].plot(U_dev, Z)
        plt.legend()
        plt.show()


def solution_ref_eq():
    x = Symbol("x", real=True)
    t = Symbol("t", real=True)
    z = Symbol("z", real=True)
    v = Function("v")(z)
    g, ex, ez, rho, nu = symbols("g ex ez rho nu", real=True, constant=True)
    h = sin(x - t)
    b = -sin(x - t)
    # define u such that it has u_m ==1 and is ln(z) + C
    # u_func = ln(z)
    u_func = v
    u_scale = integrate(u_func, (z, 0, 1))
    u = (u_func - u_scale) + 1.0
    u_m = integrate(u, (z, 0, 1))

    mass = diff(h, t) + diff(h * u_m, x)
    # print('residual mass balance', mass)

    phi_bar = lambda f: integrate(integrate(f, (z, 0, 1)) - f, (z, 0, z))
    w = 1 / h * phi_bar(diff(h * u, x))

    # sigma = rho * nu * (z * (ln(z) - 1)) * diff(h, t)
    # sigma = -rho * nu * (z * (ln(z) - 1)) * diff(h, x)
    sigma = 0

    # print(simplify(diff(h*u, t)))
    # print(simplify(diff(h*u*u, x)))
    # print(simplify(diff(h*u*w, z)))

    eq = (
        (simplify(diff(h * u, t)))
        + b * (simplify(diff(h * u * u, x)))
        + c * (simplify(diff(h * u * w, z)))
    )
    # print(solve(eq, (a, b, c)))

    momentum = (
        diff(h * u, t)
        + diff(h * u**2 + g / 2 * h**2, x)
        + diff(h * u * w - 1 / rho * sigma, z)
        - g * h * ex
        + g * h * ez * diff(b, x)
    )
    momentum = momentum.subs({ez: 1, ex: 0, nu: 1})
    # print('residual momentum', simplify(momentum))
    sol = solve(momentum, v)
    print("number of solutions ", len(sol))
    print(sol)
    print(simplify(sol[0]))
    print(simplify(sol[0].subs(v, 1)))

    # levels = 2
    # for i in range(levels+1):
    #     print(f'sigma_{i}', GP(sigma, i))
    # for i in range(levels+1):
    #     print(f'u{i}', GP(u, i))

    # Z = np.linspace(0.00001,1,100)
    # Y = Z * (np.log(Z)-1)
    # U = lambdify(z, u)
    # plt.plot(Y, Z)
    # plt.plot(U(Z), Z)
    # plt.show()


# not working
def sin_problem():
    x = Symbol("x", real=True)
    t = Symbol("t", real=True)
    a, b, c = symbols("a b c", real=True)
    al, be, ga = symbols("al be ga", real=True)
    eq = (
        cos(a * x + b * t + c) * b
        + cos(a * x + b * t + c) * a * sin(al * x + be * t + ga)
        + sin(a * x + b * t + c) * al * cos(al * x + be * t + ga)
    )
    print(eq)
    sol = solve(eq, al)
    print(sol)
    # sol = solve(eq, (a, b, c, al, be, ga))
    # print(sol)


def solve_constains_no_soace():
    z = Symbol("z", real=True)
    h = Symbol("z", real=True, constant=True)
    a, b, c = symbols("a b c", real=True)
    nu, rho, C, g, ex, sliplength = symbols(
        "nu rho C g ex sliplength", real=True, constant=True
    )
    u_ansatz = a + b * z + c * z**2

    # MATERIALS
    newtonian = -nu * diff(u_ansatz, z)

    # BOTTOM FRICTION

    # not working
    chezy = -C * u_ansatz * Abs(u_ansatz)

    chezy_monotone = -C * u_ansatz**2
    slip = -nu / sliplength * u_ansatz

    # CHOOSE OPTIONS
    stress = newtonian
    boundary_stress = slip

    momentum = diff(stress, z)

    eq_top = stress.subs(z, 1)
    eq_bottom = stress.subs(z, 0) - boundary_stress.subs(z, 0)
    eq_bulk = momentum - g * h * ex

    system = [eq_top, eq_bottom, eq_bulk]
    sol = solve(system, (a, b, c), dict=True)
    n_solutions = len(sol)
    assert n_solutions > 0
    sol_0 = sol[0]
    sol_substituted = {
        key: value.subs({nu: 1, sliplength: 1, C: 1, g: 1, ex: 1, h: 1})
        for key, value in sol_0.items()
    }
    # print(sol_substituted)
    return sol_substituted


def solution_slip():
    Z = np.linspace(0, 1, 100)

    h = 1
    g = 1
    ex = 1
    lam = 1
    u_sol = lambda z: lam + g * h * ex * (z - 1 / 2 * z**2)

    u_num = (
        lambda z, a0, a1, a2: a0 * np.ones_like(z)
        + a1 * (1 - 2 * z)
        + a2 * (1 - 6 * z + 6 * z**2)
    )

    fig, ax = plt.subplots()
    ax.plot(u_num(Z, 4 / 3, -1 / 4, -1 / 12), Z, label="numical")
    ax.plot(u_sol(Z), Z, "r*", label="exact")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # solution_slip()
    # sol = solve_constains_no_space()
    # print(sol)
    # sin_problem()
    # solution_ref_eq()
    solution_finder()
