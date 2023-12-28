import numpy as np
import os
import sys
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import griddata
from scipy.optimize import linprog

from library.solver.baseclass import BaseYaml  # nopep8
import library.solver.limiter as limiters  # nopep8

main_dir = os.getenv("SMPYTHON")


# returns functions u(z), v(z), w(z)
def reconstruct_uvw(Q, grad, lvl, matrices):
    m = matrices
    basis = lambda z: np.array([m.basis(i)(z) for i in range(lvl + 1)])
    basis_int = lambda z: np.array(
        [m.basis_integrate(m.basis(i))(z) for i in range(lvl + 1)]
    )
    offset = lvl + 1
    h = Q[0]
    alpha = Q[1 : 1 + offset] / h
    beta = Q[1 + offset : 1 + 2 * offset] / h
    dhalpha_dx = grad[1 : 1 + offset, 0]
    dhbeta_dy = grad[1 + offset : 1 + 2 * offset, 1]

    def u(z):
        u_z = 0
        for i in range(lvl + 1):
            u_z += alpha[i] * basis(z)[i]
        return u_z

    def v(z):
        v_z = 0
        for i in range(lvl + 1):
            v_z += beta[i] * basis(z)[i]
        return v_z

    def w(z):
        basis_0 = basis_int(0)
        basis_z = basis_int(z)
        u_z = 0
        v_z = 0
        grad_h = grad[0, :]
        grad_hb = grad[-1, :]
        result = 0
        for i in range(lvl + 1):
            u_z += alpha[i] * basis_z[i]
            v_z += beta[i] * basis_z[i]
        for i in range(lvl + 1):
            result -= dhalpha_dx[i] * (basis_z[i] - basis_0[i])
            result -= dhbeta_dy[i] * (basis_z[i] - basis_0[i])

        result += u_z * (z * grad_h[0] + grad_hb[0])
        result += v_z * (z * grad_h[1] + grad_hb[1])
        return result

    return u, v, w


# comute gradients based on FD using scattered pointwise data
def compute_gradient_field_2d(points, fields):
    def in_hull(points, probe):
        n_points = points.shape[0]
        n_dim = points.shape[1]
        c = np.zeros(n_points)
        A = np.r_[points.T, np.ones((1, n_points))]
        b = np.r_[probe, np.ones(1)]
        lp = linprog(c, A_eq=A, b_eq=b)
        return lp.success

    assert points.shape[1] == 2
    grad = np.zeros((fields.shape[0], 2, fields.shape[1]))
    eps_x = (points[:, 0].max() - points[:, 0].min()) / 100.0
    eps_y = (points[:, 1].max() - points[:, 1].min()) / 100.00

    # generate evaluation 'stencil' for central differences
    xi_0 = np.array(points)
    xi_xp = np.array(points)
    xi_xp[:, 0] += eps_x
    xi_xm = np.array(points)
    xi_xm[:, 0] -= eps_x
    xi_yp = np.array(points)
    xi_yp[:, 1] += eps_y
    xi_ym = np.array(points)
    xi_ym[:, 1] -= eps_y
    factors_x = 2.0 * np.ones((points.shape[0]))
    factors_y = 2.0 * np.ones((points.shape[0]))
    # correct boundary points with single sided differences
    for i in range(xi_xp.shape[0]):
        if not in_hull(points, xi_xp[i]):
            xi_xp[i, 0] -= eps_x
            factors_x[i] = 1.0
        if not in_hull(points, xi_xm[i]):
            xi_xm[i, 0] += eps_x
            factors_x[i] = 1.0
        if not in_hull(points, xi_yp[i]):
            xi_yp[i, 1] -= eps_y
            factors_y[i] = 1.0
        if not in_hull(points, xi_ym[i]):
            xi_ym[i, 1] += eps_y
            factors_y[i] = 1.0

    for i_field, values in enumerate(fields):
        f = griddata(points, values, xi_0)
        f_xp = griddata(points, values, xi_xp)
        f_xm = griddata(points, values, xi_xm)
        f_yp = griddata(points, values, xi_yp)
        f_ym = griddata(points, values, xi_ym)

        dfdx = (f_xp - f_xm) / (factors_x * eps_x + 10 ** (-10))
        dfdy = (f_yp - f_ym) / (factors_y * eps_y + 10 ** (-10))

        grad[i_field, 0, :] = dfdx
        grad[i_field, 1, :] = dfdy

    assert (np.isnan(grad) == False).all()
    assert (np.isfinite(grad) == True).all()
    return grad


# ps, vs: values at the boundary points
# p0, v0, value at the cell_center
def compute_gradient(ps, vs, p0, v0, limiter=lambda r: 1.0):
    points = np.zeros((ps.shape[0] + 1, 2))
    points[:-1, :] = ps[:, :2]
    points[-1, :] = p0[:2]
    values = np.zeros((vs.shape[0] + 1, vs.shape[1]))
    values[:-1, :] = vs
    values[-1, :] = v0

    f = LinearNDInterpolator(points, values)
    eps_x = (points[:, 0].max() - points[:, 0].min()) / 100.0
    eps_y = (points[:, 1].max() - points[:, 1].min()) / 100.00
    x0 = p0[0]
    y0 = p0[1]

    dfdx = (f(x0 + eps_x, y0) - f(x0 - eps_x, y0)) / (2 * eps_x + 10 ** (-10))
    rx = (f(x0, y0) - f(x0 - eps_x, y0)) / (f(x0 + eps_x, y0) - f(x0, y0) + 10 ** (-10))
    phix = limiter(rx)
    dfdy = (f(x0, y0 + eps_y) - f(x0, y0 - eps_y)) / (2 * eps_y + 10 ** (-10))
    ry = (f(x0, y0) - f(x0, y0 - eps_y)) / (f(x0, y0 + eps_y) - f(x0, y0) + 10 ** (-10))
    phiy = limiter(ry)

    grad = np.array([phix * dfdx, phiy * dfdy]).T
    assert (np.isnan(grad) == False).all()
    assert (np.isfinite(grad) == True).all()
    return grad


class Reconstruction(BaseYaml):
    yaml_tag = "!Reconstruction"

    def set_default_parameters(self):
        self.scheme = "interpolation"
        self.order = 0
        self.limiter = None
        self.cauchy_kav = False

    def set_runtime_variables(self):
        self.n_boundary_cells = 2
        if self.scheme == "weno":
            # TODO #WENO check
            self.n_boundary_cells = 2 + self.order

    def evaluate(self, Q, p2c, c2p, c2f, f2c, **kwargs):
        return getattr(sys.modules[__name__], self.scheme)(
            self.order, p2c, c2p, c2f, f2c, Q, self.limiter, self.cauchy_kav, **kwargs
        )


# TODO this is not very general... I think reconstruction should have a signature like this
# interpolation(order, fields, mesh, limiter_type) or
# interpolation(order, fields, dx_face, limiter_type) or
# can I do multiple interpolation options?
# I may also want to think about a 'mix' of scipy interpolator -> get gradient + limiter?
# Or are there libraries that do this out of the box?
def interpolation(order, p2c, c2p, c2f, f2c, Q, limiter_type, cauchy_kav, **kwargs):
    if order == 0:
        return recon_0st_order(f2c, Q)
    elif order == 1:
        return recon_1st_order(
            p2c, c2p, c2f, f2c, Q, limiter_type, cauchy_kav, **kwargs
        )
    else:
        assert False


def recon_0st_order(f2c, Q):
    Ql = Q[:, f2c[0]]
    Qr = Q[:, f2c[1]]
    return Ql, Qr


def recon_1st_order(p2c, c2p, c2f, f2c, Q, limiter_type, cauchy_kav, **kwargs):
    dx_face = kwargs["dx_face"]
    U = c2p(Q)
    Ul = U[:, f2c[0]]
    Ur = U[:, f2c[1]]
    # Ul = np.pad(U, ((0,0),(1,0)), mode='edge')
    # Ur = np.roll(Ul, -1, axis=1)
    edge_gradient = (Ur - Ul) / dx_face
    cell_gradient = limiters.limiter(
        limiter_type, edge_gradient[:, c2f[0]], edge_gradient[:, c2f[1]]
    )

    # TODO shouldn't this be dx instead of dx_face, since I am inside a cell.
    # Ul and Ur are the left and right side of the same face!
    # Therefore, I add (+) to get from the cell center to the right, which is a 'left face value'.
    Ul += 0.5 * dx_face * cell_gradient[:, f2c[0]]
    Ur -= 0.5 * dx_face * cell_gradient[:, f2c[1]]

    if cauchy_kav:
        model = kwargs["model"]
        dt = kwargs["dt"]
        dx = kwargs["dx"]
        # TODO add rhs and NC terms
        Qt = -(model.flux(Ul[:, c2f[1]]) - model.flux(Ur[:, c2f[0]])) / dx
        Ul += 0.5 * dt * Qt[:, f2c[0]]
        Ur += 0.5 * dt * Qt[:, f2c[1]]

    Ql = p2c(Ul)
    Qr = p2c(Ur)
    return Ql, Qr


def test_write_and_load():
    recon = Reconstruction()
    # print(yaml.dump(recon))
    recon.scheme = "weno"
    recon.order = 1
    recon.write_class_to_file()
    recon2 = Reconstruction.read_class_from_file(file_dir + "/output/", "config.yaml")
    # print(yaml.dump(recon2))
    assert recon2.scheme == "weno"


def test_recon_0st():
    Q = np.linspace(1, 6, 6).reshape((2, 3))
    f2c = np.array([[0, 0, 1, 2], [0, 1, 2, 2]])
    # print(recon_0st_order(f2c, Q))
    assert np.array_equal(
        recon_0st_order(f2c, Q),
        [[[1, 1, 2, 3], [4, 4, 5, 6]], [[1, 2, 3, 3], [4, 5, 6, 6]]],
    )


def test_recon_1st():
    recon = Reconstruction(order=1, limiter="central_differences")

    Q = np.linspace(1, 10, 10).reshape((2, 5))
    Q[1] *= 2
    f2c = np.array([[0, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 4]])
    c2f = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    dx_face = 1.0 * np.ones(6)
    identity = lambda x: x
    Ql, Qr = recon.evaluate(Q, identity, identity, c2f, f2c, dx_face)
    assert np.array_equal(Qr[1], [11.5, 13, 15, 17, 19.5, 19.5])


def test_smm_paper_recon():
    recon = Reconstruction(order=1, limiter="smm_paper_1")
    Q = np.array(
        [[4, 1, 2, 3, 4, 1], [6, 3, 4, 5, 6, 3], [10, 7, 8, 9, 10, 7]], dtype=float
    )
    f2c = np.array([[0, 0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 5]])
    c2f = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]])
    dx_face = 1.0 * np.ones(7)
    identity = lambda x: x
    Ql, Qr = recon.evaluate(Q, identity, identity, c2f, f2c, dx_face)
    Qlref = np.array(
        [[4, 1, 2.5, 3.5, 4, 1], [6, 3, 4.5, 5.5, 6, 3], [10, 7, 8.5, 9.5, 10, 7]],
        dtype=float,
    )
    Qrref = np.array(
        [[4, 1, 1.5, 2.5, 4, 1], [6, 3, 3.5, 4.5, 6, 3], [10, 7, 7.5, 8.5, 10, 7]],
        dtype=float,
    )
    # print(Ql[:,1:], '\n' ,Qlref)
    # print(Qr[:,:-1], '\n', Qrref)
    assert np.allclose(Ql[:, 1:], Qlref)
    assert np.allclose(Qr[:, :-1], Qrref)


if __name__ == "__main__":
    test_write_and_load()
    test_recon_0st()
    test_recon_1st()
    test_smm_paper_recon()
