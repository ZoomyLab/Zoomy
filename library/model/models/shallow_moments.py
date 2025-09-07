import numpy as np
import numpy.polynomial.legendre as L
import numpy.polynomial.chebyshev as C
from scipy.optimize import least_squares as lsq
import sympy
from sympy import Matrix, Piecewise
from sympy.abc import x

from sympy import integrate, diff
from sympy import legendre
from sympy import lambdify

from attrs import define, field
from typing import Union



from library.model.models.base import (
    register_sympy_attribute,
    eigenvalue_dict_to_matrix,
)
from library.model.models.base import Model
import library.model.initial_conditions as IC
from library.model.models.basismatrices import Basismatrices
from library.model.models.basisfunctions import Legendre_shifted, Basisfunction



@define(frozen=True, slots=True, kw_only=True)
class ShallowMoments2d(Model):
    dimension: int = 2
    level: int
    variables: Union[list, int] = field(init=False)
    aux_variables: Union[list, int] = field(default=2)
    basisfunctions: Union[Basisfunction, type[Basisfunction]] = field(default=Legendre_shifted)
    basismatrices: Basismatrices = field(init=False)

    _default_parameters: dict = field(
        init=False,
        factory=lambda: {"g": 9.81, "ex": 0.0, "ey": 0.0, "ez": 1.0}
    )

    def __attrs_post_init__(self):
        object.__setattr__(self, "variables", ((self.level+1)*self.dimension)+1)
        super().__attrs_post_init__()
        aux_variables = self.aux_variables
        aux_var_list = aux_variables.keys()
        if not aux_variables.contains("dudx"):
            aux_var_list += ["dudx"]
        if self.dimension == 2 and not aux_variables.contains("dvdy"):
            aux_var_list += ["dvdy"]
        object.__setattr__(self, "aux_variables", register_sympy_attribute(aux_var_list, "qaux_"))

        # Recompute basis matrices
        object.__setattr__(self, "basisfunctions", self.basisfunctions(level=self.level))
        basismatrices = Basismatrices(self.basisfunctions)
        basismatrices.compute_matrices(self.level)
        object.__setattr__(self, "basismatrices", basismatrices)



    def interpolate_3d(self):
        out = Matrix([0 for i in range(6)])
        level = self.level
        offset = level+1
        x = self.position[0]
        y = self.position[1]
        z = self.position[2]
        h = self.variables[0]
        a = [self.variables[1+i]/h for i in range(offset)]
        dudx = self.aux_variables.dudx

        rho_w = 1000.
        g = 9.81
        # rho_3d = rho_w * Piecewise((1., h-z > 0), (0.,True))
        # u_3d = u*Piecewise((1, h-z > 0), (0, True))
        # v_3d = v*Piecewise((1, h-z > 0), (0, True))
        # w_3d = (-h * dudx - h * dvdy )*Piecewise((1, h-z > 0), (0, True))
        # p_3d = rho_w * g * Piecewise((h-z, h-z > 0), (0, True))
        u_3d = self.basismatrices.basisfunctions.reconstruct_velocity_profile_at(a, z)
        if self.dimension == 2:
            b = [self.variables[1+offset+i]/h for i in range(offset)]
            dvdy = self.aux_variables.dvdy
            v_3d = self.basismatrices.basisfunctions.reconstruct_velocity_profile_at(b, z)
        else:
            v_3d = 0
            dvdy = 0
        b = 0
        out[0] = b
        out[1] = h
        out[2] = u_3d
        out[3] = v_3d
        out[4] = 0
        out[5] = rho_w * g * h * (1-z)

        return out

    def flux(self):
        offset = self.level + 1
        flux_x = Matrix([0 for i in range(self.n_variables)])
        flux_y = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        flux_x[0] = ha[0]
        flux_x[1] = p.g * p.ez * h * h / 2
        for k in range(self.level + 1):
            for i in range(self.level + 1):
                for j in range(self.level + 1):
                    # TODO avoid devision by zero
                    flux_x[k + 1] += (
                        ha[i]
                        * ha[j]
                        / h
                        * self.basismatrices.A[k, i, j]
                        / self.basismatrices.M[k, k]
                    )
        if self.dimension == 2:
            hb = self.variables[1 + self.level + 1 : 1 + 2 * (self.level + 1)]

            for k in range(self.level + 1):
                for i in range(self.level + 1):
                    for j in range(self.level + 1):
                        # TODO avoid devision by zero
                        flux_x[k + 1 + offset] += (
                            hb[i]
                            * ha[j]
                            / h
                            * self.basismatrices.A[k, i, j]
                            / self.basismatrices.M[k, k]
                        )

            flux_y[0] = hb[0]
            flux_y[1 + offset] = p.g * p.ez * h * h / 2
            for k in range(self.level + 1):
                for i in range(self.level + 1):
                    for j in range(self.level + 1):
                        # TODO avoid devision by zero
                        flux_y[k + 1] += (
                            hb[i]
                            * ha[j]
                            / h
                            * self.basismatrices.A[k, i, j]
                            / self.basismatrices.M[k, k]
                        )
            for k in range(self.level + 1):
                for i in range(self.level + 1):
                    for j in range(self.level + 1):
                        # TODO avoid devision by zero
                        flux_y[k + 1 + offset] += (
                            hb[i]
                            * hb[j]
                            / h
                            * self.basismatrices.A[k, i, j]
                            / self.basismatrices.M[k, k]
                    )
        return [flux_x, flux_y]

    def nonconservative_matrix(self):
        offset = self.level + 1
        nc_x = Matrix([[0 for i in range(self.n_variables)] for j in range(self.n_variables)])
        nc_y = Matrix([[0 for i in range(self.n_variables)] for j in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        um = ha[0] / h

        for k in range(1, self.level + 1):
            nc_x[k + 1, k + 1] += um
        for k in range(self.level + 1):
            for i in range(1, self.level + 1):
                for j in range(1, self.level + 1):
                    nc_x[k + 1, i + 1] -= (
                        ha[j]
                        / h
                        * self.basismatrices.B[k, i, j]
                        / self.basismatrices.M[k, k]
                    )

                        
        if self.dimension ==  2:
            hb = self.variables[1 + offset : 1 + offset + self.level + 1]
            vm = hb[0] / h
            for k in range(1, self.level + 1):
                nc_y[k + 1, k + 1 + offset] += um
            for k in range(self.level + 1):
                for i in range(1, self.level + 1):
                    for j in range(1, self.level + 1):
                        nc_y[k + 1, i + 1 + offset] -= (
                            ha[j]
                            / h
                            * self.basismatrices.B[k, i, j]
                            / self.basismatrices.M[k, k]
                        )

            for k in range(1, self.level + 1):
                nc_x[k + 1 + offset, k + 1] += vm
                nc_y[k + 1 + offset, k + 1 + offset] += vm
            for k in range(self.level + 1):
                for i in range(1, self.level + 1):
                    for j in range(1, self.level + 1):
                        nc_x[k + 1 + offset, i + 1] -= (
                            hb[j]
                            / h
                            * self.basismatrices.B[k, i, j]
                            / self.basismatrices.M[k, k]
                        )
                        nc_y[k + 1 + offset, i + 1 + offset] -= (
                            hb[j]
                            / h
                            * self.basismatrices.B[k, i, j]
                            / self.basismatrices.M[k, k]
                        )
        return [-nc_x, -nc_y]

    def eigenvalues(self):
        # we delete heigher order moments (level >= 2) for analytical eigenvalues
        offset = self.level + 1
        A = self.normal[0] * self.quasilinear_matrix()[0]
        for d in range(1, self.dimension):
            A += self.normal[d] * self.quasilinear_matrix()[d]
        alpha_erase = self.variables[2 : 2 + self.level]
        beta_erase = self.variables[2 + offset : 2 + offset + self.level]
        for alpha_i in alpha_erase:
            A = A.subs(alpha_i, 0)
        for beta_i in beta_erase:
            A = A.subs(beta_i, 0)
        return eigenvalue_dict_to_matrix(A.eigenvals())


    def source(self):
        out = Matrix([0 for i in range(self.n_variables)])
        return out

    def newtonian(self):
        assert "nu" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        offset = self.level + 1
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        for k in range(1 + self.level):
            for i in range(1 + self.level):
                out[1 + k] += (
                    -p.nu
                    / h
                    * ha[i]
                    / h
                    * self.basismatrices.D[i, k]
                    / self.basismatrices.M[k, k]
                )
        if self.dimension == 2:
            hb = self.variables[1 + offset : 1 + self.level + 1 + offset]
            for k in range(1 + self.level):
                for i in range(1 + self.level):
                    out[1 + k + offset] += (
                        -p.nu
                        / h
                        * hb[i]
                        / h
                        * self.basismatrices.D[i, k]
                        / self.basismatrices.M[k, k]
                    )

        return out

    def slip_mod(self):
        """
        :gui:
            - requires_parameter: ('lamda', 0.0)
            - requires_parameter: ('rho', 1.0)
        """
        assert "lamda" in vars(self.parameters)
        assert "rho" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        offset = self.level+1
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        ub = 0
        for i in range(1 + self.level):
            ub += ha[i] / h
        for k in range(1, 1 + self.level):
            out[1 + k] += (
                -1.0 * p.c_slipmod / p.lamda / p.rho * ub / self.basismatrices.M[k, k]
            )
        if self.dimension == 2:
            hb = self.variables[1+offset : 1+offset + self.level + 1]
            vb = 0
            for i in range(1 + self.level):
                vb += hb[i] / h
            for k in range(1, 1 + self.level):
                out[1+offset+k] += (
                    -1.0 * p.c_slipmod / p.lamda / p.rho * vb / self.basismatrices.M[k, k]
                )
        return out

    def newtonian_boundary_layer(self):
        assert "nu" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        offset = self.level + 1
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        phi_0 = [self.basismatrices.eval(i, 0.0) for i in range(self.level + 1)]
        dphidx_0 = [
            (diff(self.basismatrices.eval(i, x), x)).subs(x, 0.0)
            for i in range(self.level + 1)
        ]
        for k in range(1 + self.level):
            for i in range(1 + self.level):
                out[1 + k] += (
                    -p.nu
                    / h
                    * ha[i]
                    / h
                    / self.basismatrices.M[k, k]
                    * phi_0[k]
                    * dphidx_0[i]
                )
        if self.dimension==2:
            hb = self.variables[1 + offset : 1 + self.level + 1 + offset]
            for k in range(1 + self.level):
                for i in range(1 + self.level):
                    out[1 + k + offset] += (
                        -p.nu
                        / h
                        * hb[i]
                        / h
                        / self.basismatrices.M[k, k]
                        * phi_0[k]
                        * dphidx_0[i]
                    )
        return out

    def sindy(self):
        assert "nu" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        offset = self.level + 1
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        hb = self.variables[1 + offset : 1 + self.level + 1 + offset]
        p = self.parameters
        out[1] += (
            p.C1 * sympy.Abs(ha[0] / h)
            + p.C2 * sympy.Abs(ha[1] / h)
            + p.C3 * sympy.Abs(ha[0] / h) ** (7 / 3)
            + p.C4 * sympy.Abs(ha[1] / h) ** (7 / 3)
        )
        out[2] += (
            p.C5 * sympy.Abs(ha[0] / h)
            + p.C6 * sympy.Abs(ha[1] / h)
            + p.C7 * sympy.Abs(ha[0] / h) ** (7 / 3)
            + p.C8 * sympy.Abs(ha[1] / h) ** (7 / 3)
        )
        out[3] += (
            p.C1 * sympy.Abs(ha[0] / h)
            + p.C2 * sympy.Abs(ha[1] / h)
            + p.C3 * sympy.Abs(ha[0] / h) ** (7 / 3)
            + p.C4 * sympy.Abs(ha[1] / h) ** (7 / 3)
        )
        out[4] += (
            p.C5 * sympy.Abs(ha[0] / h)
            + p.C6 * sympy.Abs(ha[1] / h)
            + p.C7 * sympy.Abs(ha[0] / h) ** (7 / 3)
            + p.C8 * sympy.Abs(ha[1] / h) ** (7 / 3)
        )
        return out

    def slip(self):
        assert "lamda" in vars(self.parameters)
        assert "rho" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        offset = self.level + 1
        h = self.variables[0]
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        for k in range(1 + self.level):
            for i in range(1 + self.level):
                out[1 + k] += (
                    -1.0 / p.lamda / p.rho * ha[i] / h / self.basismatrices.M[k, k]
                )

        if self.dimension == 2:
            hb = self.variables[1 + offset : 1 + self.level + 1 + offset]
            for k in range(1 + self.level):
                for i in range(1 + self.level):
                    out[1 + k + offset] += (
                        -1.0 / p.lamda / p.rho
                    )
        return out

    def chezy(self):
        assert "C" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        offset = self.level + 1
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        hb = self.variables[1 + offset : 1 + self.level + 1 + offset]
        p = self.parameters
        tmp = 0
        for i in range(1 + self.level):
            for j in range(1 + self.level):
                tmp += ha[i] * ha[j] / h / h + hb[i] * hb[j] / h / h
        sqrt = sympy.sqrt(tmp)
        for k in range(1 + self.level):
            for l in range(1 + self.level):
                out[1 + k] += (
                    -1.0 / (p.C**2 * self.basismatrices.M[k, k]) * ha[l] * sqrt / h
                )
                out[1 + k + offset] += (
                    -1.0 / (p.C**2 * self.basismatrices.M[k, k]) * hb[l] * sqrt / h
                )
        return out


def reconstruct_uvw(Q, grad, lvl, phi, psi):
    """
    returns functions u(z), v(z), w(z)
    """
    offset = lvl + 1
    h = Q[0]
    alpha = Q[1 : 1 + offset] / h
    beta = Q[1 + offset : 1 + 2 * offset] / h
    dhalpha_dx = grad[1 : 1 + offset, 0]
    dhbeta_dy = grad[1 + offset : 1 + 2 * offset, 1]

    def u(z):
        u_z = 0
        for i in range(lvl + 1):
            u_z += alpha[i] * phi(z)[i]
        return u_z

    def v(z):
        v_z = 0
        for i in range(lvl + 1):
            v_z += beta[i] * phi(z)[i]
        return v_z

    def w(z):
        basis_0 = psi(0)
        basis_z = psi(z)
        u_z = 0
        v_z = 0
        grad_h = grad[0, :]
        # grad_hb = grad[-1, :]
        grad_hb = np.zeros(grad[0, :].shape)
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


def generate_velocity_profiles(
    Q,
    centers,
    model: Model,
    list_of_positions: list[np.ndarray],
):
    def find_closest_element(centers, pos):
        assert centers.shape[1] == np.array(pos).shape[0]
        return np.argmin(np.linalg.norm(centers - pos, axis=1))

    # find the closest element to the given position
    vertices = []
    for pos in list_of_positions:
        vertex = find_closest_element(centers, pos)
        vertices.append(vertex)

    Z = np.linspace(0, 1, 100)
    list_profiles = []
    list_means = []
    level = int((model.n_variables - 1) / model.dimension) - 1
    offset = level + 1
    list_h = []
    for vertex in vertices:
        profiles = []
        means = []
        for d in range(model.dimension):
            q = Q[vertex, :]
            h = q[0]
            coefs = q[1 + d * offset : 1 + (d + 1) * offset] / h
            profile = model.basis.basis.reconstruct_velocity_profile(coefs, Z=Z)
            mean = coefs[0]
            profiles.append(profile)
            means.append(mean)
        list_profiles.append(profiles)
        list_means.append(means)
        list_h.append(h)
    return list_profiles, list_means, list_of_positions, Z, list_h


if __name__ == "__main__":
    # basis = Legendre_shifted(1)
    # basis = Spline()
    # basis = OrthogonalSplineWithConstant(degree=2, knots=[0, 0.1, 0.3,0.5, 1,1])
    # basis=OrthogonalSplineWithConstant(degree=1, knots=[0,0, 0.02, 0.04, 0.06, 0.08, 0.1,  1])
    # basis=OrthogonalSplineWithConstant(degree=1, knots=[0,0, 0.1, 1])
    # basis.plot()

    # basis = Legendre_shifted(basis=Legendre_shifted(order=8))
    # f = basis.enforce_boundary_conditions()
    # q = np.array([[1., 0.1, 0., 0., 0., 0.], [1., 0.1, 0., 0., 3., 0.]])
    # print(f(q))

    # basis =Legendre_shifted(order=8)
    # basis.plot()
    # z = np.linspace(0,1,100)
    # f = basis.get_lambda(1)
    # print(f(z), f(1.0))
    # f = basis.get_lambda(1)
    # print(f(z))

    # X = np.linspace(0,1,100)
    # coef = np.array([0.2, -0.01, -0.1, -0.05, -0.04])
    # U = basis.basis.reconstruct_velocity_profile(coef, Z=X)
    # coef2 = coef*2
    # factor = 1.0 / 0.2
    # coef3  = coef * factor
    # U2 = basis.basis.reconstruct_velocity_profile(coef2, Z=X)
    # U3 = basis.basis.reconstruct_velocity_profile(coef3, Z=X)
    # fig, ax = plt.subplots()
    # ax.plot(U, X)
    # ax.plot(U2, X)
    # ax.plot(U3, X)
    # plt.show()

    # X = np.linspace(0,1,100)
    # nut = 10**(-5)
    # coef = np.array([nut, -nut, 0, 0, 0, 0, 0 ])
    # U = basis.basis.reconstruct_velocity_profile(coef, Z=X)
    # fig, ax = plt.subplots()
    # ax.plot(U, X)
    # plt.show()

    # nut = np.load('/home/ingo/Git/sms/nut_nut2.npy')
    # y = np.load('/home/ingo/Git/sms/nut_y2.npy')
    # coef = basis.basis.reconstruct_alpha(nut, y)
    # coef_offset = np.sum(coef)
    # coef[0] -= coef_offset
    # print(coef)
    # X = np.linspace(0,1,100)
    # _nut = basis.basis.reconstruct_velocity_profile(coef, Z=X)
    # fig, ax = plt.subplots()
    # ax.plot(_nut, X)
    # plt.show()

    basis = Legendre_shifted(basis=Legendre_shifted(level=2))
    basis.compute_matrices(2)
    print(basis.D)

@define(frozen=True, slots=True, kw_only=True)
class ShallowMoments(ShallowMoments2d):
    dimension: int = 1