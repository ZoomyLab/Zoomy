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



from zoomy_core.model.basemodel import (
    register_sympy_attribute,
    eigenvalue_dict_to_matrix,
)
from zoomy_core.model.basemodel import Model
import zoomy_core.model.initial_conditions as IC
from zoomy_core.model.models.basismatrices import Basismatrices, Legendre_shifted, Basisfunction


class HybridSFFSMM(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        aux_initial_conditions=IC.Constant(),
        dimension=1,
        fields=3,
        aux_variables=0,
        parameters={},
        _default_parameters={"g": 1.0, "ex": 0.0, "ez": 1.0},
        basismatrices=Basismatrices(),
    ):
        self.basismatrices = basis
        self.variables = register_sympy_attribute(fields, "q")
        self.n_variables = self.variables.length()
        self.level = self.n_variables - 3
        self.basismatrices.compute_matrices(self.level)
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_variables=aux_variables,
            parameters=parameters,
            _default_parameters=_default_parameters,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            aux_initial_conditions=aux_initial_conditions,
            settings={**settings_default, **settings},
        )

    def get_alphas(self):
        Q = self.variables
        h = Q[0]
        # exlude h and P
        ha = Q[1:-1]
        return ha

    def flux(self):
        flux_x = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.get_alphas()
        a = [_ha / h for _ha in ha]
        hu = self.variables[1]
        u = hu / h
        P11 = self.variables[-1]
        p = self.parameters
        # mass malance
        flux_x[0] = ha[0]
        # mean momentum (following SSF)
        flux_x[1] = hu * u + h * P11 + p.g * h**2 / 2
        # flux_x[1] = 0
        # for i in range(self.level+1):
        #     for j in range(self.level+1):
        #         flux_x[1] += ha[i] * ha[j] / h * self.basismatrices.A[0, i, j] / self.basismatrices.M[ 0, 0 ]
        # higher order moments (SMM)
        for k in range(1, self.level + 1):
            for i in range(self.level + 1):
                for j in range(self.level + 1):
                    flux_x[k + 1] += (
                        h
                        * a[i]
                        * a[j]
                        * self.basismatrices.A[k, i, j]
                        / self.basismatrices.M[k, k]
                    )
        # P
        flux_x[-1] = 2 * P11 * u
        for k in range(1, self.level + 1):
            for i in range(1, self.level + 1):
                for j in range(1, self.level + 1):
                    flux_x[-1] += a[i] * a[j] * a[k] * self.basismatrices.A[k, i, j]
        return [flux_x]

    def nonconservative_matrix(self):
        nc = Matrix([[0 for i in range(self.n_variables)] for j in range(self.n_variables)])
        h = self.variables[0]
        ha = self.get_alphas()
        hu = self.variables[1]
        u = hu / h
        a = [_ha / h for _ha in ha]
        P11 = self.variables[-1]
        p = self.parameters
        um = ha[0] / h

        # mean momentum
        # nc[1, 0] = - p.g * p.ez * h
        nc[1, 0] = 0
        # higher order momennts (SMM)
        for k in range(1, self.level + 1):
            nc[k + 1, k + 1] += um
        for k in range(self.level + 1):
            for i in range(1, self.level + 1):
                for j in range(1, self.level + 1):
                    nc[k + 1, i + 1] -= (
                        ha[j]
                        / h
                        * self.basismatrices.B[k, i, j]
                        / self.basismatrices.M[k, k]
                    )
        nc[-1, -1] = u
        return [nc]

    def eigenvalues(self):
        A = self.normal[0] * self.sympy_quasilinear_matrix[0]
        for d in range(1, self.dimension):
            A += self.normal[d] * self.sympy_quasilinear_matrix[d]
        # alpha_erase = self.variables[2:]
        # for alpha_i in alpha_erase:
        #     A = A.subs(alpha_i, 0)
        # return eigenvalue_dict_to_matrix(A.eigenvals())
        evs = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        hu = self.variables[1]
        u = hu / h
        P11 = self.variables[2]
        p = self.parameters

        b = sympy.sqrt(P11)
        a = sympy.sqrt(p.g * h + 3 * P11)

        evs[0] = u
        evs[1] = u + a
        evs[2] = u - a

        return evs

    def source(self):
        out = Matrix([0 for i in range(self.n_variables)])
        return out

    def topography(self):
        assert "dhdx" in vars(self.aux_variables)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        p = self.parameters
        dhdx = self.aux_variables.dhdx
        out[1] = h * p.g * (p.ex - p.ez * dhdx)
        return out


class ShallowMomentsAugmentedSSF(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        aux_initial_conditions=IC.Constant(),
        dimension=1,
        fields=3,
        aux_variables=0,
        parameters={},
        _default_parameters={"g": 1.0, "ex": 0.0, "ez": 1.0},
        settings={},
        settings_default={"topography": False, "friction": []},
        basis=Legendre_shifted(),
    ):
        self.basismatrices = basis
        self.variables = register_sympy_attribute(fields, "q")
        self.n_variables = self.variables.length()
        self.level = self.n_variables - 2
        self.basismatrices.compute_matrices(self.level)
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_variables=aux_variables,
            parameters=parameters,
            _default_parameters=_default_parameters,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            aux_initial_conditions=aux_initial_conditions,
            settings={**settings_default, **settings},
        )

    def get_alphas(self):
        Q = self.variables
        h = Q[0]
        # exlude u0 and P
        ha = Q[2:-1]
        P = Q[-1]
        sum_aiMii = 0
        N = self.level
        for i, hai in enumerate(ha):
            sum_aiMii += hai * self.basismatrices.M[i + 1, i + 1] / h

        aN = sympy.sqrt((P - h * sum_aiMii) / (h * self.basismatrices.M[N, N]))
        # now I want to include u0
        ha = Q[1:]
        ha[-1] = aN * h
        return ha

    def flux(self):
        flux = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.get_alphas()
        P11 = self.variables[-1]
        p = self.parameters
        flux[0] = ha[0]
        flux[1] = p.g * p.ez * h * h / 2
        for k in range(self.level + 1):
            for i in range(self.level + 1):
                for j in range(self.level + 1):
                    # TODO avoid devision by zero
                    flux[k + 1] += (
                        ha[i]
                        * ha[j]
                        / h
                        * self.basismatrices.A[k, i, j]
                        / self.basismatrices.M[k, k]
                    )
        flux[-1] = 2 * P11 * ha[0] / h
        return [flux]

    def nonconservative_matrix(self):
        nc = Matrix([[0 for i in range(self.n_variables)] for j in range(self.n_variables)])
        h = self.variables[0]
        ha = self.get_alphas()
        P11 = self.variables[-1]
        p = self.parameters
        um = ha[0] / h
        # nc[1, 0] = - p.g * p.ez * h
        for k in range(1, self.level + 1):
            nc[k + 1, k + 1] += um
        for k in range(self.level + 1):
            for i in range(1, self.level + 1):
                for j in range(1, self.level + 1):
                    nc[k + 1, i + 1] -= (
                        ha[j]
                        / h
                        * self.basismatrices.B[k, i, j]
                        / self.basismatrices.M[k, k]
                    )
        for k in range(nc.shape[1]):
            nc[-1, k] = 0
        nc[-1, 1] = -P11
        nc[-1, -1] = +P11
        return [nc]

    def eigenvalues(self):
        A = self.normal[0] * self.sympy_quasilinear_matrix[0]
        for d in range(1, self.dimension):
            A += self.normal[d] * self.sympy_quasilinear_matrix[d]
        # alpha_erase = self.variables[2:]
        # for alpha_i in alpha_erase:
        #     A = A.subs(alpha_i, 0)
        return eigenvalue_dict_to_matrix(A.eigenvals())

    def source(self):
        out = Matrix([0 for i in range(self.n_variables)])
        return out

    def topography(self):
        assert "dhdx" in vars(self.aux_variables)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        p = self.parameters
        dhdx = self.aux_variables.dhdx
        out[1] = h * p.g * (p.ex - p.ez * dhdx)
        return out


class ShallowMoments(Model):
    """
    Shallow Moments 1d

    :gui:
    - tab: model
    - requires: [ 'mesh.dimension': 1 ]

    """

    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        aux_initial_conditions=IC.Constant(),
        dimension=1,
        fields=2,
        aux_variables=0,
        parameters={},
        _default_parameters={"g": 1.0, "ex": 0.0, "ez": 1.0},
        settings={},
        settings_default={"topography": False, "friction": []},
        basis=Basismatrices(),
    ):
        self.variables = register_sympy_attribute(fields, "q")
        self.n_variables = self.variables.length()
        self.level = self.n_variables - 2
        self.basismatrices = basis
        self.basismatrices.basisfunctions = type(self.basismatrices.basisfunctions)(
            level=self.level
        )
        self.basismatrices.compute_matrices(self.level)
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_variables=aux_variables,
            parameters=parameters,
            _default_parameters=_default_parameters,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            aux_initial_conditions=aux_initial_conditions,
            settings={**settings_default, **settings},
        )

    def flux(self):
        flux = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        flux[0] = ha[0]
        flux[1] = p.g * p.ez * h * h / 2
        for k in range(self.level + 1):
            for i in range(self.level + 1):
                for j in range(self.level + 1):
                    # TODO avoid devision by zero
                    flux[k + 1] += (
                        ha[i]
                        * ha[j]
                        / h
                        * self.basismatrices.A[k, i, j]
                        / self.basismatrices.M[k, k]
                    )
        return [flux]

    def nonconservative_matrix(self):
        nc = Matrix([[0 for i in range(self.n_variables)] for j in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        um = ha[0] / h
        # nc[1, 0] = - p.g * p.ez * h
        for k in range(1, self.level + 1):
            nc[k + 1, k + 1] += um
        for k in range(self.level + 1):
            for i in range(1, self.level + 1):
                for j in range(1, self.level + 1):
                    nc[k + 1, i + 1] -= (
                        ha[j]
                        / h
                        * self.basismatrices.B[k, i, j]
                        / self.basismatrices.M[k, k]
                    )
        return [nc]

    def eigenvalues(self):
        A = self.normal[0] * self.sympy_quasilinear_matrix[0]
        for d in range(1, self.dimension):
            A += self.normal[d] * self.sympy_quasilinear_matrix[d]
        alpha_erase = self.variables[2:]
        for alpha_i in alpha_erase:
            A = A.subs(alpha_i, 0)
        return eigenvalue_dict_to_matrix(A.eigenvals())


    def source(self):
        out = Matrix([0 for i in range(self.n_variables)])
        return out

    def topography(self):
        assert "dhdx" in vars(self.aux_variables)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        p = self.parameters
        dhdx = self.aux_variables.dhdx
        out[1] = h * p.g * (p.ex - p.ez * dhdx)
        return out

    def inclined_plane(self):
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        p = self.parameters
        out[1] = h * p.g * (p.ex)
        return out

    def material_wave(self):
        assert "nu" in vars(self.parameters)
        assert "rho" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        return out

    def newtonian(self):
        """
        :gui:
            - requires_parameter: ('nu', 0.0)
        """
        assert "nu" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        for k in range(1 + self.level):
            for i in range(1 + self.level):
                # out[1+k] += -p.nu/h * p.eta_bulk * ha[i]  / h * self.basismatrices.D[i, k]/ self.basismatrices.M[k, k]
                out[1 + k] += (
                    -p.nu
                    / h
                    * ha[i]
                    / h
                    * self.basismatrices.D[i, k]
                    / self.basismatrices.M[k, k]
                )
        return out

    def newtonian_no_ibp(self):
        """
        :gui:
            - requires_parameter: ('nu', 0.0)
        """
        assert "nu" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        for k in range(1 + self.level):
            for i in range(1 + self.level):
                # out[1+k] += -p.nu/h * p.eta_bulk * ha[i]  / h * self.basismatrices.D[i, k]/ self.basismatrices.M[k, k]
                out[1 + k] += (
                    -p.nu
                    / h
                    * ha[i]
                    / h
                    * self.basismatrices.DD[k, i]
                    / self.basismatrices.M[k, k]
                )
        return out

    def newtonian_turbulent(self):
        """
        :gui:
            - requires_parameter: ('nu', 0.0)
        """
        p = self.parameters
        nut1 = [
            1.06245397e-05,
            -8.64966128e-06,
            -4.24655215e-06,
            1.51861028e-06,
            2.25140517e-06,
            1.81867029e-06,
            -1.02154323e-06,
            -1.78795289e-06,
            -5.07515843e-07,
        ]
        nut2 = np.array(
            [
                0.21923893,
                -0.04171894,
                -0.05129916,
                -0.04913612,
                -0.03863209,
                -0.02533469,
                -0.0144186,
                -0.00746847,
                -0.0031811,
                -0.00067986,
                0.0021782,
            ]
        )
        nut2 = nut2 / nut2[0] * 1.06245397 * 10 ** (-5)
        nut3 = [
            1.45934315e-05,
            -1.91969629e-05,
            5.80456268e-06,
            -5.13207491e-07,
            2.29489571e-06,
            -1.24361978e-06,
            -2.78720732e-06,
            -2.01469118e-07,
            1.24957663e-06,
        ]
        nut4 = [
            1.45934315e-05,
            -1.45934315e-05 * 3 / 4,
            -1.45934315e-05 * 1 / 4,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        nut5 = [p.nut, -p.nut * 3 / 4, -p.nut * 1 / 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        nut = nut5
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        for k in range(1 + self.level):
            for i in range(1 + self.level):
                for j in range(1 + self.level):
                    out[1 + k] += (
                        -p.c_nut
                        * nut[j]
                        / h
                        * ha[i]
                        / h
                        * self.basismatrices.DT[k, i, j]
                        / self.basismatrices.M[k, k]
                    )
        return out

    def newtonian_boundary_layer_classic(self):
        assert "nu" in vars(self.parameters)
        assert "eta" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        phi_0 = [
            self.basismatrices.basisfunctions.eval(i, 0.0)
            for i in range(self.level + 1)
        ]
        dphidx_0 = [
            (diff(self.basismatrices.basisfunctions.eval(i, x), x)).subs(x, 0.0)
            for i in range(self.level + 1)
        ]
        tau_bot = 0
        for i in range(1 + self.level):
            tau_bot += ha[i] / h * dphidx_0[i]
        for k in range(1 + self.level):
            out[k + 1] = (
                -p.c_bl
                * p.eta
                * (p.nu + p.nut_bl)
                / h
                * tau_bot
                * phi_0[k]
                / self.basismatrices.M[k, k]
            )
        return out

    def newtonian_boundary_layer(self):
        assert "nu" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        phi_0 = [self.basismatrices.basis.eval(i, 0.0) for i in range(self.level + 1)]
        dphidx_0 = [
            (diff(self.basismatrices.basis.eval(i, x), x)).subs(x, 0.0)
            for i in range(self.level + 1)
        ]
        dz_boundary_layer = 0.005
        u_bot = 0
        for i in range(1 + self.level):
            u_bot += ha[i] / h / self.basismatrices.M[i, i]
        tau_bot = p.nu * (u_bot - 0.0) / dz_boundary_layer
        for k in range(1 + self.level):
            out[k + 1] = -p.eta * tau_bot / h
        return out

    def steady_state_channel(self):
        assert "eta_ss" in vars(self.parameters)
        moments_ss = np.array(
            [
                0.21923893,
                -0.04171894,
                -0.05129916,
                -0.04913612,
                -0.03863209,
                -0.02533469,
                -0.0144186,
                -0.00746847,
                -0.0031811,
                -0.00067986,
                0.0021782,
            ]
        )[: self.level + 1]
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        mean = ha[0] / h
        mean_ss = moments_ss[0]
        scaling = mean / mean_ss
        p = self.parameters
        for i in range(1, self.level + 1):
            out[1 + i] = -p.eta_ss * h * (ha[i] / h - scaling * moments_ss[i])
        return out

    def slip(self):
        """
        :gui:
            - requires_parameter: ('lamda', 0.0)
            - requires_parameter: ('rho', 1.0)
        """
        assert "lamda" in vars(self.parameters)
        assert "rho" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        for k in range(1 + self.level):
            for i in range(1 + self.level):
                out[1 + k] += (
                    -1.0 / p.lamda / p.rho * ha[i] / h / self.basismatrices.M[k, k]
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
        return out

    def no_slip(self):
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        a0 = [ha[i] / h for i in range(self.level + 1)]
        a = [ha[i] / h for i in range(self.level + 1)]
        # for i in range(self.level+1):
        #     out[i+1] = ha[i]
        phi_0 = np.zeros(self.level + 1)
        ns_iterations = 2
        for k in range(self.level + 1):
            phi_0[k] = self.basismatrices.basis.eval(k, x).subs(x, 0.0)

        def f(j, a, a0, basis_0):
            out = 0
            for i in range(self.level + 1):
                # out += -2*p.ns_1*(a[i] - a0[i]) -2*p.ns_2*basis_0[j] * a[i] * basis_0[i]
                out += -2 * p.ns_2 * basis_0[j] * a[i] * basis_0[i]
            return out

        for i in range(ns_iterations):
            for k in range(1, 1 + self.level):
                out[1 + k] += h * (f(k, a, a0, phi_0))
            a = [a[k] + out[k] / h for k in range(self.level + 1)]
        # return sympy.simplify(out)
        return out

    def chezy(self):
        """
        :gui:
            - requires_parameter: ('C', 1000.0)
        """
        assert "C" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        tmp = 0
        for i in range(1 + self.level):
            for j in range(1 + self.level):
                tmp += ha[i] * ha[j] / h / h
        sqrt = sympy.sqrt(tmp)
        for k in range(1 + self.level):
            for l in range(1 + self.level):
                out[1 + k] += (
                    -1.0 / (p.C**2 * self.basismatrices.M[k, k]) * ha[l] * sqrt / h
                )
        return out

    def chezy_mean(self):
        """
        :gui:
            - requires_parameter: ('C', 1000.0)
        """
        assert "C" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        tmp = 0
        u = ha[0] / h
        for k in range(1 + self.level):
            out[1 + k] += -1.0 / (p.C**2 * self.basismatrices.M[k, k]) * u**2
        return out

    def chezy_ssf(self):
        """
        :gui:
            - requires_parameter: ('C', 1000.0)
        """
        assert "Cf" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        tmp = 0
        for i in range(1 + self.level):
            for j in range(1 + self.level):
                tmp += ha[i] * ha[j] / h / h
        sqrt = sympy.sqrt(tmp)
        for k in range(1 + self.level):
            for l in range(1 + self.level):
                out[1 + k] += -(p.Cf * self.basismatrices.M[k, k]) * ha[l] * sqrt / h
        return out

    def shear_new(self):
        """
        :gui:
            - requires_parameter: ('beta', 1.0)
        """
        assert "beta" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        tmp = 0
        d_phi_0 = np.zeros(self.level + 1)
        phi_0 = np.zeros(self.level + 1)
        for k in range(self.level + 1):
            d_phi_0[k] = diff(self.basismatrices.basis.eval(k, x), x).subs(x, 0.0)
            phi_0[k] = self.basismatrices.basis.eval(k, x).subs(x, 0.0)
        friction_factor = 0.0
        for k in range(self.level + 1):
            friction_factor -= p.beta * d_phi_0[k] * ha[k] / h / h
        k = 0
        # for k in range(1+self.level):
        out[1 + k] += friction_factor * phi_0[k] / (self.basismatrices.M[k, k])
        return out

    def shear(self):
        """
        :gui:
            - requires_parameter: ('beta', 1.0)
        """
        assert "beta" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        tmp = 0
        d_phi_0 = np.zeros(self.level + 1)
        phi_0 = np.zeros(self.level + 1)
        for k in range(self.level + 1):
            d_phi_0[k] = diff(self.basismatrices.basis.eval(k, x), x).subs(x, 0.0)
            phi_0[k] = self.basismatrices.basis.eval(k, x).subs(x, 0.0)
        friction_factor = 0.0
        for k in range(self.level + 1):
            friction_factor -= p.beta * d_phi_0[k] * ha[k] / h
        for k in range(self.level + 1):
            out[1 + k] += friction_factor * phi_0[k] / (self.basismatrices.M[k, k]) / h
        return out

    def shear_crazy(self):
        """
        :gui:
            - requires_parameter: ('beta', 1.0)
        """
        assert "beta" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        tmp = 0
        d_phi_0 = np.zeros(self.level + 1)
        phi_0 = np.zeros(self.level + 1)
        for k in range(self.level + 1):
            d_phi_0[k] = diff(self.basismatrices.basis.eval(k, x), x).subs(x, 0.0)
            phi_0[k] = self.basismatrices.basis.eval(k, x).subs(x, 0.0)
        for k in range(self.level + 1):
            out[1 + k] += -p.beta * d_phi_0[k] * phi_0[k] * ha[k] / h
        return out

    def manning_mean(self):
        """
        :gui:
            - requires_parameter: ('C', 1000.0)
        """
        assert "kst" in vars(self.parameters)
        assert "g" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        u = ha[0] / h
        for k in range(1 + self.level):
            out[1 + k] += (
                -1.0
                / p.kst**2
                * p.g
                / h ** (1 / 3)
                * u**2
                * (self.basismatrices.M[k, k])
            )
        return out

    def steady_state(self):
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        tmp = 0
        d_phi_0 = np.zeros(self.level + 1)
        phi_0 = np.zeros(self.level + 1)
        for k in range(self.level + 1):
            d_phi_0[k] = diff(self.basismatrices.basis.eval(k, x), x).subs(x, 0.0)
            phi_0[k] = self.basismatrices.basis.eval(k, x).subs(x, 0.0)
        shear_factor = 0.0
        u_bottom = 0.0
        u_diff = ha[0] / h
        for k in range(self.level + 1):
            u_diff += phi_0[k] * ha[k] / h
        for k in range(self.level + 1):
            shear_factor += d_phi_0[k] * (
                ha[k] / h - eval(f"p.Q_ss{k + 1}") / eval(f"p.Q_ss{0}")
            )
        # for k in range(1, self.level+1):
        #     out[1+k] += - shear_factor *  np.abs(u_diff) * p.S *  phi_0[k] /(self.basismatrices.M[k,k])
        for k in range(1, self.level + 1):
            # out[1+k] += - p.A * np.abs(u_diff)* (ha[k]/h - eval(f'p.Q_ss{k+1}')/eval(f'p.Q_ss{0}'))
            out[1 + k] += (
                -p.A
                * np.abs(ha[0] / h)
                * (ha[k] / h - eval(f"p.Q_ss{k + 1}") / eval(f"p.Q_ss{0}"))
            )
        return out


class ShallowMomentsSSF(ShallowMoments):
    def eigenvalues(self):
        A = self.normal[0] * self.sympy_quasilinear_matrix[0]
        for d in range(1, self.dimension):
            A += self.normal[d] * self.sympy_quasilinear_matrix[d]
        # alpha_erase = self.variables[2:]
        # for alpha_i in alpha_erase:
        #     A = A.subs(alpha_i, 0)
        return eigenvalue_dict_to_matrix(sympy.simplify(A).eigenvals())


class ShallowMomentsSSFEnergy(ShallowMoments):
    def flux(self):
        flux = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1] - self.variables[1] / np.diag(
            self.basismatrices.M
        )
        ha[1:] -= self.variables[1] * np.diag(self.basismatrices.M)[1:]
        p = self.parameters
        flux[0] = ha[0]
        flux[1] = p.g * p.ez * h * h / 2
        for k in range(self.level + 1):
            for i in range(self.level + 1):
                for j in range(self.level + 1):
                    # TODO avoid devision by zero
                    flux[k + 1] += (
                        ha[i]
                        * ha[j]
                        / h
                        * self.basismatrices.A[k, i, j]
                        / self.basismatrices.M[k, k]
                    )

        flux_hu = flux[1]

        for k in range(1, self.level + 1):
            flux[k + 1] += flux_hu / self.basismatrices.M[k, k]
        return [flux]

    def nonconservative_matrix(self):
        nc = Matrix([[0 for i in range(self.n_variables)] for j in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1] - self.variables[1] / np.diag(
            self.basismatrices.M
        )
        ha[1:] -= self.variables[1] * np.diag(self.basismatrices.M)[1:]
        p = self.parameters
        um = ha[0] / h
        # nc[1, 0] = - p.g * p.ez * h
        for k in range(1, self.level + 1):
            nc[k + 1, k + 1] += um
        for k in range(self.level + 1):
            for i in range(1, self.level + 1):
                for j in range(1, self.level + 1):
                    nc[k + 1, i + 1] -= (
                        ha[j]
                        / h
                        * self.basismatrices.B[k, i, j]
                        / self.basismatrices.M[k, k]
                    )
        return [nc]

    def chezy(self):
        """
        :gui:
            - requires_parameter: ('C', 1000.0)
        """
        assert "C" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1] - self.variables[1] / np.diag(
            self.basismatrices.M
        )
        ha[1:] -= self.variables[1] * np.diag(self.basismatrices.M)[1:]
        p = self.parameters
        tmp = 0
        for i in range(1 + self.level):
            for j in range(1 + self.level):
                tmp += ha[i] * ha[j] / h / h
        sqrt = sympy.sqrt(tmp)
        for k in range(1 + self.level):
            for l in range(1 + self.level):
                out[1 + k] += -1.0 / (p.C**2) * ha[l] / h * sqrt

        for k in range(1 + self.level):
            out[1 + k] += out[1] / self.basismatrices.M[k, k]

        return out

    def eigenvalues(self):
        A = self.normal[0] * self.sympy_quasilinear_matrix[0]
        for d in range(1, self.dimension):
            A += self.normal[d] * self.sympy_quasilinear_matrix[d]
        return eigenvalue_dict_to_matrix(sympy.simplify(A).eigenvals())


class ShallowMomentsTurbulenceSimple(ShallowMoments):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=1,
        fields=4,
        aux_variables=0,
        parameters={},
        _default_parameters={"g": 1.0, "ex": 0.0, "ey": 0.0, "ez": 1.0},
        settings={},
        settings_default={"topography": False, "friction": []},
        basis=Legendre_shifted(),
    ):
        super().__init__(
            dimension=dimension,
            fields=fields - 2,
            aux_variables=aux_variables,
            parameters=parameters,
            _default_parameters=_default_parameters,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            settings={**settings_default, **settings},
            basis=basis,
        )
        self.variables = register_sympy_attribute(fields, "q")
        self.n_variables = self.variables.length()
        assert self.n_variables >= 4


@define(frozen=True, slots=True, kw_only=True)
class ShallowMoments2d(Model):
    dimension: int = 2
    level: int
    variables: Union[list, int] = field(init=False)
    aux_variables: Union[list, int] = field(default=0)
    basis: Basis
    basis: Basismatrices = field(factory=Basismatrices)

    _default_parameters: dict = field(
        init=False,
        factory=lambda: {"g": 9.81, "ex": 0.0, "ey": 0.0, "ez": 1.0}
    )

    def __attrs_post_init__(self):
        object.__setattr__(self, "variables", (self.level+1*2)+1)
        super().__attrs_post_init__()

        # Recompute basis matrices
        basis = self.basis
        basis.basisfunctions = type(basis.basisfunctions)(level=self.level)
        basis.compute_matrices(self.level)
        object.__setattr__(self, "basis", basis)



    def project_2d_to_3d(self):
        out = Matrix([0 for i in range(5)])
        level = self.level
        offset = level+1
        x = self.position[0]
        y = self.position[1]
        z = self.position[2]
        h = self.variables[0]
        a = [self.variables[1+i]/h for i in range(offset)]
        b = [self.variables[1+offset+i]/h for i in range(offset)]
        dudx = self.aux_variables[0]
        dvdy = self.aux_variables[1]
        rho_w = 1000.
        g = 9.81
        # rho_3d = rho_w * Piecewise((1., h-z > 0), (0.,True))
        # u_3d = u*Piecewise((1, h-z > 0), (0, True))
        # v_3d = v*Piecewise((1, h-z > 0), (0, True))
        # w_3d = (-h * dudx - h * dvdy )*Piecewise((1, h-z > 0), (0, True))
        # p_3d = rho_w * g * Piecewise((h-z, h-z > 0), (0, True))
        u_3d = self.basismatrices.basisfunctions.reconstruct_velocity_profile_at(a, z)
        v_3d = self.basismatrices.basisfunctions.reconstruct_velocity_profile_at(b, z)
        out[0] = h
        out[1] = u_3d
        out[2] = v_3d
        out[3] = 0
        out[4] = rho_w * g * h * (1-z)

        return out

    def flux(self):
        offset = self.level + 1
        flux_x = Matrix([0 for i in range(self.n_variables)])
        flux_y = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        hb = self.variables[1 + self.level + 1 : 1 + 2 * (self.level + 1)]
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
        hb = self.variables[1 + offset : 1 + offset + self.level + 1]
        p = self.parameters
        um = ha[0] / h
        vm = hb[0] / h
        for k in range(1, self.level + 1):
            nc_x[k + 1, k + 1] += um
            nc_y[k + 1, k + 1 + offset] += um
        for k in range(self.level + 1):
            for i in range(1, self.level + 1):
                for j in range(1, self.level + 1):
                    nc_x[k + 1, i + 1] -= (
                        ha[j]
                        / h
                        * self.basismatrices.B[k, i, j]
                        / self.basismatrices.M[k, k]
                    )
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
        return [nc_x, nc_y]

    def eigenvalues(self):
        # we delete heigher order moments (level >= 2) for analytical eigenvalues
        offset = self.level + 1
        A = self.normal[0] * self.sympy_quasilinear_matrix[0]
        for d in range(1, self.dimension):
            A += self.normal[d] * self.sympy_quasilinear_matrix[d]
        alpha_erase = self.variables[2 : 2 + self.level]
        beta_erase = self.variables[2 + offset : 2 + offset + self.level]
        for alpha_i in alpha_erase:
            A = A.subs(alpha_i, 0)
        for beta_i in beta_erase:
            A = A.subs(beta_i, 0)
        return eigenvalue_dict_to_matrix(A.eigenvals())

    def constraints_implicit(self):
        assert "dhdx" in vars(self.aux_variables)
        assert "dhdy" in vars(self.aux_variables)
        out = Matrix([0 for i in range(1)])
        h = self.variables[0]
        hu = self.variables[1]
        hv = self.variables[2]
        p = self.parameters
        dhdt = self.aux_variables.dhdt
        dhudt = self.aux_variables.dhudt
        dhvdt = self.aux_variables.dhvdt
        dhudx = self.aux_variables.dhudx
        dhudy = self.aux_variables.dhudy
        dhvdx = self.aux_variables.dhudx
        dhvdy = self.aux_variables.dhudy
        out[0] = dhdt + dhudx + dhvdx
        out[1] = dhudt + dhudx + dhvdx
        return out

    def source(self):
        out = Matrix([0 for i in range(self.n_variables)])
        return out

    def topography(self):
        assert "dhdx" in vars(self.aux_variables)
        assert "dhdy" in vars(self.aux_variables)
        offset = self.level + 1
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        p = self.parameters
        dhdx = self.aux_variables.dhdx
        dhdy = self.aux_variables.dhdy
        out[1] = h * p.g * (p.ex - p.ez * dhdx)
        out[1 + offset] = h * p.g * (p.ey - p.ez * dhdy)
        return out

    def newtonian(self):
        assert "nu" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        offset = self.level + 1
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        hb = self.variables[1 + offset : 1 + self.level + 1 + offset]
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
        hb = self.variables[1+offset : 1+offset + self.level + 1]
        p = self.parameters
        ub = 0
        vb = 0
        for i in range(1 + self.level):
            ub += ha[i] / h
            vb += hb[i] / h
        for k in range(1, 1 + self.level):
            out[1 + k] += (
                -1.0 * p.c_slipmod / p.lamda / p.rho * ub / self.basismatrices.M[k, k]
            )
            out[1+offset+k] += (
                -1.0 * p.c_slipmod / p.lamda / p.rho * vb / self.basismatrices.M[k, k]
            )
        return out

    def newtonian_boundary_layer(self):
        assert "nu" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        offset = self.level + 1
        h = self.variables[0]
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        hb = self.variables[1 + offset : 1 + self.level + 1 + offset]
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
        hb = self.variables[1 + offset : 1 + self.level + 1 + offset]
        p = self.parameters
        for k in range(1 + self.level):
            for i in range(1 + self.level):
                out[1 + k] += (
                    -1.0 / p.lamda / p.rho * ha[i] / h / self.basismatrices.M[k, k]
                )
                out[1 + k + offset] += (
                    -1.0 / p.lamda / p.rho * hb[i] / h / self.basismatrices.M[k, k]
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
