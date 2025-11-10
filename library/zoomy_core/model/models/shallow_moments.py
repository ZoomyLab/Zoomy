import numpy as np
import numpy.polynomial.legendre as L
import numpy.polynomial.chebyshev as C
from scipy.optimize import least_squares as lsq
import sympy
from sympy import Matrix, Piecewise, sqrt
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
from zoomy_core.model.models.basismatrices import Basismatrices
from zoomy_core.model.models.basisfunctions import Legendre_shifted, Basisfunction



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
        



    def project_2d_to_3d(self):
        out = Matrix([0 for i in range(6)])
        level = self.level
        offset = level+1
        x = self.position[0]
        y = self.position[1]
        z = self.position[2]
        h = self.variables[0]
        a = [self.variables[1+i]/h for i in range(offset)]
        dhdx = self.aux_variables[0]
        dadx = [self.aux_variables[1+i] for i in range(offset)]
        
        psi = [self.basisfunctions.eval_psi(k, z) for k in range(level+1)]
        phi = [self.basisfunctions.eval(k, z) for k in range(level+1)]

        rho_w = 1000.
        g = 9.81
        u_3d = self.basismatrices.basisfunctions.reconstruct_velocity_profile_at(a, z)
        v_3d = 0
        b = 0
        dbdx = 0
        def dot(a, b):
            s = 0
            for i in range(len(a)):
                s += a[i] * b[i]
            return s
        w_3d = - dhdx * dot(a,psi) - h * dot(dadx,psi) + dot(a, phi) * (z * dhdx + dbdx)
        if self.dimension == 2:
            beta = [self.variables[1+offset+i]/h for i in range(offset)]
            v_3d = self.basismatrices.basisfunctions.reconstruct_velocity_profile_at(beta, z)

        b = 0
        out[0] = b
        out[1] = h
        out[2] = u_3d
        out[3] = v_3d
        out[4] = w_3d
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
                        * self.basismatrices.A[k, i, j] / self.basismatrices.M[k, k]
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
                            * self.basismatrices.A[k, i, j]/ self.basismatrices.M[k, k]
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
                            * self.basismatrices.A[k, i, j]/ self.basismatrices.M[k, k]
                        )
            for k in range(self.level + 1):
                for i in range(self.level + 1):
                    for j in range(self.level + 1):
                        # TODO avoid devision by zero
                        flux_y[k + 1 + offset] += (
                            hb[i]
                            * hb[j]
                            / h
                            * self.basismatrices.A[k, i, j]/ self.basismatrices.M[k, k]
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
                        * self.basismatrices.B[k, i, j]/ self.basismatrices.M[k, k]
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
                            * self.basismatrices.B[k, i, j]/ self.basismatrices.M[k, k]
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
                        )
                        nc_y[k + 1 + offset, i + 1 + offset] -= (
                            hb[j]
                            / h
                            * self.basismatrices.B[k, i, j]/ self.basismatrices.M[k, k]
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
    
    def gravity(self):
        out = Matrix([0 for i in range(self.n_variables)])
        out[1] = -self.parameters.g * self.parameters.ex * self.variables[0]
        if self.dimension == 2:
            offset = self.level + 1
            out[1 + offset] = -self.parameters.g * self.parameters.ey * self.variables[0]
        return out
    
    def newtonian_turbulent(self):
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
                        * self.basismatrices.DT[k, i, j]/ self.basismatrices.M[k, k]
                    )
        return  out
    
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
                * phi_0[k]/ self.basismatrices.M[k, k]
            )
        return  out

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
                    * self.basismatrices.D[i, k]/ self.basismatrices.M[k, k]
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
                        * self.basismatrices.D[i, k]/ self.basismatrices.M[k, k]
                    )

        return out
    
    def newtonian_turbulent_algebraic(self):
        assert "nu" in vars(self.parameters)
        assert "l_bl" in vars(self.parameters)
        assert "l_turb" in vars(self.parameters)
        assert "kappa" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        offset = self.level + 1
        h = self.variables[0]
        a = [_ha / h for _ha in self.variables[1 : 1 + self.level + 1]]
        p = self.parameters
        dU_dx = a[0] / (p.l_turb * h)
        abs_dU_dx = sympy.Piecewise((dU_dx, dU_dx >=0), (-dU_dx, True))
        for k in range(1 + self.level):
            out[1 + k] += (
                -(p.nu + p.kappa * sympy.sqrt(p.nu * abs_dU_dx) * p.l_bl * ( 1-p.l_bl)) * dU_dx * self.basismatrices.phib[k] / h/ self.basismatrices.M[k, k]
            )
            for i in range(1 + self.level):
                out[1 + k] += (
                    -p.nu
                    / h
                    * a[i]
                    * self.basismatrices.D[i, k]/ self.basismatrices.M[k, k]
                )
                out[1 + k] += (
                    -p.kappa * sympy.sqrt(p.nu * abs_dU_dx)
                    / h
                    * a[i]
                    * (self.basismatrices.Dxi[i, k] - self.basismatrices.Dxi2[i, k])/ self.basismatrices.M[k, k]
                )
        if self.dimension == 2:
            b = [_hb / h for _hb in self.variables[1 + offset : 1 + self.level + 1 + offset] ]
            dV_dy = b[0] / (p.l_turb * h)
            abs_dV_dy = sympy.Piecewise((dV_dy, dV_dy >=0), (-dV_dy, True))
            for k in range(1 + self.level):
                out[1 + k + offset] += (
                    -(p.nu + p.kappa * sympy.sqrt(p.nu * abs_dV_dy) * p.l_bl * ( 1-p.l_bl)) * dV_dy * self.basismatrices.phib[k] / h/ self.basismatrices.M[k, k]
                )
                for i in range(1 + self.level):
                    out[1 + k + offset] += (
                        -p.nu
                        / h
                        * b[i]
                        * self.basismatrices.D[i, k]/ self.basismatrices.M[k, k]
                    )
                    out[1 + k + offset] += (
                    -p.kappa * sympy.sqrt(p.nu * abs_dV_dy)
                    / h
                    * b[i]
                    * (self.basismatrices.Dxi[i, k] - self.basismatrices.Dxi2[i, k])/ self.basismatrices.M[k, k]
                )

        return out
    
    def regress_against_power_profile(self):
        """
        :gui:
            - requires_parameter: ('lamda', 0.0)
            - requires_parameter: ('rho', 1.0)
        """
        assert "r_pp" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        offset = self.level+1
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        ub = 0
        Z = np.linspace(0,1,100)
        U = 1-(1-Z)**8
        power_profile_coefs = self.basisfunctions.project_onto_basis(U)
        for i in range(1 + self.level):
            ub += ha[i] / h
        for k in range(1, 1 + self.level):
            out[1 + k] += (
                -p.r_pp * sympy.Piecewise((ub, ub >=0), (-ub, True)) * (ha[i] - ha[0]*power_profile_coefs[i])/ self.basismatrices.M[k, k]
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
                -p.c_slipmod / p.lamda / p.rho * ub / self.basismatrices.M[k, k]
            )
        if self.dimension == 2:
            hb = self.variables[1+offset : 1+offset + self.level + 1]
            vb = 0
            for i in range(1 + self.level):
                vb += hb[i] / h
            for k in range(1, 1 + self.level):
                out[1+offset+k] += (
                    -1.0 * p.c_slipmod / p.lamda / p.rho * vb/ self.basismatrices.M[k, k]
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
                    * phi_0[k]
                    * dphidx_0[i]/ self.basismatrices.M[k, k]
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
                        * phi_0[k]
                        * dphidx_0[i]/ self.basismatrices.M[k, k]
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
                        -1.0 / p.lamda / p.rho/ self.basismatrices.M[k, k]
                    )
        return out

    def chezy(self):
        assert "C" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.level + 1]
        p = self.parameters
        tmp = 0
        if self.dimension == 1:
            for i in range(1 + self.level):
                for j in range(1 + self.level):
                    tmp += ha[i] * ha[j] / h / h
            sqrt = sympy.sqrt(tmp)
            for k in range(1 + self.level):
                for l in range(1 + self.level):
                    out[1 + k] += (
                        -1.0 / (p.C**2) * ha[l] * sqrt / h/ self.basismatrices.M[k, k]
                    )
        if self.dimension == 2:
            offset = self.level + 1
            hb = self.variables[1 + offset : 1 + self.level + 1 + offset]
            for i in range(1 + self.level):
                for j in range(1 + self.level):
                    tmp += ha[i] * ha[j] / h / h + hb[i] * hb[j] / h / h
            sqrt = sympy.sqrt(tmp)
            for k in range(1 + self.level):
                for l in range(1 + self.level):
                    out[1 + k] += (
                        -1.0 / (p.C**2) * ha[l] * sqrt / h/ self.basismatrices.M[k, k]
                    )
                    out[1 + k + offset] += (
                        -1.0 / (p.C**2) * hb[l] * sqrt / h/ self.basismatrices.M[k, k]
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
