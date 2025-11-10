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
import attr
from typing import Union, Dict, List


from zoomy_core.model.basemodel import (
    register_sympy_attribute,
    eigenvalue_dict_to_matrix,
)
from zoomy_core.model.basemodel import Model
from zoomy_core.model.models.basismatrices import Basismatrices
from zoomy_core.model.models.basisfunctions import Legendre_shifted, Basisfunction


    

@define(frozen=True, slots=True, kw_only=True)
class ShallowMomentsTopo(Model):
    dimension: int = 2
    level: int
    variables: Union[list, int] = field(init=False)
    positive_variables: Union[List[int], Dict[str, int], None] = attr.ib(default=attr.Factory(lambda: [1]))    
    aux_variables: Union[list, int] = field(default=0)
    basisfunctions: Union[Basisfunction, type[Basisfunction]] = field(default=Legendre_shifted)
    basismatrices: Basismatrices = field(init=False)

    _default_parameters: dict = field(
        init=False,
        factory=lambda: {"g": 9.81, "ex": 0.0, "ey": 0.0, "ez": 1.0, "eps_low_water": 1e-6, "rho": 1000},
    )

    def __attrs_post_init__(self):
        object.__setattr__(self, "variables", ((self.level+1)*self.dimension)+2)
        object.__setattr__(self, "aux_variables", 2*((self.level+1)*self.dimension+2))
        super().__attrs_post_init__()
        aux_variables = self.aux_variables
        aux_var_list = aux_variables.keys()
        object.__setattr__(self, "aux_variables", register_sympy_attribute(aux_var_list, "qaux_"))

        # Recompute basis matrices
        object.__setattr__(self, "basisfunctions", self.basisfunctions(level=self.level))
        basismatrices = Basismatrices(self.basisfunctions)
        basismatrices.compute_matrices(self.level)
        object.__setattr__(self, "basismatrices", basismatrices)

    def get_primitives(self):
        offset = self.level + 1
        b = self.variables[0]
        h = self.variables[1]
        hinv = 1/h
        ha = self.variables[2 : 2 + self.level + 1]
        alpha = [ha[i] * hinv for i in range(offset)]
        if self.dimension == 1:
            hb = [0 for i in range(self.level+1)]
        else:
            hb = self.variables[2 + offset : 2 + offset + self.level + 1]
        beta = [hb[i] * hinv for i in range(offset)]
        return [b, h, alpha, beta, hinv]


    def project_2d_to_3d(self):
        out = Matrix([0 for i in range(6)])
        level = self.level
        offset = level+1
        offset_aux = self.n_variables
        x = self.position[0]
        y = self.position[1]
        z = self.position[2]
        
        b, h, alpha, beta, hinv = self.get_primitives()
        dbdx = self.aux_variables[0]
        dhdx = self.aux_variables[1]
        dbdy = self.aux_variables[offset_aux]
        dhdy = self.aux_variables[1+offset_aux]
        dalphadx = [self.aux_variables[2+i] for i in range(offset)]
        if self.dimension == 2:
            dbetady = [self.aux_variables[2+i+offset_aux] for i in range(offset)]
        
        psi = [self.basisfunctions.eval_psi(k, z) for k in range(level+1)]
        phi = [self.basisfunctions.eval(k, z) for k in range(level+1)]

        rho_w = 1000.
        g = 9.81
        u_3d = self.basismatrices.basisfunctions.reconstruct_velocity_profile_at(alpha, z)
        v_3d = 0
        def dot(a, b):
            s = 0
            for i in range(len(a)):
                s += a[i] * b[i]
            return s
        w_3d = - dhdx * dot(alpha,psi) - h * dot(dalphadx,psi) + dot(alpha, phi) * (z * dhdx + dbdx)
        if self.dimension == 2:
            v_3d = self.basismatrices.basisfunctions.reconstruct_velocity_profile_at(beta, z)
            w_3d += - dhdy * dot(beta,psi) - h * dot(dbetady,psi) + dot(beta, phi) * (z * dhdy + dbdy)

        out[0] = b
        out[1] = h
        out[2] = u_3d
        out[3] = v_3d
        out[4] = w_3d
        out[5] = rho_w * g * h * (1-z)
        return out

    def flux(self):
        flux_x = Matrix([0 for i in range(self.n_variables)])
        flux_y = Matrix([0 for i in range(self.n_variables)])
        b, h, alpha, beta, hinv = self.get_primitives()
        p = self.parameters
        flux_x[1] = h * alpha[0]
        flux_x[2] = p.g * p.ez * h * h / 2
        for k in range(self.level + 1):
            for i in range(self.level + 1):
                for j in range(self.level + 1):
                    flux_x[k + 2] += (
                        h * alpha[i] * alpha[j]
                        * self.basismatrices.A[k, i, j]
                        / self.basismatrices.M[k, k]
                    )
        if self.dimension == 2:
            offset = self.level + 1
            p = self.parameters
            for k in range(self.level + 1):
                for i in range(self.level + 1):
                    for j in range(self.level + 1):
                        flux_x[1 + k + 1 + offset] += (
                            h * beta[i] * alpha[j]
                            * self.basismatrices.A[k, i, j]
                            / self.basismatrices.M[k, k]
                        )

            flux_y[1] = h * beta[0]
            flux_y[2 + offset] = p.g * p.ez * h * h / 2
            for k in range(self.level + 1):
                for i in range(self.level + 1):
                    for j in range(self.level + 1):
                        flux_y[1 + k + 1] += (
                            h * beta[i] * alpha[j]
                            * self.basismatrices.A[k, i, j]
                            / self.basismatrices.M[k, k]
                        )
            for k in range(self.level + 1):
                for i in range(self.level + 1):
                    for j in range(self.level + 1):
                        flux_y[1 + k + 1 + offset] += (
                            h * beta[i] * beta[j]
                            * self.basismatrices.A[k, i, j]
                            / self.basismatrices.M[k, k]
                        )
        return [flux_x, flux_y][:self.dimension]

    def nonconservative_matrix(self):
        nc_x = Matrix([[0 for i in range(self.n_variables)] for j in range(self.n_variables)])
        nc_y = Matrix([[0 for i in range(self.n_variables)] for j in range(self.n_variables)])
        b, h, alpha, beta, hinv = self.get_primitives()
        p = self.parameters
        um = alpha[0]
        for k in range(1, self.level + 1):
            nc_x[1+k + 1, 1+k + 1] += um
        for k in range(self.level + 1):
            for i in range(1, self.level + 1):
                for j in range(1, self.level + 1):
                    nc_x[1+k + 1, 1+i + 1] -= (
                        alpha[j]
                        * self.basismatrices.B[k, i, j]
                        / self.basismatrices.M[k, k]
                    )
        if self.dimension == 2:
            offset = self.level + 1
            b, h, alpha, beta, hinv = self.get_primitives()
            p = self.parameters
            um = alpha[0]
            vm = beta[0]
            for k in range(1, self.level + 1):
                nc_y[1+k + 1, 1+k + 1 + offset] += um
            for k in range(self.level + 1):
                for i in range(1, self.level + 1):
                    for j in range(1, self.level + 1):
                        nc_x[1+k + 1, 1+i + 1] -= (
                            alpha[j]
                            * self.basismatrices.B[k, i, j]
                            / self.basismatrices.M[k, k]
                        )
                        nc_y[1+k + 1, 1+i + 1 + offset] -= (
                            alpha[j]
                            * self.basismatrices.B[k, i, j]
                            / self.basismatrices.M[k, k]
                        )

            for k in range(1, self.level + 1):
                nc_x[1+k + 1 + offset, 1+k + 1] += vm
                nc_y[1+k + 1 + offset, 1+k + 1 + offset] += vm
            for k in range(self.level + 1):
                for i in range(1, self.level + 1):
                    for j in range(1, self.level + 1):
                        nc_x[1+k + 1 + offset, 1+i + 1] -= (
                            beta[j]
                            * self.basismatrices.B[k, i, j]
                            / self.basismatrices.M[k, k]
                        )
                        nc_y[1+k + 1 + offset, 1+i + 1 + offset] -= (
                            beta[j]
                            * self.basismatrices.B[k, i, j]
                            / self.basismatrices.M[k, k]
                        )
        return [-nc_x, -nc_y][:self.dimension]

    def eigenvalues(self):
        # we delete heigher order moments (level >= 2) for analytical eigenvalues
        offset = self.level + 1
        A = self.normal[0] * self.quasilinear_matrix()[0]
        for d in range(1, self.dimension):
            A += self.normal[d] * self.quasilinear_matrix()[d]
        b, h, alpha, beta, hinv = self.get_primitives()
        alpha_erase = alpha[1:] if self.level >= 2 else []
        beta_erase = beta[1:] if self.level >= 2 else []
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
        b, h, alpha, beta, hinv = self.get_primitives()
        p = self.parameters
        for k in range(1 + self.level):
            for i in range(1 + self.level):
                out[1+1 + k] += (
                    -p.nu
                    * alpha[i]
                    * hinv
                    * self.basismatrices.D[i, k]
                    / self.basismatrices.M[k, k]
                )
                if self.dimension == 2:
                    out[1+1 + k + offset] += (
                        -p.nu
                        * beta[i]
                        * hinv
                        * self.basismatrices.D[i, k]
                        / self.basismatrices.M[k, k]
                    )
        return out

    def slip_mod(self):
        assert "lamda" in vars(self.parameters)
        assert "rho" in vars(self.parameters)
        assert "c_slipmod" in vars(self.parameters)

        out = Matrix([0 for i in range(self.n_variables)])
        offset = self.level+1
        b, h, alpha, beta, hinv = self.get_primitives()
        p = self.parameters
        ub = 0
        vb = 0
        for i in range(1 + self.level):
            ub += alpha[i]
            vb += beta[i]
        for k in range(1, 1 + self.level):
            out[2 + k] += (
                -1.0 * p.c_slipmod / p.lamda / p.rho * ub / self.basismatrices.M[k, k]
            )
            if self.dimension == 2:
                out[2+offset+k] += (
                    -1.0 * p.c_slipmod / p.lamda / p.rho * vb / self.basismatrices.M[k, k]
                )
        return out


    def slip(self):
        assert "lamda" in vars(self.parameters)
        assert "rho" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        offset = self.level + 1
        b, h, alpha, beta, hinv = self.get_primitives()
        p = self.parameters
        for k in range(1 + self.level):
            for i in range(1 + self.level):
                out[1+1 + k] += (
                    -1.0 / p.lamda / p.rho * alpha[i] / self.basismatrices.M[k, k]
                )
                if self.dimension == 2:
                    out[1+1 + k + offset] += (
                        -1.0 / p.lamda / p.rho * beta[i] / self.basismatrices.M[k, k]
                    )
        return out

    def chezy(self):
        assert "C" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        offset = self.level + 1
        b, h, alpha, beta, hinv = self.get_primitives()
        p = self.parameters
        tmp = 0
        for i in range(1 + self.level):
            for j in range(1 + self.level):
                tmp += alpha[i] * alpha[j]  + beta[i] * beta[j]
        sqrt = sympy.sqrt(tmp)
        for k in range(1 + self.level):
            for l in range(1 + self.level):
                out[1 + k] += (
                    -1.0 / (p.C**2 * self.basismatrices.M[k, k]) * alpha[l] * sqrt
                )
                if self.dimension == 2:
                    out[1 + k + offset] += (
                        -1.0 / (p.C**2 * self.basismatrices.M[k, k]) * beta[l] * sqrt
                    )
        return out
    
    def gravity(self):
        out = Matrix([0 for i in range(self.n_variables)])
        out[2] = -self.parameters.g * self.parameters.ex * self.variables[0]
        if self.dimension == 2:
            offset = self.level + 1
            out[2 + offset] = -self.parameters.g * self.parameters.ey * self.variables[0]
        return out
    
    def newtonian_turbulent_algebraic(self):
        assert "nu" in vars(self.parameters)
        assert "l_bl" in vars(self.parameters)
        assert "l_turb" in vars(self.parameters)
        assert "kappa" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        offset = self.level + 1
        
        b, h, a, b, hinv = self.get_primitives()
        
        p = self.parameters
        dU_dx = a[0] / (p.l_turb * h)
        abs_dU_dx = sympy.Piecewise((dU_dx, dU_dx >=0), (-dU_dx, True))
        for k in range(1 + self.level):
            out[1 + k] += (
                -(p.nu + p.kappa * sympy.sqrt(p.nu * abs_dU_dx) * p.l_bl * ( 1-p.l_bl)) * dU_dx * self.basismatrices.phib[k] * hinv
            )
            for i in range(1 + self.level):
                out[1 + k] += (
                    -p.nu * hinv
                    * a[i]
                    * self.basismatrices.D[i, k]
                )
                out[1 + k] += (
                    -p.kappa * sympy.sqrt(p.nu * abs_dU_dx) * hinv
                    * a[i]
                    * (self.basismatrices.Dxi[i, k] - self.basismatrices.Dxi2[i, k])
                )
        if self.dimension == 2:
            dV_dy = b[0] / (p.l_turb * h)
            abs_dV_dy = sympy.Piecewise((dV_dy, dV_dy >=0), (-dV_dy, True))
            for k in range(1 + self.level):
                out[1 + k + offset] += (
                    -(p.nu + p.kappa * sympy.sqrt(p.nu * abs_dV_dy) * p.l_bl * ( 1-p.l_bl)) * dV_dy * self.basismatrices.phib[k] *hinv
                )
                for i in range(1 + self.level):
                    out[1 + k + offset] += (
                        -p.nu
                        / h
                        * b[i]
                        * self.basismatrices.D[i, k]
                    )
                    out[1 + k + offset] += (
                    -p.kappa * sympy.sqrt(p.nu * abs_dV_dy) * hinv
                    * b[i]
                    * (self.basismatrices.Dxi[i, k] - self.basismatrices.Dxi2[i, k])
                )

        return out

@define(frozen=True, slots=True, kw_only=True)
class ShallowMomentsTopoNumerical(ShallowMomentsTopo):
    ref_model: Model = field(init=False)
    
    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        object.__setattr__(self, "ref_model", ShallowMomentsTopo(level=self.level, dimension=self.dimension, boundary_conditions=self.boundary_conditions))

    def flux(self):
        return [self.substitute_precomputed_denominator(f, self.variables[1], self.aux_variables.hinv) for f in self.ref_model.flux()]      
    
    def nonconservative_matrix(self):
        return [self.substitute_precomputed_denominator(f, self.variables[1], self.aux_variables.hinv) for f in self.ref_model.nonconservative_matrix()]  
    
    def quasilinear_matrix(self):
        return [self.substitute_precomputed_denominator(f, self.variables[1], self.aux_variables.hinv) for f in self.ref_model.quasilinear_matrix()]    
    
    def source(self):
        return self.substitute_precomputed_denominator(self.ref_model.source(), self.variables[1], self.aux_variables.hinv)
    
    def source_implicit(self):
        return self.substitute_precomputed_denominator(self.ref_model.source_implicit(), self.variables[1], self.aux_variables.hinv)
    
    def residual(self):
        return self.substitute_precomputed_denominator(self.ref_model.residual(), self.variables[1], self.aux_variables.hinv)
    
    def left_eigenvectors(self):
        return self.substitute_precomputed_denominator(self.ref_model.left_eigenvectors(), self.variables[1], self.aux_variables.hinv)
    
    def right_eigenvectors(self):
        return self.substitute_precomputed_denominator(self.ref_model.right_eigenvectors(), self.variables[1], self.aux_variables.hinv)

    def eigenvalues(self):
        h = self.variables[1]
        evs = self.substitute_precomputed_denominator(self.ref_model.eigenvalues(), self.variables[1], self.aux_variables.hinv)
        for i in range(self.n_variables):
            evs[i] = Piecewise((evs[i], h > 1e-8), (0, True))
        return evs
