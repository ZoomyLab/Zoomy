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


from library.model.models.base import (
    register_sympy_attribute,
    eigenvalue_dict_to_matrix,
)
from library.model.models.base import Model
from library.model.models.basismatrices import Basismatrices
from library.model.models.basisfunctions import Legendre_shifted, Basisfunction


    

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
        factory=lambda: {"g": 9.81, "ex": 0.0, "ey": 0.0, "ez": 1.0, 'rho': 1000.0}
    )

    def __attrs_post_init__(self):
        object.__setattr__(self, "variables", ((self.level+1)*2)+2)
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
        hb = self.variables[2 + offset : 2 + offset + self.level + 1]
        alpha = [ha[i] * hinv for i in range(offset)]
        beta = [hb[i] * hinv for i in range(offset)]
        return [b, h, alpha, beta, hinv]


    def interpolate_3d(self):
        out = Matrix([0 for i in range(6)])
        level = self.level
        offset = level+1
        x = self.position[0]
        y = self.position[1]
        z = self.position[2]
        b, h, alpha, beta, hinv = self.get_primitives()
        # dalphadx = self.aux_variables[1:1+offset]
        # dbetady = self.aux_variables[1+offset:1+2*offset]
        assert "rho" in vars(self.parameters)
        assert "g" in vars(self.parameters)
        p = self.parameters
        u_3d = self.basismatrices.basisfunctions.reconstruct_velocity_profile_at(alpha, z)
        v_3d = self.basismatrices.basisfunctions.reconstruct_velocity_profile_at(beta, z)
        out[0] = b
        out[1] = h
        out[2] = u_3d
        out[3] = v_3d
        out[4] = 0
        out[5] = p.rho * p.g * h * (1-z)
        return out

    def flux(self):
        offset = self.level + 1
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
        return [flux_x, flux_y]

    def nonconservative_matrix(self):
        offset = self.level + 1
        nc_x = Matrix([[0 for i in range(self.n_variables)] for j in range(self.n_variables)])
        nc_y = Matrix([[0 for i in range(self.n_variables)] for j in range(self.n_variables)])
        b, h, alpha, beta, hinv = self.get_primitives()
        p = self.parameters
        um = alpha[0]
        vm = beta[0]
        nc_x[2, 0] += -p.ez * p.g * h
        nc_y[2+offset, 0] += -p.ez * p.g * h
        for k in range(1, self.level + 1):
            nc_x[1+k + 1, 1+k + 1] += um
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
        return [-nc_x, -nc_y]

    def eigenvalues(self):
        # we delete heigher order moments (level >= 2) for analytical eigenvalues
        offset = self.level + 1
        A = self.normal[0] * self.quasilinear_matrix()[0]
        for d in range(1, self.dimension):
            A += self.normal[d] * self.quasilinear_matrix()[d]
        b, h, alpha, beta, hinv = self.get_primitives()
        alpha_erase = alpha[2:] if self.level >= 2 else []
        beta_erase = beta[2:] if self.level >= 2 else []
        for alpha_i in alpha_erase:
            A = A.subs(alpha_i, 0)
        for beta_i in beta_erase:
            A = A.subs(beta_i, 0)
        return eigenvalue_dict_to_matrix(A.eigenvals())

    def source(self):
        out = Matrix([0 for i in range(self.n_variables)])
        return out
    
    def inclination(self):
        out = Matrix([0 for i in range(self.n_variables)])
        assert "ex" in vars(self.parameters)
        assert "ey" in vars(self.parameters)
        assert "g" in vars(self.parameters)
        offset = self.level + 1
        b, h, alpha, beta, hinv = self.get_primitives()
        p = self.parameters
        out[2] = p.g * p.ex * h
        out[2+offset] = p.g * p.ey * h
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
            out[1+1 + k] += (
                -1.0 * p.c_slipmod / p.lamda / p.rho * ub / self.basismatrices.M[k, k]
            )
            out[1+1+offset+k] += (
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
                out[2 + k] += (
                    -1.0 / (p.C**2 * self.basismatrices.M[k, k]) * alpha[l] * sqrt
                )
                out[2 + k + offset] += (
                    -1.0 / (p.C**2 * self.basismatrices.M[k, k]) * beta[l] * sqrt
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
    
    def interpolate_3d(self):
        return self.substitute_precomputed_denominator(self.ref_model.interpolate_3d(), self.variables[1], self.aux_variables.hinv)

    def eigenvalues(self):
        h = self.variables[1]
        evs = self.substitute_precomputed_denominator(self.ref_model.eigenvalues(), self.variables[1], self.aux_variables.hinv)
        # for i in range(self.n_variables):
        #     evs[i] = Piecewise((evs[i], h > 1e-2), (0, True))
        return evs
