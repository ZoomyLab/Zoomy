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

from sympy.integrals.quadrature import gauss_legendre


    

@define(frozen=True, slots=True, kw_only=True)
class SMET(Model):
    dimension: int = 2
    level: int
    variables: Union[list, int] = field(init=False)
    positive_variables: Union[List[int], Dict[str, int], None] = attr.ib(default=attr.Factory(lambda: [1]))    
    aux_variables: Union[list, int] = field(default=0)
    basisfunctions: Union[Basisfunction, type[Basisfunction]] = field(default=Legendre_shifted)
    basismatrices: Basismatrices = field(init=False)
    order_numerical_integration: int = 4

    _default_parameters: dict = field(
        init=False,
        factory=lambda: {"g": 9.81, "ex": 0.0, "ey": 0.0, "ez": 1.0, "eps_low_water": 1e-6, "rho": 1000., 'nu': 1e-6, 'kappa': 0.41},
    )

    def __attrs_post_init__(self):
        object.__setattr__(self, "variables", ((self.level+1)*self.dimension)+2)
        object.__setattr__(self, "aux_variables", 2*((self.level+1)*self.dimension+2)+2+self.aux_variables)
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

    def get_gradient(self):
        offset = self.level + 1
        z = sympy.Symbol('z')
        phi = [self.basisfunctions.eval(k, z) for k in range(self.level+1)]
        
        
        if self.dimension == 1:
            grad_b = [self.aux_variables[0]]
            grad_h = [self.aux_variables[1]]
            dalpha_dx = self.aux_variables[2:2+offset]
            dU_dx = sum([dalpha_dx[k] * phi[k] for k in range(self.level+1)])
            grad_alpha = [dalpha_dx]
            grad_beta = [0 for i in range(offset)]
            grad_U = [[dU_dx, 0], [0, 0]]
        elif self.dimension == 2:
            gradient_offset = self.n_variables
            grad_b = [self.aux_variables[0], self.aux_variables[0+gradient_offset]]
            grad_h = [self.aux_variables[1], self.aux_variables[1+gradient_offset]]
            dalpha_dx = self.aux_variables[2:2+offset]
            dalpha_dy = self.aux_variables[2+gradient_offset: 2+gradient_offset+offset]
            dU_dx = sum([dalpha_dx[k] * phi[k] for k in range(self.level+1)])
            dU_dy = sum([dalpha_dy[k] * phi[k] for k in range(self.level+1)])
            grad_alpha = [dalpha_dx, dalpha_dy]
            
            dbeta_dx = self.aux_variables[2+offset: 2+2*offset]
            dbeta_dy = self.aux_variables[2+offset+gradient_offset: 2+2*offset+gradient_offset]
            dV_dx = sum([dbeta_dx[k] * phi[k] for k in range(self.level+1)])
            dV_dy = sum([dbeta_dy[k] * phi[k] for k in range(self.level+1)])
            grad_beta = [dbeta_dx, dbeta_dy]
            grad_U = [[dU_dx, dU_dy], [dV_dx, dV_dy]]
        else:
            assert False



        return grad_b, grad_h, grad_alpha, grad_beta, grad_U
        
    def get_ustar(self):
        gradient_offset = self.n_variables

        if self.dimension == 1:
            u_star = self.aux_variables[gradient_offset]
        elif self.dimension == 2:
            u_star = self.aux_variables[2*gradient_offset: 2*gradient_offset+2]
        return u_star

    def interpolate_3d(self):
        out = Matrix([0 for i in range(6)])
        level = self.level
        offset = level+1
        offset_aux = self.n_aux_variables
        x = self.position[0]
        y = self.position[1]
        z = self.position[2]
        
        b, h, alpha, beta, hinv = self.get_primitives()
        grad_b, grad_h, grad_alpha, grad_beta, grad_U = self.get_gradient()
        
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
        w_3d = - grad_h[0] * dot(alpha,psi) - h * dot(grad_alpha[0],psi) + dot(alpha, phi) * (z * grad_h[0] + grad_b[0])
        if self.dimension == 2:
            v_3d = self.basismatrices.basisfunctions.reconstruct_velocity_profile_at(beta, z)
            w_3d += - grad_h[1] * dot(beta,psi) - h * dot(grad_beta[1],psi) + dot(beta, phi) * (z * grad_h[1] + grad_b[1])

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
    

    def Sij(self):
        out = sympy.zeros(self.dimension, self.dimension)

        grad_b, grad_h, grad_alpha, grad_beta, grad_U = self.get_gradient()
        for d1 in range(self.dimension):
            for d2 in range(self.dimension):
                out[d1, d2] = 0.5 * (grad_U[d1][d2] + grad_U[d2][d1])

        return out
    
    def abs_Sij(self):
        Sij = self.Sij()
        out = 0
        for i in range(self.dimension):
            for j in range(self.dimension):
                out += Sij[i,j]**2
        out = sympy.sqrt(2 * out)
        return out
    
    def Szj(self):
        out = sympy.zeros(self.dimension)

        b, h, alpha, beta, hinv = self.get_primitives()
        dphi_dz = self.basisfunctions.get_diff_basis()
        
        for k in range(self.level+1):
            out[0] += alpha[k] * dphi_dz[k]
        if self.dimension == 2:
            for k in range(self.level+1):
                out[1] += beta[k] * dphi_dz[k]
        return out
    
    def source(self):
        out = Matrix([0 for i in range(self.n_variables)])
        out += self.mixing_length_vertical()
        
        return out
    
    
    def dflux(self):
        return self.smagorinsky_dflux()
    
    def smagorinsky_dflux(self):
        """
        diffusive flux due to the Smagorinsky model
        """
        assert "rho" in vars(self.parameters)
        assert "Cs" in vars(self.parameters)
        assert "rho" in vars(self.parameters)

        dflux = [sympy.zeros(self.n_variables, 1) for d in range(self.dimension)]
        z = sympy.Symbol('z')
        level = self.level
        offset = level+1
        phi = [self.basisfunctions.eval(k, z) for k in range(level+1)]
        rho = self.parameters.rho
        dX = self.distance
        Cs = self.parameters.Cs
        
        xi, wi = gauss_legendre(self.order_numerical_integration, 8)
        xi = [0.5 * (x + 1) for x in xi]
        wi = [0.5 * w for w in wi]
        b, h, alpha, beta, hinv = self.get_primitives()

        abs_Sij = self.abs_Sij()
        Sij = self.Sij()
        
        for i in range(len(xi)):
            zi = xi[i]
            for k in range(level+1):
                for d1 in range(self.dimension):
                    for d2 in range(self.dimension):
                        dflux[d1][2 + d2 * offset + k, 0] += wi[i] * h*  phi[k].subs(z, zi) * 2 * rho * (Cs * dX)**2 * abs_Sij.subs(z, zi) * Sij[d1, d2].subs(z, zi)
        for k in range(level+1):
            for d in range(self.dimension):
                dflux[d][2+k, 0] /= self.basismatrices.M[k, k]
        return [dflux[d] for d in range(self.dimension)]
    


        
    def smagorinsky_source(self):
        out = Matrix([0 for i in range(self.n_variables)])
        z = sympy.Symbol('z')
        level = self.level
        offset = level+1
        phi = self.basisfunctions.basis
        dphi_dz = self.basisfunctions.get_diff_basis()
        rho = self.parameters.rho
        dX = self.distance
        Cs = self.parameters.Cs
        
        xi, wi = gauss_legendre(self.order_numerical_integration, 8)
        xi = [0.5 * (x + 1) for x in xi]
        wi = [0.5 * w for w in wi]
        b, h, alpha, beta, hinv = self.get_primitives()
        grad_b, grad_h, grad_alpha, grad_beta, grad_U = self.get_gradient()
        sigma = [grad_b[0] + z * grad_h[0], grad_b[1] + z * grad_h[1]]
        
        
        abs_Sij = self.abs_Sij()
        Sij = self.Sij()
        
        u_star = self.get_ustar()
        taub = [rho * u_star[d]**2 for d in range(self.dimension)]
        for d in range(self.dimension):
            for k in range(1 + self.level):
                out[2+k + d*offset] += sigma[d].subs(z, 0) * taub[d] * phi[k].subs(z, 0)
            
        
        for i in range(len(xi)):
            zi = xi[i]
            for k in range(level+1):
                for d1 in range(self.dimension):
                    for d2 in range(self.dimension):
                        out[2 + d2 * offset + k, 0] += wi[i] * sigma[d1].subs(z, zi) *  2 * rho * (Cs * dX)**2 * abs_Sij.subs(z, zi) * Sij[d1, d2].subs(z, zi)  * dphi_dz[k].subs(z, zi)

        
    
    def mixing_length_numerical_term(self):
        assert "kappa" in vars(self.parameters)
        out = 0
        b, h, alpha, beta, hinv = self.get_primitives()
        p = self.parameters
        z = sympy.Symbol('z')
        phi = [self.basisfunctions.eval(k, z) for k in range(self.level+1)]
        u = sum([alpha[k] * phi[k] for k in range(self.level+1)])
        v = sum([beta[k] * phi[k] for k in range(self.level+1)] )
        U = [u,v][:self.dimension]
        
        for d1 in range(self.dimension):
            for d2 in range(self.dimension):
                out += U[d1] * U[d2]
                
        eps = 1e-10
        out += eps
        out = sympy.sqrt(out)
        out *= (p.kappa * z * (1-z))**2
        return out


    def mixing_length_vertical(self):
        assert "rho" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        offset = self.level + 1
        b, h, alpha, beta, hinv = self.get_primitives()
        p = self.parameters
        z = sympy.Symbol('z')
        phi = [self.basisfunctions.eval(k, z) for k in range(self.level+1)]
        dphi_dz = self.basisfunctions.get_diff_basis()
        
        Szj = self.Szj()


        u_star = self.get_ustar()
        taub = [p.rho * u_star[d]**2 for d in range(self.dimension)]
        for d in range(self.dimension):
            for k in range(1 + self.level):
                out[2+k + d*offset] += taub[d] * phi[k].subs(z, 0)
            
        xi, wi = gauss_legendre(self.order_numerical_integration, 8)
        xi = [0.5 * (x + 1) for x in xi]
        wi = [0.5 * w for w in wi]
        
        for d in range(self.dimension):
            for i in range(len(xi)):
                for k in range(self.level+1):
                    out[2+k+d*offset] -= wi[i] * self.mixing_length_numerical_term().subs(z, xi[i]) * Szj[d].subs(z, xi[i]) * dphi_dz[k].subs(z, xi[i])
        return out
    


@define(frozen=True, slots=True, kw_only=True)
class SMETNum(SMET):
    ref_model: Model = field(init=False)
    
    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        object.__setattr__(self, "ref_model", SMET(level=self.level, dimension=self.dimension, boundary_conditions=self.boundary_conditions))

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
