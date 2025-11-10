import numpy as np
import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss
from functools import partial
import jax

class NonconservativeFlux:
    def get_flux_operator(self, model):
        pass  

class Zero(NonconservativeFlux):
    def get_flux_operator(self, model):
        @jax.jit
        def compute( 
            Qi,
            Qj,
            Qauxi,
            Qauxj,
            parameters,
            normal,
            Vi,
            Vj,
            Vij,
            dt
        ):
            return jnp.zeros_like(Qi), jnp.zeros_like(Qi)
        return compute

class Rusanov(NonconservativeFlux):
    def __init__(self, integration_order=3, identity_matrix=None, eps=-1, index_h=1):
        self.integration_order = integration_order
        samples, weights = leggauss(integration_order)
        # shift from [-1, 1] to [0,1]
        samples = 0.5 * (samples + 1)
        weights *= 0.5
        self.wi = jnp.array(weights)
        self.xi = jnp.array(samples)
        self.Id = identity_matrix if identity_matrix else lambda n: jnp.eye(n)
        self.eps = eps
        self.index_h = index_h
        
    def _get_A(self, model):
        def A(q, qaux, parameters, n):                      
            # q : (n_dof,)
            # evaluate the matrices A_d
            _A = model.quasilinear_matrix(q, qaux, parameters)
            return jnp.einsum('d,ijd->ij', n, _A)

        return A
    
    def _integrate_path(self, model):
        compute_A = self._get_A(model)
        def A_int(Qi, Qj,
                            Qauxi, Qauxj,
                            parameters,
                            normal):
            dQ     =  Qj    - Qi
            dQaux  =  Qauxj - Qauxi

            # ------------------------------------------------------
            # 1)   path integral   A_int = ∫₀¹ A(q(s),qaux(s),n) ds
            # ------------------------------------------------------
            q_path     = Qi[:, None]    + self.xi * dQ    [:, None]    # (n_dof , 3)
            qaux_path  = Qauxi[:, None] + self.xi * dQaux[:, None]     # (n_aux , 3)

            # evaluate A for the N quadrature points
            A_mats = jax.vmap(compute_A, in_axes=(1, 1, None, None))(q_path, qaux_path, parameters, normal)
            A_int  = jnp.tensordot(self.wi, A_mats, axes=1)           # (n_dof, n_dof)
            return A_int
        return A_int
    
    
    def _single(self, model):
        compute_path_integral = self._integrate_path(model)
        Id = self.Id(model.n_variables)
        def _rusanov(Qi, Qj,
                            Qauxi, Qauxj,
                            parameters,
                            normal,
                            Vi, Vj, Vij, dt):

            # -- very same wet/dry & wall checks as in the original ----------------
            # cond1 = (Qi[index_h] < eps) & (Qi[index_topography] > Qj[index_h] + Qj[index_topography])
            # cond2 = (Qj[index_h] < eps) & (Qj[index_topography] > Qi[index_h] + Qi[index_topography])
            # dry   = cond1 | cond2
            dry = (self.eps >=0) & (Qi[self.index_h] < self.eps) & (Qj[self.index_h] < self.eps)


            def _compute():                         # executed only if not dry
                dQ     =  Qj    - Qi
                dQaux  =  Qauxj - Qauxi

                A_int = compute_path_integral(Qi, Qj, Qauxi, Qauxj, parameters, normal)

                # ------------------------------------------------------
                # 2)   spectral radius along n
                # ------------------------------------------------------
                ev_i = model.eigenvalues(Qi, Qauxi, parameters, normal)  # (n_dof,)
                ev_j = model.eigenvalues(Qj, Qauxj, parameters, normal)
                sM   = jnp.max(jnp.maximum(jnp.abs(ev_i), jnp.abs(ev_j)))

                # ------------------------------------------------------
                # 3)   Rusanov fluctuation that leaves cell "i"
                # ------------------------------------------------------

                # Id = Id.at[index_h, index_topography].set(1.0)
                Am   = 0.5 * (A_int - sM * Id)
                Ap = 0.5 * (A_int + sM * Id)
                Dm = (Am @ dQ)
                Dp = (Ap @ dQ)

                return Dp, Dm
            
            def _dry():
                return jnp.zeros_like(Qi), jnp.zeros_like(Qi)

            return jax.lax.cond(dry,
                                _dry,
                                _compute)
        return _rusanov


    def get_flux_operator(self, model):
        compute_single = self._single(model)
        @partial(jax.named_call, name="NC_Flux")
        @jax.jit
        def compute( 
            Qi,
            Qj,
            Qauxi,
            Qauxj,
            parameters,
            normal,
            Vi,
            Vj,
            Vij,
            dt
        ):
            """
            Vectorised (vmap) Rusanov fluctuation.

            Shapes
            ------
            Qi, Qj            : (n_dof , N)        states for the two cells
            Qauxi, Qauxj      : (n_aux , N)
            parameters        : (n_param ,)
            normal            : (dim   , N)  or (dim,)   oriented outward for cell "i"
            Vi                : (N,)   or scalar         cell volume
            Vij               : (N,)   or scalar         face measure
            """

            # vmapping is along the "cell/face" axis (=1 for Q*, =0 for Vi,Vij)
            Dp, Dm = jax.vmap(
                compute_single,
                in_axes=(
                    1,
                    1,  # Qi , Qj
                    1,
                    1,  # Qauxi, Qauxj
                    None,  # parameters
                    1,  # normal
                    0,
                    0,
                    0,
                    None,  # dt
                ),
            )(Qi, Qj, Qauxi, Qauxj, parameters, normal, Vi, Vj, Vij, dt)
            return Dp.T, Dm.T
        return compute

class PriceC(Rusanov):
    
    def _single(self, model):
        compute_path_integral = self._integrate_path(model)
        def _priceC(Qi, Qj, Qauxi, Qauxj, parameters, normal, Vi, Vj, Vij, dt):
            """
            One face computation of the PriceC fluctuation terms.
            Shapes: Qi, Qj ∈ (n_dof,), Qauxi ∈ (n_aux,), etc.
            """
            Bint = compute_path_integral(Qi, Qj, Qauxi, Qauxj, parameters, normal)
            Bint_sq = jnp.einsum("ij,jk->ik", Bint, Bint)
            n_dof = Qi.shape[0]

            # Identity matrix
            Id = jnp.eye(n_dof, dtype=Qi.dtype)

            # Compute Am / Ap
            visc = (Vi * Vj) / (Vi + Vj)
            time_factor = (dt * Vij) / (Vi + Vj)
            Am = 0.5 * Bint - visc / (dt * Vij) * Id - 0.25 * time_factor * Bint_sq
            Ap = 0.5 * Bint + visc / (dt * Vij) * Id - 0.25 * time_factor * Bint_sq

            dQ = Qj - Qi
            Dm = Am @ dQ
            Dp = Ap @ dQ
            return Dp, Dm

        return _priceC
