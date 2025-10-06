import numpy as np
import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss
from functools import partial
import jax


def zero():
    def nc_flux(Qi, Qauxi, Qj, Qauxj, parameters, normal, model):
        return np.zeros_like(Qi), False

    return nc_flux


def segmentpath_ssf(integration_order=3):
    """
    doi: 10.1016/j.jcp.2020.109457
    """

    def nc_flux(Qi, Qj, Qauxi, Qauxj, parameters, normal, model):
        dim = normal.shape[0]
        n_variables = Qi.shape[0]

        flux = model.flux
        dim = normal.shape[0]
        num_eq = Qi.shape[0]
        assert dim == 1
        Fi = flux[0](Qi, Qauxi, parameters)
        Fj = flux[0](Qj, Qauxj, parameters)

        EVi = model.eigenvalues(Qi, Qauxi, parameters, normal)
        EVj = model.eigenvalues(Qj, Qauxj, parameters, normal)
        assert not np.isnan(EVi).any()
        assert not np.isnan(EVj).any()
        SL = np.min([EVi[0], 0.5 * (EVi[0] + EVj[0])], axis=0)
        SR = np.min([EVj[5], 0.5 * (EVi[5] + EVj[5])], axis=0)

        QR = Qj
        QL = Qi
        FR = Fj
        FL = Fi

        Q_star = 1 / (SR - SL) * (SR * QR - SL * QL - (FR - FL))
        nc = model.nonconservative_matrix[0]
        BL = nc(0.5 * (QL + Q_star), 0.5 * (Qauxi + Qauxj), parameters)[:, 0] * (
            Q_star[0] - QL[0]
        )
        BR = nc(0.5 * (QR + Q_star), 0.5 * (Qauxi + Qauxj), parameters)[:, 0] * (
            QR[0] - Q_star[0]
        )
        NC_star = 1 / (SR - SL) * (-BL - BR)
        Q_star = Q_star + NC_star

        Dp = np.max(SL, 0) * (Q_star - QL) + np.max(SR, 0) * (QR - Q_star)

        return Dp, False

    return nc_flux


def segmentpath_1d(integration_order=3):
    # compute integral of NC-Matrix int NC(Q(s)) ds for segment path Q(s) = Ql + (Qr-Ql)*s for s = [0,1]
    samples, weights = leggauss(integration_order)
    # shift from [-1, 1] to [0,1]
    samples = 0.5 * (samples + 1)
    weights *= 0.5

    def nc_flux(Qi, Qj, Qauxi, Qauxj, parameters, normal, model):
        dim = normal.shape[0]
        assert dim == 1
        n_variables = Qi.shape[0]

        def B(s):
            out = np.zeros((n_variables, n_variables), dtype=float)
            tmp = np.zeros_like(out)
            for d in range(dim):
                tmp = model.nonconservative_matrix[0](
                    Qi + s * (Qj - Qi), Qauxi + s * (Qauxj - Qauxi), parameters
                )
            return out

        Bint = np.zeros((n_variables, n_variables))
        for w, s in zip(weights, samples):
            Bint += w * B(s)
        return 0.5 * np.einsum("ij, j->i", Bint, (Qj - Qi)), False

    return nc_flux

@partial(jax.named_call, name="NC_Flux")
def segmentpath(integration_order=3):
    # compute integral of NC-Matrix int NC(Q(s)) ds for segment path Q(s) = Ql + (Qr-Ql)*s for s = [0,1]
    samples, weights = leggauss(integration_order)
    # shift from [-1, 1] to [0,1]
    samples = 0.5 * (samples + 1)
    weights *= 0.5

    def nc_flux(Qi, Qj, Qauxi, Qauxj, parameters, normal, model):
        dim = normal.shape[0]
        n_variables = Qi.shape[0]

        # n_cells = Qi.shape[1]
        def B(s):
            # out = np.zeros((n_variables, n_variables, n_cells), dtype=float)
            out = np.zeros((n_variables, n_variables), dtype=float)
            tmp = np.zeros_like(out)
            for d in range(dim):
                tmp = model.nonconservative_matrix[d](
                    Qi + s * (Qj - Qi), Qauxi + s * (Qauxj - Qauxi), parameters
                )
                out = tmp * normal[d]
            return out

        # Bint = np.zeros((n_variables, n_variables, n_cells))
        Bint = np.zeros((n_variables, n_variables))
        for w, s in zip(weights, samples):
            Bint += w * B(s)
        # The multiplication with (Qj-Qi) the part dPsi/ds out of the integral above. But since I use a segment path, dPsi/ds is (Qj-Qi)=const
        # and taken out of the integral

        return -0.5 * Bint @ (Qj - Qi), False
        # return -0.5 * np.einsum('ij..., j...->i...', Bint, (Qj-Qi)), False

    def nc_flux_quasilinear(
        Qi, Qj, Qauxi, Qauxj, parameters, normal, svA, svB, vol_face, dt, model
    ):
        dim = normal.shape[0]
        n_variables = Qi.shape[0]
        n_cells = Qi.shape[1]

        def B(s):
            out = jnp.zeros((n_variables, n_variables, n_cells), dtype=float)
            tmp = jnp.zeros_like(out)
            for d in range(dim):
                tmp = model.quasilinear_matrix[d](
                    Qi + s * (Qj - Qi), Qauxi + s * (Qauxj - Qauxi), parameters
                )
                out = out + tmp * normal[d]
                # out[:,:,:] += tmp * normal[d]
            return out

        Bint = jnp.zeros((n_variables, n_variables, n_cells))
        for w, s in zip(weights, samples):
            Bint += w * B(s)

        Bint_sq = jnp.einsum("ij..., jk...->ik...", Bint, Bint)
        I = jnp.zeros_like(Bint)
        for i in range(n_variables):
            # I[i, i, :] = 1.
            I = I.at[i, i, :].set(1.0)
            
        Am = (
            0.5 * Bint
            - (svA * svB) / (svA + svB) * 1.0 / (dt * vol_face) * I
            - 1 / 4 * (dt * vol_face) / (svA + svB) * Bint_sq
        )
        # Am = 0.5* Bint - jnp.einsum('..., ij...->ij...', (svA * svB)/(svA + svB) * 1./(dt * vol_face) ,I)  - 1/4 * jnp.einsum('..., ij...->ij...', (dt * vol_face)/(svA + svB) , Bint_sq)

        return jnp.einsum("ij..., j...->i...", Am, (Qj - Qi)), False

    def nc_flux_quasilinear_componentwise(
        Qi, Qj, Qauxi, Qauxj, parameters, normal, svA, svB, vol_face, dt, model
    ):
        dim = normal.shape[0]
        n_variables = Qi.shape[0]
        n_cells = 1

        def B(s):
            out = np.zeros((n_variables, n_variables, n_cells), dtype=float)
            # out = np.zeros((n_variables, n_variables), dtype=float)
            tmp = np.zeros_like(out)
            for d in range(dim):
                tmp = model.quasilinear_matrix[d](
                    Qi + s * (Qj - Qi), Qauxi + s * (Qauxj - Qauxi), parameters
                )
                out[:, :, 0] += tmp * normal[d]
            return out

        Bint = np.zeros((n_variables, n_variables, n_cells))
        for w, s in zip(weights, samples):
            Bint += w * B(s)

        Bint_sq = np.einsum("ij..., jk...->ik...", Bint, Bint)
        I = np.zeros_like(Bint)
        _I = np.zeros_like(Bint)
        for i in range(n_variables):
            I[i, i, :] = 1.0
        # for d in range(dim):
        #     I += normal[d] * _I

        # Am = 0.5* Bint - 2*np.einsum('..., ij...->ij...', (svA * svB)/(svA + svB) * 2/(dt * vol_face), I)
        # Am = 0.5* Bint - np.einsum('..., ij...->ij...', (svA * svB)/(svA + svB) * 1./(dt * vol_face), I)  -1/4 * (dt * vol_face)/(svA + svB) * Bint_sq
        Am = (
            0.5 * Bint
            - (svA * svB) / (svA + svB) * 1.0 / (dt * vol_face) * I
            - 1 / 4 * (dt * vol_face) / (svA + svB) * Bint_sq
        )
        # if normal[0] > 0:
        #     Am = 1/4 * (2.0* Bint - vol_face/dt * I  - (dt / vol_face) * Bint_sq)
        #     return np.einsum('ij..., j...->i...', Am, (Qj-Qi))[:, 0], False
        # else:
        #     Ap = 1/4 * (2.0* Bint +  vol_face/dt * I  + (dt / vol_face) * Bint_sq)
        #     return np.einsum('ij..., j...->i...', Ap, (Qi-Qj))[:, 0], False

        return np.einsum("ij..., j...->i...", Am, (Qj - Qi))[:, 0], False

    # def nc_flux_quasilinear(Qi, Qj, Qauxi, Qauxj, parameters, normal, svA, svB, vol_face, dt, model):
    #     dim = normal.shape[0]
    #     n_variables = Qi.shape[0]

    #     n_cells = Qi.shape[1]
    #     def B(s):
    #         out = np.zeros((n_variables, n_variables, n_cells), dtype=float)
    #         # out = np.zeros((n_variables, n_variables), dtype=float)
    #         tmp = np.zeros_like(out)
    #         for d in range(dim):
    #             tmp = model.quasilinear_matrix[d](Qi + s * (Qj - Qi), Qauxi + s * (Qauxj - Qauxi), parameters)
    #             out = tmp * normal[d]
    #         return out

    #     Bint = np.zeros((n_variables, n_variables, n_cells))
    #     for w, s in zip(weights, samples):
    #         Bint += w * B(s)

    #     Bint_sq = np.einsum('ij..., jk...->ik...', Bint, Bint)
    #     I = np.empty_like(Bint)
    #     for i in range(n_variables):
    #         I[i, i, :] = 1.

    #     # Am = 0.5* Bint - 2*np.einsum('..., ij...->ij...', (svA * svB)/(svA + svB) * 2/(dt * vol_face), I)
    #     # Am = 0.5* Bint - np.einsum('..., ij...->ij...', (svA * svB)/(svA + svB) * 1./(dt * vol_face), I)  -1/4 * (dt * vol_face)/(svA + svB) * Bint_sq
    #     Am = 0.5* Bint - (svA * svB)/(svA + svB) * 0.5/(dt * vol_face) * I  -1/4 * (dt * vol_face)/(svA + svB) * Bint_sq

    #     return np.einsum('ij..., j...->i...', Am, (Qj-Qi)), False

    def nc_flux_vectorized(Qi, Qj, Qauxi, Qauxj, parameters, normal, model):
        dim = normal.shape[0]
        n_variables = Qi.shape[0]

        n_cells = Qi.shape[1]

        def B(s):
            out = np.zeros((n_variables, n_variables, n_cells), dtype=float)
            # out = np.zeros((n_variables, n_variables), dtype=float)
            tmp = np.zeros_like(out)
            for d in range(dim):
                tmp = model.nonconservative_matrix[d](
                    Qi + s * (Qj - Qi), Qauxi + s * (Qauxj - Qauxi), parameters
                )
                out = tmp * normal[d]
            return out

        Bint = np.zeros((n_variables, n_variables, n_cells))
        # Bint = np.zeros((n_variables, n_variables))
        for w, s in zip(weights, samples):
            Bint += w * B(s)
        # The multiplication with (Qj-Qi) the part dPsi/ds out of the integral above. But since I use a segment path, dPsi/ds is (Qj-Qi)=const
        # and taken out of the integral

        # return -0.5 * Bint@(Qj-Qi), False
        return -0.5 * np.einsum("ij..., j...->i...", Bint, (Qj - Qi)), False
    

        # @jax.jit
    def nc_flux_jax(Qi, Qj, Qauxi, Qauxj, parameters, normal, Vi, Vj, Vij, dt, model):
        
        # ---------------------------------------------------------------------------
        # 3-point Gauss integration on [0,1]
        # ---------------------------------------------------------------------------

        wi = jnp.array(weights)
        xi = jnp.array(samples)
        
        index_h = 1
        index_topography = 0
        
        def _get_A(model):
            n_dir = len(model.quasilinear_matrix)   # 1, 2 or 3

            def A(q, qaux, n):                      # q : (n_dof,)
                # evaluate the matrices A_d
                mats = [model.quasilinear_matrix[d](q, qaux, parameters)  for d in range(n_dir)]
                mats = jnp.stack(mats, axis=0)      # (n_dir, n_dof, n_dof)
                return jnp.einsum('d,dij->ij', n, mats)

            return A
        
        get_A = _get_A(model)
    
        # ---------------------------------------------------------------------------
        # ONE face-and-cell contribution (not yet batched) ---------------------------
        # ---------------------------------------------------------------------------
        eps = 1e-4
        def _rusanov_single(Qi, Qj,
                            Qauxi, Qauxj,
                            normal,
                            Vi, Vj, Vij):

            n_dof = Qi.shape[0]

            # -- very same wet/dry & wall checks as in the original ----------------
            # cond1 = (Qi[index_h] < eps) & (Qi[index_topography] > Qj[index_h] + Qj[index_topography])
            # cond2 = (Qj[index_h] < eps) & (Qj[index_topography] > Qi[index_h] + Qi[index_topography])
            # dry   = cond1 | cond2
            dry = (Qi[index_h] < eps) & (Qj[index_h] < eps)

            


            def _compute():                         # executed only if not dry
                dQ     =  Qj    - Qi
                dQaux  =  Qauxj - Qauxi

                # ------------------------------------------------------
                # 1)   path integral   A_int = ∫₀¹ A(q(s),qaux(s),n) ds
                # ------------------------------------------------------
                q_path     = Qi[:, None]    + xi * dQ    [:, None]    # (n_dof , 3)
                qaux_path  = Qauxi[:, None] + xi * dQaux[:, None]     # (n_aux , 3)

                # evaluate A for the three quadrature points
                A_mats = jax.vmap(get_A, in_axes=(1, 1, None))(q_path, qaux_path, normal)
                A_int  = jnp.tensordot(wi, A_mats, axes=1)           # (n_dof, n_dof)

                # ------------------------------------------------------
                # 2)   spectral radius along n
                # ------------------------------------------------------
                ev_i = model.eigenvalues(Qi, Qauxi, parameters, normal)  # (n_dof,)
                ev_j = model.eigenvalues(Qj, Qauxj, parameters, normal)
                sM   = jnp.max(jnp.maximum(jnp.abs(ev_i), jnp.abs(ev_j)))

                # ------------------------------------------------------
                # 3)   Rusanov fluctuation that leaves cell "i"
                # ------------------------------------------------------
                Id = jnp.eye(n_dof, dtype=Qi.dtype)
                # Id = Id.at[index_topography, index_topography].set(0.0)
                # Id = Id.at[index_h, index_h].set(0.0)

                # Id = Id.at[index_h, index_topography].set(1.0)
                Am   = 0.5 * (A_int - sM * Id)
                flux = (Am @ dQ) * (Vij / Vi)
                return flux

            # return jax.lax.cond(dry,
            #                     lambda: jnp.zeros_like(Qi),
            #                     _compute)
            return jax.lax.cond(dry,
                    _compute,
                    _compute)


        # ---------------------------------------------------------------------------
        # batched wrapper ------------------------------------------------------------
        # ---------------------------------------------------------------------------
        @jax.jit
        def rusanov_batched(Qi, Qj,
                            Qauxi, Qauxj,
                            normal,
                            Vi, Vj, Vij,
                            ):
            """
            Vectorised (vmap) Rusanov fluctuation.

            Shapes
            ------
            Qi, Qj            : (n_dof , N)        states for the two cells
            Qauxi, Qauxj      : (n_aux , N)
            normal            : (dim   , N)  or (dim,)   oriented outward for cell "i"
            Vi                : (N,)   or scalar         cell volume
            Vij               : (N,)   or scalar         face measure
            """

            # vmapping is along the "cell/face" axis (=1 for Q*, =0 for Vi,Vij)
            flux = jax.vmap(_rusanov_single,
                            in_axes=(1, 1,   # Qi , Qj
                                    1, 1,   # Qauxi, Qauxj
                                    1,      # normal
                                    0, 0, 0))(Qi, Qj, Qauxi, Qauxj, normal, Vi, Vj, Vi)
            return flux.T

        flux = rusanov_batched(Qi, Qj, Qauxi, Qauxj, normal, Vi, Vj, Vij)
        return flux,  False

    ### OLD
    # return nc_flux
    # return nc_flux_vectorized
    # return nc_flux_quasilinear_componentwise
    ### JAX
    # return nc_flux_quasilinear
    return nc_flux_jax
