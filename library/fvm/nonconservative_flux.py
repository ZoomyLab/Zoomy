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

    # return nc_flux
    # return nc_flux_vectorized
    # return nc_flux_quasilinear_componentwise
    return nc_flux_quasilinear
