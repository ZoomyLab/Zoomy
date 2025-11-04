import numpy as np
from numpy.polynomial.legendre import leggauss
from functools import partial


def zero():
    def nc_flux(Qi, Qauxi, Qj, Qauxj, parameters, normal, model):
        return np.zeros_like(Qi), False

    return nc_flux

def segmentpath(integration_order=3, scheme='rusanov'):
    # compute integral of NC-Matrix int NC(Q(s)) ds for segment path Q(s) = Ql + (Qr-Ql)*s for s = [0,1]
    samples, weights = leggauss(integration_order)
    # shift from [-1, 1] to [0,1]
    samples = 0.5 * (samples + 1)
    weights *= 0.5

    def priceC(
        Qi, Qj, Qauxi, Qauxj, parameters, normal, svA, svB, vol_face, dt, model
    ):
        dim = normal.shape[0]
        n_variables = Qi.shape[0]
        n_cells = Qi.shape[1]

        def B(s):
            out = np.zeros((n_variables, n_variables, n_cells), dtype=float)
            tmp = np.zeros_like(out)
            for d in range(dim):
                tmp = model.quasilinear_matrix[d](
                    Qi + s * (Qj - Qi), Qauxi + s * (Qauxj - Qauxi), parameters
                )
                out = out + tmp * normal[d]
                # out[:,:,:] += tmp * normal[d]
            return out

        Bint = np.zeros((n_variables, n_variables, n_cells))
        for w, s in zip(weights, samples):
            Bint += w * B(s)

        Bint_sq = np.einsum("ij..., jk...->ik...", Bint, Bint)
        I = np.zeros_like(Bint)
        for i in range(n_variables):
            # I[i, i, :] = 1.
            I = I.at[i, i, :].set(1.0)
            
        Am = (
            0.5 * Bint
            - (svA * svB) / (svA + svB) * 1.0 / (dt * vol_face) * I
            - 1 / 4 * (dt * vol_face) / (svA + svB) * Bint_sq
        )
        # Am = 0.5* Bint - np.einsum('..., ij...->ij...', (svA * svB)/(svA + svB) * 1./(dt * vol_face) ,I)  - 1/4 * np.einsum('..., ij...->ij...', (dt * vol_face)/(svA + svB) , Bint_sq)

        return np.einsum("ij..., j...->i...", Am, (Qj - Qi)), False


    def rusanov(
        Qi, Qj, Qauxi, Qauxj, parameters, normal, svA, svB, vol_face, dt, model
    ):
        dim = normal.shape[0]
        n_variables = Qi.shape[0]
        n_cells = Qi.shape[1]

        def B(s):
            # out = np.zeros((n_variables, n_variables, n_cells), dtype=float)
            # tmp = np.zeros_like(out)
            # for d in range(dim):
            #     tmp = model.quasilinear_matrix[d](
            #         Qi + s * (Qj - Qi), Qauxi + s * (Qauxj - Qauxi), parameters
            #     )
            #     out = out + tmp * normal[d]
            out = np.einsum("ijd..., d...->ij...", model.quasilinear_matrix(
                Qi + s * (Qj - Qi), Qauxi + s * (Qauxj - Qauxi), parameters
            ), normal)
            return out

        Am = np.zeros((n_variables, n_variables, n_cells))
        for w, s in zip(weights, samples):
            Am += w * B(s)

        I = np.zeros_like(Am)
        for i in range(n_variables):
            I[i, i, :] = 1.0

        ev_i = model.eigenvalues(Qi, Qauxi, parameters, normal)
        ev_j = model.eigenvalues(Qj, Qauxj, parameters, normal)
        sM = np.maximum(np.abs(ev_i).max(axis=0), np.abs(ev_j).max(axis=0))

        return np.einsum("ij..., j...->i...", 0.5 * Am + 0.5 * sM * I, (Qj - Qi)), np.einsum("ij..., j...->i...", 0.5 * Am - 0.5 * sM * I, (Qj - Qi)), False

    if scheme == 'rusanov':
        return rusanov
    elif scheme=='priceC':
        return priceC



