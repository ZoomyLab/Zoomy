import numpy as np
from numpy.polynomial.legendre import leggauss

def zero():
    def nc_flux(Qi, Qauxi, Qj, Qauxj, parameters, normal, model):
        return np.zeros_like(Qi), False

    return nc_flux

def segmentpath(integration_order=3):
    # compute integral of NC-Matrix int NC(Q(s)) ds for segment path Q(s) = Ql + (Qr-Ql)*s for s = [0,1]
    samples, weights = leggauss(integration_order)
    # shift from [-1, 1] to [0,1]
    samples = 0.5 * (samples + 1)
    weights *= 0.5

    def nc_flux(Qi, Qj, Qauxi, Qauxj, parameters, normal, model):
        dim = normal.shape[0]
        n_fields = Qi.shape[0]
        def B(s):
            out = np.zeros((n_fields, n_fields), dtype=float)
            tmp = np.zeros_like(out)
            for d in range(dim):
                tmp = model.nonconservative_matrix[d](Qi + s * (Qj - Qi), Qauxi + s * (Qauxj - Qauxi), parameters) 
                out = tmp * normal[d]
            return out

        Bint = np.zeros((n_fields, n_fields))
        for w, s in zip(weights, samples):
            Bint += w * B(s)
        # The multiplication with (Qj-Qi) the part dPsi/ds out of the integral above. But since I use a segment path, dPsi/ds is (Qj-Qi)=const
        # and taken out of the integral

        return -0.5 * Bint@(Qj-Qi), False

    return nc_flux