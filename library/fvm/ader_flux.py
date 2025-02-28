import numpy as np
from numpy.polynomial.legendre import leggauss

def zero():
    def nc_flux(Qi, Qauxi, Qj, Qauxj, parameters, normal, model):
        return np.zeros_like(Qi), False

    return nc_flux

def quadrature(integration_order=3):
    # compute integral of NC-Matrix int NC(Q(s)) ds for segment path Q(s) = Ql + (Qr-Ql)*s for s = [0,1]
    samples, weights = leggauss(integration_order)
    # shift from [-1, 1] to [0,1]
    samples = 0.5 * (samples + 1)
    weights *= 0.5

    def flux(Q, gradQ, Qaux, parameters, volume, dt, model):
        dim = gradQ.shape[2]
        # compute u(t+dt/2)
        Qp = Q.copy()
        for d in range(dim):
            Qp -= dt/2 * np.einsum('ij...,j...->i...', model.quasilinear_matrix[d](Q, Qaux, parameters), gradQ[:, :, d])

        # compute ader_flux
        ader_flux = np.zeros_like(Q)
        for d in range(dim):
            ader_flux += dt*volume*np.einsum('ij...,j...->i...', model.quasilinear_matrix[d](Qp, Qaux, parameters), gradQ[:, :, d])

        return ader_flux, False

    return flux
