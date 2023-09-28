import numpy as np
import os
import sys
from numpy.polynomial.legendre import leggauss


from library.solver.baseclass import BaseYaml  # nopep8
from library.solver.model import *  # nopep8

main_dir = os.getenv("SMPYTHON")


class NonConservativeTerms(BaseYaml):
    yaml_tag = "!NonConservativeTerms"

    def set_default_parameters(self):
        self.scheme = "segmentpath"

    def evaluate(self, model):
        return getattr(sys.modules[__name__], self.scheme)(model)


def segmentpath(model):
    # compute integral of NC-Matrix int NC(Q(s)) ds for segment path Q(s) = Ql + (Qr-Ql)*s for s = [0,1]
    samples, weights = leggauss(3)
    # shift from [-1, 1] to [0,1]
    samples = 0.5 * (samples + 1)
    weights *= 0.5

    def nc_flux(Qi, Qj, nij, **kwargs):
        B = lambda s: model.nonconservative_matrix(Qi + s * (Qj - Qi), **kwargs)

        # somehow this is incorrect
        # Bint_test, err = fixed_quad(B, 0, 1, n=5)
        # Bint =  B(0.5)
        # project Bint onto normal
        Bint = np.zeros((Qi.shape[0], Qi.shape[0], model.dimension, Qi.shape[1]))
        # Bint = weightskkl[0] * B(samples[0])
        for w, s in zip(weights, samples):
            Bint += w * B(s)
        Bint = np.einsum("ijk..., k...->ij...", Bint, nij)
        # The multiplication with (Qj-Qi) the part dPsi/ds out of the integral above. But since I use a segment path, dPsi/ds is (Qj-Qi)=const
        # and taken out of the integral
        # if "ShallowMoments" in model.yaml_tag or "ShallowWater" in model.yaml_tag:
        #     if Qi[0] <= 10 ** (-12) and Qj[0] + Qj[-1] < Qi[-1]:
        #         Bint[:, :] = 0.0
        #     if Qj[0] <= 10 ** (-12) and Qi[0] + Qi[-1] < Qj[-1]:
        #         Bint[:, :] = 0.0
        if "ShallowMoments" in model.yaml_tag or "ShallowWater" in model.yaml_tag:
            set_zero = np.logical_and((Qi[0] <= 10 ** (-12)) ,(Qj[0] + Qj[-1] < Qi[-1]))
            set_zero = np.logical_or(set_zero, np.logical_and((Qi[0] <= 10 ** (-12)) ,(Qj[0] + Qj[-1] < Qi[-1])))
            Bint[:,:, set_zero] = 0.0
        return -0.5 * np.einsum("ij..., j...->i...", Bint, (Qj - Qi))
        # return -0.5 * np.einsum("ij..., i...->j...", Bint, (Qj - Qi))

        # Qav = 0.5 * (Qi + Qj)
        # Bint = model.nonconservative_matrix(Qav, **kwargs)
        # Bint = np.einsum("ijk..., k...->ij...", Bint, nij)
        # return -0.5 * np.einsum("ij..., j...->i...", Bint, (Qj - Qi))

    return nc_flux


def mean(model):
    def nc_flux(Ql, Qr, nij, **kwargs):
        B = model.nonconservative_matrix(0.5 * (Ql + Qr), **kwargs)
        Bn = np.einsum("ijk..., k...->ij...", B, nij)
        return -0.5 * np.einsum("ij..., j...->i...", Bn, (Qr - Ql))

    return nc_flux


def swe_1d(model):
    assert model.yaml_tag == "!ShallowWaterWithBottom"

    # compute integral of NC-Matrix int NC(Q(s)) ds for segment path Q(s) = Ql + (Qr-Ql)*s for s = [0,1]
    def nc_flux(Ql, Qr, nij, **kwargs):
        Qav = 0.5 * (Ql + Qr)
        B = model.nonconservative_matrix(Qav)
        Bproj = np.einsum("ijk..., k...->ij...", B, nij)
        return -0.5 * np.einsum("ij..., j...->i...", Bproj, (Qr - Ql))

    return nc_flux


def none(model):
    # compute integral of NC-Matrix int NC(Q(s)) ds for segment path Q(s) = Ql + (Qr-Ql)*s for s = [0,1]
    def nc_flux(Ql, Qr, nij, **kwargs):
        return np.zeros_like(Ql)

    return nc_flux
