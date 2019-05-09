# -*- coding: utf-8 -*-
import numpy as np

from yabf import Parameter, Component


def phenom_model(freqs, A, tau, w, nu0):
    """ really bad inverse gaussian thing."""
    B = 4 * (freqs - nu0) ** 2 / w ** 2 * np.log(-1 / tau * np.log((1 + np.exp(-tau)) / 2))
    return -A * (1 - np.exp(-tau * np.exp(B))) / (1 - np.exp(-tau))


class AbsorptionProfile(Component):
    provides = ['eor_spectrum']

    base_parameters = [
        Parameter("A", 0.5, min=0, latex=r"a_{21}"),
        Parameter("tau", 7, min=0, latex=r"\tau"),
        Parameter("w", 17.0, min=0),
        Parameter("nu0", 75, min=0, latex=r"\nu_0"),
    ]

    def __init__(self, freqs, **kwargs):
        self.freqs = freqs
        super().__init__(**kwargs)

    def calculate(self, ctx, **params):
        return phenom_model(self.freqs, **params)
