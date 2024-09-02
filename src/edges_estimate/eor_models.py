"""21cm absorption feature models."""

import attr
import numpy as np
from yabf import Component, Parameter


def phenom_model(freqs, amp, tau, w, nu0):
    """Really bad inverse gaussian thing."""
    B = 4 * (freqs - nu0) ** 2 / w**2 * np.log(-1 / tau * np.log((1 + np.exp(-tau)) / 2))
    return -amp * (1 - np.exp(-tau * np.exp(B))) / (1 - np.exp(-tau))


@attr.s
class AbsorptionProfile(Component):
    provides = ["eor_spectrum"]

    base_parameters = [
        Parameter("amp", 0.5, min=0, latex=r"a_{21}"),
        Parameter("tau", 7, min=0, latex=r"\tau"),
        Parameter("w", 17.0, min=0),
        Parameter("nu0", 75, min=0, latex=r"\nu_0"),
    ]

    freqs: np.ndarray = attr.ib(kw_only=True, eq=attr.cmp_using(eq=np.array_equal))

    def calculate(self, ctx, **params):
        return phenom_model(self.freqs, **params)

    def spectrum(self, ctx, **params):
        return ctx["eor_spectrum"]
