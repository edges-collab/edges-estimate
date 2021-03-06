"""Provides extra routines for fitting that are not in yabf."""
from edges_cal.modelling import Model
from yabf import Component
import numpy as np
from scipy.optimize import minimize
from scipy import stats


class SemiLinearFit:
    def __init__(self, fg: Model, eor: Component, spectrum, sigma):
        """Perform a quick fit to data with a sum of linear and non-linear models.

        Useful for fitting foregrounds and EoR at the same time, where the EoR model is
        not linear, but the foreground model is.
        """

        self.fg = fg
        self.eor = eor
        self.spectrum = spectrum
        self.sigma = sigma

    def get_eor(self, p):
        return self.eor(params=p)['eor_spectrum']

    def fg_fit(self, p):
        eor = self.get_eor(p)
        resid = self.spectrum - eor
        return self.fg.fit(
            ydata=resid,
            weights=1 / self.sigma ** 2 if hasattr(self.sigma, '__len__') else None
        )

    def fg_params(self, p):
        return self.fg_fit(p).model_parameters

    def get_resid(self, p):
        return self.fg_fit(p).residual

    def neg_lk(self, p):
        resid = self.get_resid(p)
        if hasattr(self.sigma, 'ndim') and self.sigma.ndim == 2:
            norm_obj = stats.multivariate_normal(mean=np.zeros_like(resid), cov=self.sigma)
        else:
            norm_obj = stats.norm(loc=0, scale=self.sigma)

        return -np.sum(norm_obj.logpdf(resid))

    def __call__(self):
        return minimize(
            self.neg_lk,
            x0=np.array([apar.fiducial for apar in self.eor.child_active_params]),
            bounds=[(apar.min, apar.max) for apar in self.eor.child_active_params],
        )
