"""
Models of the foregrounds
"""
import attr
import numpy as np
from cached_property import cached_property

from .mcmc_framework.likelihood import Parameter, Component


class Foreground(Component):
    """Base class for all foreground models, don't use this directly!"""
    def __init__(self, freqs, nuc=75.0, **kwargs):
        self.freqs = freqs
        self.nuc = nuc  # MHz

        super().__init__(**kwargs)

    @cached_property
    def f(self):
        return self.freqs / self.nuc

    def __call__(self, dct, ctx):
        ctx[f'{self.__class__.__name__}_spectrum'] = self.model(**dct)


class PhysicalHills(Foreground):
    """
    Eq 6. from Hills et al.
    """
    Te = Parameter(1000, min=0, max=5000, latex=r"T_e [K]")
    b0 = Parameter(1750, min=0, max=1e5, latex=r"b_0 [K]")
    b1 = Parameter(0, latex=r"b_1 [K]")
    b2 = Parameter(0, latex=r"b_2 [K]")
    b3 = Parameter(0, latex=r"b_3 [K]")
    ion_spec_index = Parameter(-2, min=-3, max=-1, latex=r"\alpha_{\rm ion}")

    def model(self, Te, **p):
        b = [p[f"b{i}"] for i in range(4)]
        alpha = p['ion_spec_index']

        x = np.exp(-b[3] * self.f ** alpha)
        return b[0] * self.f ** (b[1] + b[2] * np.log(self.f) - 2.5) * x + Te * (1 - x)


class PhysicalSmallIonDepth(Foreground):
    """
    Eq. 7 from Hills et al.
    """
    b0 = Parameter(1750, min=0, max=1e5, latex=r"b_0 [K]")
    b1 = Parameter(0, latex=r"b_1 [K]")
    b2 = Parameter(0, latex=r"b_2 [K]")
    b3 = Parameter(0, latex=r"b_3 [K]")
    b4 = Parameter(0, latex=r"b_4 [K]")
    ion_spec_index = Parameter(-2, min=-3, max=-1, latex=r"\alpha_{\rm ion}")

    def model(self, **p):
        b = [p[f"b{i}"] for i in range(5)]
        alpha = p['ion_spec_index']

        x = np.exp(-b[3] * self.f ** alpha)
        return b[0] * self.f ** (b[1] + b[2] * np.log(self.f) - 2.5) * x + b[4] / self.f ** 2

    # Possible derived quantities
    def Te(self, dct, ctx):
        """Approximate value of Te in the small-ion-depth limit"""
        return dct['b4'] / dct['b3']


class PhysicalLin(Foreground):
    """
    Eq. 8 from Hills et al.
    """
    p0 = Parameter(0, latex=r"p_0")
    p1 = Parameter(0, latex=r"p_1")
    p2 = Parameter(0, latex=r"p_2")
    p3 = Parameter(0, latex=r"p_3")
    p4 = Parameter(0, latex=r"p_4")

    def model(self, p0, p1, p2, p3, p4):
        return self.f ** -2.5 * (
            p0 + np.log(self.f) * (p1 + p2 * np.log(self.f))) + p3 * self.f ** -4.5 + p4 * self.f ** -2

    # Possible derived parameters
    def b0(self, dct, ctx):
        """The corresponding b0 from PhysicalHills"""
        return dct['p0']

    def b1(self, dct, ctx):
        """The corresponding b1 from PhysicalHills"""
        return dct['p1'] / dct['p0']

    def b2(self, dct, ctx):
        """The corresponding b2 from PhysicalHills"""
        return dct['p2'] / dct['p0'] - self.b1(dct, ctx) ** 2 / 2

    def b3(self, dct, ctx):
        """The corresponding b3 from PhysicalHills"""
        return -dct['p3'] / dct['p0']

    def b4(self, dct, ctx):
        """The corresponding b4 from PhysicalHills"""
        return dct['p4']


class LinLog(Foreground):
    def __new__(cls, n=5, *args, **kwargs):

        # First create the parameters.
        for i in range(n):
            setattr(cls, f"p{i}", Parameter(0, latex=r"p_{}".format(i)))

        obj = super(LinLog, cls).__new__(cls)

        return obj

    def __init__(self, n=5, *args, **kwargs):
        # Need to add n to signature to take it out of the call to __init__
        print(args)
        print(kwargs)
        print(n)
        super().__init__(*args, **kwargs)

    def model(self, **p):
        logf = np.log(self.f)
        terms = []
        for pp in p:
            i = int(pp[1:])
            terms.append(p[pp] * logf ** i)

        return self.f ** -2.5 * np.sum(terms, axis=0)


class LinPoly(LinLog):
    def model(self, **p):
        """
        Eq. 10 from Hills et al.
        """
        terms = []
        for pp in p:
            i = int(pp[1:])
            terms.append(p[pp] * self.f ** (i - 2.5))

        return np.sum(terms, axis=0)
