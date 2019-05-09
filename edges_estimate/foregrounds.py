"""
Models of the foregrounds
"""
import numpy as np
from cached_property import cached_property

from yabf import Parameter, Component


class Foreground(Component):
    """Base class for all foreground models, don't use this directly!"""
    def __init__(self, freqs, nuc=75.0, **kwargs):
        self.freqs = freqs
        self.nuc = nuc  # MHz

        super().__init__(**kwargs)

        self.provides = [f"{self.name}_spectrum"]

    @cached_property
    def f(self):
        return self.freqs / self.nuc

    def calculate(self, ctx, **params):
        return self.model(**params)


class _PhysicalBase(Foreground):
    base_parameters = [
        Parameter("b0", 1750, min=0, max=1e5, latex=r"b_0 [K]"),
        Parameter("b1", 0, latex=r"b_1 [K]"),
        Parameter("b2", 0, latex=r"b_2 [K]"),
        Parameter("b3", 0, latex=r"b_3 [K]"),
        Parameter("ion_spec_index", -2, min=-3, max=-1, latex=r"\alpha_{\rm ion}"),
    ]

    def model(self, **p):
        b = [p[f"b{i}"] for i in range(4)]
        alpha = p['ion_spec_index']

        x = np.exp(-b[3] * self.f ** alpha)
        return b[0] * self.f ** (b[1] + b[2] * np.log(self.f) - 2.5) * x, x


class PhysicalHills(_PhysicalBase):
    """
    Eq 6. from Hills et al.
    """
    base_parameters = _PhysicalBase.base_parameters + [
        Parameter("Te", 1000, min=0, max=5000, latex=r"T_e [K]"),
    ]

    def model(self, Te, **p):
        first_term, x = super().model(**p)
        return first_term + Te*(1 - x)


class PhysicalSmallIonDepth(_PhysicalBase):
    """
    Eq. 7 from Hills et al.
    """
    base_parameters = _PhysicalBase.base_parameters + [Parameter("b4", 0, latex=r"b_4 [K]")]

    def model(self, **p):
        first_term, x = super().model(p)
        b4 = p['b4']
        return first_term + b4 / self.f ** 2

    # Possible derived quantities
    def Te(self, ctx, **params):
        """Approximate value of Te in the small-ion-depth limit"""
        return params['b4'] / params['b3']


class PhysicalLin(Foreground):
    """
    Eq. 8 from Hills et al.
    """
    base_parameters = [Parameter("p0", 1750, min=0, latex=r"p_0")] + \
                 [Parameter(f'p{i}', 0, latex=r"p_{}".format(i)) for i in range(1, 5)]

    def model(self, **p):
        p = [p[f"p{i}"] for i in range(5)]

        return self.f ** -2.5 * (
            p[0] + np.log(self.f) * (p[1] + p[2] * np.log(self.f))) + p[3] * self.f ** -4.5 + p[4] * self.f ** -2

    # Possible derived parameters
    def b0(self, ctx, **p):
        """The corresponding b0 from PhysicalHills"""
        return p['p0']

    def b1(self, ctx, **p):
        """The corresponding b1 from PhysicalHills"""
        return p['p1'] / p['p0']

    def b2(self, ctx, **p):
        """The corresponding b2 from PhysicalHills"""
        return p['p2'] / p['p0'] - self.b1(ctx, **p) ** 2 / 2

    def b3(self, ctx, **p):
        """The corresponding b3 from PhysicalHills"""
        return -p['p3'] / p['p0']

    def b4(self, ctx, **p):
        """The corresponding b4 from PhysicalHills"""
        return p['p4']


class LinLog(Foreground):
    def __new__(cls, n=5, *args, **kwargs):

        # First create the parameters.
        p = []
        for i in range(n):
            p.append(Parameter(f"p{i}", 0, latex=r"p_{}".format(i)))
        cls.base_parameters= tuple(p)

        obj = super(LinLog, cls).__new__(cls)

        return obj

    def __init__(self, n=5, *args, **kwargs):
        # Need to add n to signature to take it out of the call to __init__
        self.poly_order = n

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


class MultiplicativeBias(Component):
    base_parameters = [
        Parameter("bias", 1, min=0, latex=r"b_\cross")
    ]

    def calculate(self, ctx, **params):
        for key, val in ctx.item():
            if key.endswith("_spectrum"):
                ctx[key] = val * params['mult_bias']


class AdditiveBias(Component):
    base_parameters = [
        Parameter("bias", 1, min=0, latex=r'b_+')
    ]

    def calculate(self, ctx, **params):
        for key, val in ctx.item():
            if key.endswith("_spectrum"):
                ctx[key] = val * params['mult_bias']


