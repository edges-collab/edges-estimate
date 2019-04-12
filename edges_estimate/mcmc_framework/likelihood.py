"""
Framework for MCMC likelihoods (and parameters).
"""
import warnings
from collections import OrderedDict
from copy import deepcopy

import attr
import numpy as np


@attr.s
class Parameter:
    """
    A fiducial parameter of a model. Used to *define* the model
    and its defaults. A Parameter of a model does not mean it *will* be
    constrained, but rather that it *can* be constrained. To be constrained,
    a :class:`Param` must be set on the instance at run-time.

    min/max in this class specify the total physically/logically allowable domain
    for the parameter. This can be reduced via the specification of :class:`Param`
    at run-time.
    """
    default = attr.ib()
    min = attr.ib(-np.inf)
    max = attr.ib(np.inf)
    latex = attr.ib(None)

    prior = None
    generate_ref = None
    name = None
    _value = None

    @property
    def ltx(self):
        if self.latex is not None:
            return self.latex
        else:
            return self.name

    @property
    def value(self):
        if self._value is None:
            return self.default
        else:
            return self._value

    def update_base(self, p):
        self.min = max(self.min, p.min)
        self.max = min(self.max, p.max)
        self.prior = p.prior
        self.generate_ref = p.generate_ref
        p.latex = self.latex

    def logprior(self, val):
        if not (self.min <= val <= self.max):
            return -np.inf

        if self.prior is None:
            return 0
        else:
            try:
                return self.prior.logpdf(val)
            except AttributeError:
                return self.prior(val)


@attr.s
class Param:
    """
    A run-time parameter (i.e. one that is to be constrained).
    """
    name = attr.ib()
    ref = attr.ib(None)
    min = attr.ib(-np.inf, type=float)
    max = attr.ib(np.inf, type=float)
    prior = attr.ib(None)

    def generate_ref(self):
        if self.ref is None:
            # Use prior
            if self.prior is None:
                ref = np.random.uniform(self.min, self.max)
            else:
                try:
                    ref = self.prior.rvs()
                except AttributeError:
                    raise NotImplementedError("callable priors not yet implemented")
        else:
            try:
                ref = self.ref.rvs()
            except AttributeError:
                try:
                    ref = self.ref()
                except TypeError:
                    ref = self.ref

        if not self.min <= ref <= self.max:
            raise ValueError(f"param {self.name} produced a reference value outside its domain.")

        return ref


class Component:
    """
    A component of a likelihood. These are mainly for re-usability, so they
    can be mixed and matched inside likelihoods.
    """
    def __init__(self, **kwargs):
        self.all_parameters = {}

        for cls in self.__class__.mro():
            for name, p in cls.__dict__.items():
                if isinstance(p, Parameter):
                    p.name = name
                    self.all_parameters[name] = deepcopy(p)

                if name in kwargs:
                    self.all_parameters[name]._value = kwargs.pop(name)

        if kwargs:
            raise ValueError(f"The following kwargs were not expected: {kwargs}")

    def __call__(self, dct, ctx):
        """
        Every component should take a dct of parameter values and return
        a dict of values to be used.
        """
        pass


class Likelihood(Component):
    _use_prior = True

    def __init__(self, data=None, params=None, derived=None, components=None, **kwargs):
        super().__init__(**kwargs)

        self.components = components
        for component in components:
            assert isinstance(component, Component)
            self.all_parameters.update(component.all_parameters)

        self.derived = derived or []

        if data is None:
            if hasattr(self, "mock"):
                self.data = self.mock()

            else:
                warnings.warn("You have not passed any data... logp will not work!")
                self.data = None
        else:
            self.data = data

        self.parameters = OrderedDict()
        for p in params:
            # Update the base parameters
            try:
                self.all_parameters[p.name].update_base(p)
            except KeyError:
                raise ValueError(f"You submitted a parameter {p.name} which does not exist in the likelihood")

            # Make a list of updateable parameters
            self.parameters[p.name] = self.all_parameters[p.name]

    def _get_params(self, p=None):
        # First make base dict
        params = {p.name: p.value for p in self.all_parameters.values()}

        if p is not None:
            for i, name in enumerate(self.parameters):
                params[name] = p[i]
        return params

    def _fill_dct(self, dct):
        fiducial = self._get_params()
        fiducial.update(dct)
        return fiducial

    def derived_quantities(self, model, dct, ctx):
        dquants = []
        for d in self.derived:
            if type(d) == str:
                try:
                    # Append local quantity
                    dquants.append(getattr(self, d)(model, dct, ctx))
                except AttributeError:
                    # Search for quantity in components
                    for c in self.components:
                        if hasattr(c, d):
                            dquants.append(getattr(c, d)(dct, ctx))
            elif callable(d):
                dquants.append(d(model, dct, ctx))
            else:
                raise ValueError("{} is not a valid entry for derived".format(d))

        return dquants

    def _prior(self, dct):
        return np.sum([p.logprior(dct[k]) for k, p in self.parameters.items()])

    def model(self, ctx, fill_in_dct=True, **dct):
        if fill_in_dct:
            dct = self._fill_dct(dct)

        return self._model(ctx, **dct)

    def _get_current_ctx(self, dct):
        # Fill up a context object
        ctx = {}
        for c in self.components:
            c({k: v for k, v in dct.items() if k in c.all_parameters}, ctx)

        return ctx

    def __call__(self, p):
        dct = self._get_params(p)
        ctx = self._get_current_ctx(dct)

        model = self.model(ctx, fill_in_dct=False,  **dct)
        prior = self._prior(dct) if self._use_prior else 0

        logp = self.logp(model=model, **dct)
        return prior + logp, self.derived_quantities(model, dct, ctx)
