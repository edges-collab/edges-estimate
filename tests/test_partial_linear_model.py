import pytest

import attr
import matplotlib.pyplot as mpl
import numpy as np
from edges_cal.modelling import Polynomial
from yabf import Component, Parameter, run_map
from yabf.samplers.polychord import polychord

from edges_estimate.likelihoods import PartialLinearModel
from edges_estimate.plots import get_evidence


@attr.s(kw_only=True)
class Sin(Component):
    provides = ["power"]

    base_parameters = [
        Parameter("t", fiducial=0, min=-100, max=100, latex=r"t"),
    ]

    x: np.ndarray = attr.ib(kw_only=True, eq=attr.cmp_using(eq=np.array_equal))

    def calculate(self, ctx, **params):
        return params["t"] * np.sin(self.x * np.pi)


@attr.s(kw_only=True)
class Scale(Component):
    provides = ["scale"]

    base_parameters = [
        Parameter("scale", fiducial=1, min=0.1, max=100, latex=r"t"),
    ]

    x: np.ndarray = attr.ib(kw_only=True, eq=attr.cmp_using(eq=np.array_equal))

    def calculate(self, ctx, **params):
        # can't be scale*(anything) because then there's degeneracies with the other models.
        return 3 + params["scale"] * np.sin(self.x**2)


@pytest.fixture(scope="module")
def simple_plm():

    x = np.linspace(-1, 1, 100)

    linear_model = Polynomial(parameters=[1, 3]).at(x=x)
    sinx = Sin(x=x, params={"t": {"fiducial": 10, "min": -10, "max": 30}})

    y = linear_model() + sinx()["power"] + np.random.normal(scale=0.01, size=len(x))

    return PartialLinearModel(
        components=sinx,
        linear_model=linear_model,
        data={"data": y, "data_variance": 0.01},
        data_func=lambda ctx, data: data["data"] - ctx["power"],
    )


@pytest.fixture(scope="module")
def scaled_plm():

    x = np.linspace(-1, 1, 100)

    linear_model = Polynomial(parameters=[1, 3]).at(x=x)
    sinx = Sin(x=x, params={"t": {"fiducial": 10, "min": -10, "max": 30}})
    scale = Scale(x=x, params={"scale": {"fiducial": 1, "min": 0.1, "max": 30}})

    y = (
        linear_model() + sinx()["power"] + np.random.normal(scale=0.01, size=len(x))
    ) / scale()["scale"]

    def vfunc(ctx, data):
        v = ctx["scale"] ** 2 * data["data_variance"]
        return v

    return PartialLinearModel(
        components=(sinx, scale),
        linear_model=linear_model,
        data={
            "data": y,
            "data_variance": np.ones_like(y) * (0.01 / scale()["scale"]) ** 2,
        },
        data_func=lambda ctx, data: ctx["scale"] * data["data"] - ctx["power"],
        variance_func=vfunc,
    )


@pytest.fixture(scope="module")
def large_plm(simple_plm: PartialLinearModel):
    linear_model = Polynomial(parameters=[1, 3, 0]).at(x=simple_plm.linear_model.x)
    return attr.evolve(simple_plm, linear_model=linear_model)


@pytest.fixture(scope="module")
def scaled_large_plm(scaled_plm: PartialLinearModel):
    linear_model = Polynomial(parameters=[1, 3, 0]).at(x=scaled_plm.linear_model.x)
    return attr.evolve(scaled_plm, linear_model=linear_model)


def test_correct_result_simple(simple_plm):
    out = run_map(simple_plm)
    assert out.success
    np.testing.assert_allclose(out.x[0], 10, atol=0.02)

    fit, data, var = simple_plm.reduce_model(params=out.x)
    np.testing.assert_allclose(fit.residual, 0, atol=0.05)  # 5 sigma.


def test_correct_result_scaled(scaled_plm, plt):
    out = run_map(scaled_plm, x0=[10, 1])
    print(out)

    if plt == mpl:
        ctx = scaled_plm.get_ctx(params=[10, 1])
        fit, data, var = scaled_plm.reduce_model(ctx=ctx)

        ctx2 = scaled_plm.get_ctx(params=out.x)
        fit2, data2, var2 = scaled_plm.reduce_model(ctx=ctx2)
        fig, ax = plt.subplots(3, 2, sharex=True)
        x = scaled_plm.linear_model.x

        ax[0, 0].plot(scaled_plm.linear_model.x, scaled_plm.data["data"], label="data")
        ax[0, 0].plot(
            scaled_plm.linear_model.x,
            (fit.evaluate() + ctx["power"]) / ctx["scale"],
            label="model",
        )
        ax[0, 0].plot(
            scaled_plm.linear_model.x,
            (fit.evaluate()) / ctx["scale"],
            label="linear_model",
        )
        ax[0, 0].plot(
            scaled_plm.linear_model.x, (ctx["power"]) / ctx["scale"], label="sinx"
        )

        ax[1, 0].plot(scaled_plm.linear_model.x, data)
        ax[1, 0].plot(scaled_plm.linear_model.x, fit.evaluate())

        ax[2, 0].plot(scaled_plm.linear_model.x, fit.residual)
        ax[2, 0].fill_between(
            x, -np.sqrt(var) * np.ones_like(x), np.sqrt(var) * np.ones_like(x)
        )

        ax[0, 0].legend()

        ax[0, 1].plot(scaled_plm.linear_model.x, scaled_plm.data["data"], label="data")
        ax[0, 1].plot(
            scaled_plm.linear_model.x,
            (fit2.evaluate() + ctx2["power"]) / ctx2["scale"],
            label="model",
        )
        ax[0, 1].plot(
            scaled_plm.linear_model.x,
            (fit2.evaluate()) / ctx2["scale"],
            label="linear_model",
        )
        ax[0, 1].plot(
            scaled_plm.linear_model.x, (ctx2["power"]) / ctx2["scale"], label="sinx"
        )

        ax[1, 1].plot(scaled_plm.linear_model.x, data2)
        ax[1, 1].plot(scaled_plm.linear_model.x, fit2.evaluate())

        ax[2, 1].plot(scaled_plm.linear_model.x, fit2.residual)
        ax[2, 1].fill_between(
            x, -np.sqrt(var2) * np.ones_like(x), np.sqrt(var2) * np.ones_like(x)
        )

    assert out.success
    np.testing.assert_allclose(out.x[0], 10, atol=0.1)
    np.testing.assert_allclose(out.x[1], 1, atol=0.02)

    fit, data, v = scaled_plm.reduce_model(params=out.x)
    print(5 * np.sqrt(v))
    assert np.allclose(fit.residual, 0, atol=5 * np.sqrt(v))  # 5 sigma.


def test_correct_result_larger_model(large_plm):
    out = run_map(large_plm)

    np.testing.assert_allclose(out.x[0], 10, atol=0.05)
    fit, data, var = large_plm.reduce_model(params=out.x)
    np.testing.assert_allclose(fit.residual, 0, atol=0.05)  # 5 sigma.
    np.testing.assert_allclose(fit.model_parameters[-1], 0, rtol=0, atol=0.1)


def test_correct_result_larger_model_scaled(scaled_large_plm):
    out = run_map(scaled_large_plm)

    np.testing.assert_allclose(out.x[0], 10, atol=0.05)
    fit, data, var = scaled_large_plm.reduce_model(params=out.x)
    np.testing.assert_allclose(fit.residual, 0, atol=0.05)  # 5 sigma.
    np.testing.assert_allclose(fit.model_parameters[-1], 0, rtol=0, atol=0.1)


@pytest.mark.slow
def test_polychord_evidence(simple_plm, large_plm, tmp_path_factory):
    direc = tmp_path_factory

    poly = polychord(
        save_full_config=False,
        likelihood=simple_plm,
        output_dir=str(direc),
        output_prefix=f"{direc}/correct_model",
        sampler_kwargs={"nlive": 256, "precision_criterion": 0.1},
    )
    correct_samples = poly.sample()
    correct_lnz = get_evidence(correct_samples.root)

    poly = polychord(
        save_full_config=False,
        likelihood=large_plm,
        output_dir=str(direc),
        output_prefix=f"{direc}/larger_model",
        sampler_kwargs={"nlive": 256, "precision_criterion": 0.1},
    )
    large_samples = poly.sample()
    large_lnz = get_evidence(large_samples.root)
    assert large_lnz <= correct_lnz


@pytest.mark.slow
def test_polychord_evidence_scaled(scaled_plm, scaled_large_plm, tmp_path):
    direc = tmp_path / "cache"

    poly = polychord(
        save_full_config=False,
        likelihood=scaled_plm,
        output_dir=str(direc),
        output_prefix=f"{direc}/correct_scaled_model",
        sampler_kwargs={"nlive": 256, "precision_criterion": 0.1},
    )
    correct_samples = poly.sample()
    correct_lnz = get_evidence(correct_samples.root)

    poly = polychord(
        save_full_config=False,
        likelihood=scaled_large_plm,
        output_dir=str(direc),
        output_prefix=f"{direc}/larger_scaled_model",
        sampler_kwargs={"nlive": 256, "precision_criterion": 0.1},
    )
    large_samples = poly.sample()
    large_lnz = get_evidence(large_samples.root)
    assert large_lnz <= correct_lnz
