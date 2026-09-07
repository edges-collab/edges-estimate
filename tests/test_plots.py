import numpy as np
import pytest

from edges_estimate import plots as p


@pytest.fixture(scope="session")
def models():
    return {key: {"model": np.random.normal(size=(20, 50))} for key in ["one", "two"]}


@pytest.fixture(scope="session")
def freqs():
    return {key: np.random.normal(size=50) for key in ["one", "two"]}


@pytest.fixture(scope="session")
def temp():
    return {key: np.random.normal(size=50) for key in ["one", "two"]}


def test_make_residual_plot_shaded(models, freqs, temp):
    p.make_residual_plot_shaded(models, freqs, temp)
