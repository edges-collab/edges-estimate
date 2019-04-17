# -*- coding: utf-8 -*-

"""Console script for edges_estimate."""
import sys
import click
from .mcmc_framework.util import load_likelihood_from_yaml
from .mcmc import run_mcmc
from os import path
import yaml
import getdist
from matplotlib import pyplot as plt

@click.command()
@click.argument("yaml_file", type=click.Path(exists=True, dir_okay=False))
@click.option("-s", "--sampler", default='polychord', type=click.Choice(["polychord", "emcee"]))
@click.option("-f", "--sampler_yaml", default=None, type=click.Path(dir_okay=False))
@click.option('--plot/--no-plot', default=True)
def main(yaml_file, sampler, sampler_yaml, plot):
    """Console script for edges_estimate."""
    likelihood = load_likelihood_from_yaml(yaml_file)

    if sampler_yaml is not None:
        with open(sampler_yaml, 'rb') as f:
            sampler_kw = yaml.load(f)
    else:
        sampler_kw = {}

    if "file_root" not in sampler_kw:
        sampler_kw['file_root'] = path.basename(yaml_file)

    sampler = run_mcmc(
        likelihood,
        sampler=sampler,
        sampler_kw=sampler_kw
    )

    if plot:
        posterior = sampler.posterior
        g = getdist.plots.getSubplotPlotter()
        g.triangle_plot(posterior, shaded=True)
        plt.savefig(path.join(sampler_kw.get("base_dir"), sampler_kw.get("file_root")) + "_corner.pdf")

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
