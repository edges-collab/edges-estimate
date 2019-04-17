from edges_estimate.likelihoods import MultiComponentSpectrumChi2, Param
from edges_estimate.mcmc import run_mcmc
from edges_estimate import foregrounds as fg

import numpy as np
import matplotlib.pyplot as plt

import getdist
from getdist import plots


if __name__=="__main__":
    import sys

    data = np.genfromtxt("figure1_plotdata.csv", skip_header=1, delimiter=',')
    freq, tsky = data[data[:, 1] > 0, 0], data[data[:, 1] > 0, 2]

    print(sys.argv)

    if len(sys.argv) > 2:
        fg.LinPoly._n = int(sys.argv[-1])

    likelihood = MultiComponentSpectrumChi2(**likelihoods[sys.argv[1]])

    print(likelihood.__class__.__name__)

    sampler = run_mcmc(
        likelihood,
        sampler="polychord",
        sampler_kw = {
            "read_resume":True,
            "base_dir":'HillsChains',
            "file_root":sys.argv[1],
            "nlive": 2048
        },
    )

    posterior = sampler.posterior
    g = getdist.plots.getSubplotPlotter()
    g.triangle_plot(posterior, shaded=True)
    plt.savefig(sys.argv[1]+"_corner.pdf")
