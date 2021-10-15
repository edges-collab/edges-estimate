import numpy as np
from getdist import loadMCSamples
from matplotlib import pyplot as plt
from tqdm import tqdm
from yabf import LikelihoodContainer, load_likelihood_from_yaml


def get_evidence(mcsamplesr):

    """Read the Bayesian evidence of the Polychord run
    ------------------------------------------------

    Parameters:
    ------------
    mcsamplesr: str
    The root file path of the run without the extension
    """

    with open(mcsamplesr + ".stats") as f:
        for line in f:
            if line.startswith("log(Z)"):
                evidence = float(line.split("=")[1].lstrip().split(" ")[0])
                break
    return evidence


def get_models_from_mcsamples(
    mcsamples, lk_names, lk=None, extras=None, n=None, progress=True
):
    if type(mcsamples) == str:
        mcsamples = loadMCSamples(mcsamples)

    if n is not None:
        samples = mcsamples.samples[
            np.random.choice(mcsamples.samples.shape[0], size=n)
        ]
    else:
        samples = mcsamples.samples

    out = {}
    if lk is None:
        lk = load_likelihood_from_yaml(mcsamples.rootname + "_config.yml")

    top_level = True
    if isinstance(lk, LikelihoodContainer):
        top_level = False

    if top_level and len(lk_names) > 1:
        raise ValueError("you have specified more lk_names than actually exist!")

    for params in tqdm(samples, disable=not progress):
        ctx = lk.get_ctx(params=params)
        model = lk.reduce_model(ctx=ctx, params=params)

        if top_level:
            ctx = {lk_names[0]: ctx}
            model = {lk_names[0]: model}

        for lk_name in lk_names:

            if lk_name not in out:
                out[lk_name] = {}

            if "model" not in out[lk_name]:
                out[lk_name]["model"] = []

            out[lk_name]["model"].append(model[lk_name])

            for extra in extras or []:
                if extra in ctx[lk_name]:
                    out[lk_name][extra] = ctx[lk_name][extra]

    return out


def make_residual_plot_shaded(models, freqs=None, temps=None, labels=None, color=None):
    labels= labels or {}
    for i, (key, freq) in enumerate(freqs.items()):
        model = models[key]["model"]
        temp = temps[key]
        label = labels.get(key,key)

        q = np.quantile(model, q=(0.04, 0.16, 0.5, 0.84, 0.96), axis=0)

        plt.plot(freq, temp - q[2], color=color or f"C{i}", label=label)
        plt.fill_between(
            freq, temp - q[1], temp - q[3], color=color or f"C{i}", alpha=0.6
        )
        plt.fill_between(
            freq, temp - q[0], temp - q[4], color=color or f"C{i}", alpha=0.4
        )
