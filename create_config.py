import sys
import os
import click
from edges_cal import CalibrationObservation
from edges_estimate.calibration import CalibratorQ
from edges_estimate.likelihoods import CalibrationChi2
from getdist import plots
from yabf import samplers
from yabf.core import yaml
import numpy as np
from pathlib import Path

@click.command()
@click.argument("path",type=click.Path(exists=True, file_okay=False))
@click.option("-l", "--freq-low", type=float, default=50)
@click.option("-h", "--freq-high", type=float, default=100)
@click.option("-c", "--cterms", type=int, default=10)
@click.option("-w", "--wterms", type=int, default=12)
@click.option("-m", "--use-var-model", type=bool, default=False)
@click.option("-b/-B", "--bounds/--no-bounds", default=True)
def main(path, freq_low, freq_high, cterms, wterms, use_var_model, bounds):

    direc = Path(__file__).parent.absolute()
    path = Path(path)
    
    if '2015_09' in str(path):
        resistance_f = 49.98
        resistance_m = 50.12
    else:
        resistance_f = 49.999
        resistance_m = 50.1501
        
    calobs = CalibrationObservation(
        path=path,
        f_low=freq_low, f_high=freq_high, run_num=None, cterms=cterms, wterms=wterms, 
        resistance_f=resistance_f, 
        load_kwargs={'cache_dir': '/data4/smurray/edges_cal_cache', 'ignore_times_percent': 5},
        s11_kwargs={'resistance': resistance_m},
        include_previous=False,
        compile_from_def=False,
    )

    fig = calobs.plot_coefficients()
    fig.savefig("coefficients.pdf")
    fig = calobs.plot_calibrated_temps()
    fig.savefig("calibrated_temps.pdf")
    
    # Create a CalibratorQ component that has all the active parameters, but 
    # is very basic.
    cq = CalibratorQ(
        calobs=calobs,
        params = {
            **{f"C1_{i}": {"fiducial": calobs.C1_poly.coefficients[-(i+1)]} for i in range(calobs.cterms)},
            **{f"C2_{i}": {"fiducial": calobs.C2_poly.coefficients[-(i+1)]} for i in range(calobs.cterms)},
            **{f"Tunc_{i}": {"fiducial": calobs.Tunc_poly.coefficients[-(i+1)]} for i in range(calobs.wterms)},
            **{f"Tcos_{i}": {"fiducial": calobs.Tcos_poly.coefficients[-(i+1)]} for i in range(calobs.wterms)},
            **{f"Tsin_{i}": {"fiducial": calobs.Tsin_poly.coefficients[-(i+1)]} for i in range(calobs.wterms)},        
        }
    )

    if not use_var_model:
        # Create a likelihood that has data that is Qp (we need to choose Tns and Tload here)
        lk = CalibrationChi2(
            components=(cq,),
            use_model_sigma=False,
            sigma = {k: np.sqrt(spec.spectrum.variance_Q) for k, spec in calobs._loads.items()},
            data={k: spec.spectrum.averaged_Q for k, spec in calobs._loads.items()}
        )

    file_name = direc/f"{path.name}_l{freq_low}MHz_h{freq_high}MHz_c{cterms}_w{wterms}{'_use_model' if use_var_model else ''}{'bounds' if bounds else '_no_bounds'}"
    
    if not file_name.exists():
        os.mkdir(file_name)

    # Write out necessary data files
    np.savez(file_name / "data.npz", **lk.data)
    np.savez(file_name / "sigma.npz", **lk.sigma)

    prms = {}
    for kind in ['C1', "C2", "Tunc", "Tcos", "Tsin"]:
        if bounds:
            prms[kind] = "\n          ".join(f"{kind}_{i}:\n            min: {coeff - 20*np.abs(coeff)}\n            max: {coeff + 20*np.abs(coeff)}\n            fiducial: {coeff}" for i, coeff in enumerate(getattr(calobs, f"{kind}_poly").coefficients[::-1]))
        else:
            prms[kind] = "\n          ".join(f"{kind}_{i}:\n            fiducial: {coeff}" for i, coeff in enumerate(getattr(calobs, f"{kind}_poly").coefficients[::-1]))

    pwd = os.getcwd()
    
    config = f"""
name: {file_name.name}
external_modules:
- edges_estimate
likelihoods:
  calibration:
    class: CalibrationChi2
    components:
      calibrator:
        class: CalibratorQ
        kwargs:
          path: {path}
          calobs_args:
            f_low: {freq_low}
            f_high: {calobs.freq.max}
            cterms: {cterms}
            wterms: {wterms}
            resistance_f: {calobs.lna.resistance}
            s11_kwargs:
              resistance: {calobs.open.reflections.resistance}
            load_kwargs:
              ignore_times_percent: {calobs.open.spectrum.ignore_times_percent}
              cache_dir: {calobs.open.spectrum.cache_dir}
            load_s11s:
              open:
                n_terms: 151
            compile_from_def: false
        params:
          {prms['C1']}
          {prms['C2']}
          {prms['Tunc']}
          {prms['Tcos']}
          {prms['Tsin']}
    kwargs:
      use_model_sigma: {use_var_model}
      sigma: {os.path.join(pwd, file_name, 'sigma.npz')}
    data: {os.path.join(pwd, file_name, 'data.npz')}
"""
    
    with open(file_name / "config.yml", 'w') as fl:
        fl.write(config)

    print(f"Wrote config to {file_name / 'config.yml'}")
    
if __name__=="__main__":
    main()
