"""Check CV2 h/2 stationary moments and mixing at matched hK."""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.anchors import find_anchor
from sgld_cv.data import synthetic_logistic
from sgld_cv.diagnostics import chain_diagnostics
from sgld_cv.estimators import prepare_estimator
from sgld_cv.randomness import generator
from sgld_cv.samplers import run_chains
from sgld_cv.workflow import save_json


def run(config_path, output, *, draws=None):
    config = json.loads(Path(config_path).read_text())
    spec = dict(config["cv2"])
    if draws is not None:
        if draws < spec["draws_per_chain"] or draws % 2:
            raise ValueError("extended draws must be even and at least the original length")
        spec["draws_per_chain"] = draws
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    pilot = Path(config["source_pilot"])
    base_config = json.loads((pilot / "manifest.json").read_text())["config"]
    model, _ = synthetic_logistic(**{k:v for k,v in base_config["problem"].items()
                                   if k != "kind"})
    anchor = find_anchor(model)
    with np.load(pilot / "geometry.npz") as data:
        np.testing.assert_allclose(anchor.theta, data["anchor"], rtol=0, atol=1e-10)
        H = data["hessian"]
    setting = json.loads(Path(config["source_selection"]).read_text())["selection"]["cv2"]
    h = setting["step_size"] * spec["step_scale"]
    previous = json.loads((Path(config["source_chain_validation"]) / "cv2.json").read_text())
    if previous["seed"] != spec["chain_seed"] or previous["step_size"] != 2*h:
        raise ValueError("full-step comparison does not match this chain protocol")
    ref = json.loads((Path(config["source_reference"]) /
                      "reference_diagnostics.json").read_text())
    if not ref["reliable"]:
        raise ValueError("saved HMC reference failed checks")
    initial = anchor.theta + generator(spec["chain_seed"],0,83).normal(
        size=(spec["chains"],model.d)) @ np.linalg.cholesky(np.linalg.inv(H)).T
    estimator = prepare_estimator(model,"cv2",anchor)
    with threadpool_limits(limits=1):
        samples, chain_run = run_chains(estimator, initial=initial,
            step_size=h, batch_size=setting["batch_size"],
            warmup=spec["warmup"],draws=spec["draws_per_chain"],
            chains=spec["chains"],seed=spec["chain_seed"])
    if chain_run.failures or not np.isfinite(samples).all():
        raise RuntimeError("CV2 half-step chain failed")
    whitening = np.linalg.cholesky(H)
    results = {}
    for draws in (spec["draws_per_chain"]//2,spec["draws_per_chain"]):
        x = samples[:,:draws,:]
        diag = chain_diagnostics(x)
        flat = x.reshape(-1,model.d)
        mean_error = float(np.linalg.norm((flat.mean(axis=0)-ref["reference_mean"]) @
            whitening) / np.sqrt(model.d) + ref["standardized_mean_difference"])
        cov_error = float(np.linalg.norm(whitening.T @
            (np.cov(flat,rowvar=False,ddof=0)-ref["reference_covariance"]) @
            whitening) / model.d + ref["standardized_covariance_difference"])
        results[str(draws)] = {"diagnostics":diag,
            "guarded_mean_error":mean_error,
            "guarded_covariance_error":cov_error}
    output.mkdir(parents=True)
    save_json(output / "summary.json", {"method":"cv2", "step_size":h,
        "batch_size":setting["batch_size"],"chain_seed":spec["chain_seed"],
        "warmup":spec["warmup"],"chains":spec["chains"],
        "retained_prefixes":results,
        "full_step_comparison": {"step_size":previous["step_size"],
            "draws_per_chain":previous["draws_per_chain"],
            "diagnostics":previous["diagnostics"],
            "guarded_mean_error":previous["guarded_mean_error"],
            "guarded_covariance_error":previous["guarded_covariance_error"]},
        "full_cost":chain_run.costs[-1],
        "interpretation":"Exploratory h/2 sensitivity; 60k retained at h/2 matches 30k retained at h in cumulative step size, while setup/warmup and paths differ."})
    print("CV2 half-step chain check complete", flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,
        default=Path("configs/logistic_postheldout_diagnostics.json"))
    parser.add_argument("--output",type=Path,
        default=Path("results/logistic_cv2_half_chain_diagnostic"))
    parser.add_argument("--draws",type=int)
    args=parser.parse_args()
    run(args.config,args.output,draws=args.draws)


if __name__ == "__main__":
    main()
