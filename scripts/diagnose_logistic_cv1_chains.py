"""Extend the same CV1 chain seed to test the held-out R-hat failure."""
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


def run(config_path, output):
    config = json.loads(Path(config_path).read_text())
    if config["purpose"] != "post_heldout_targeted_diagnostics":
        raise ValueError("not a post-held-out diagnostic protocol")
    spec = config["cv1"]
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    pilot = Path(config["source_pilot"])
    base_config = json.loads((pilot / "manifest.json").read_text())["config"]
    model, _ = synthetic_logistic(**{k: v for k,v in base_config["problem"].items()
                                   if k != "kind"})
    anchor = find_anchor(model)
    with np.load(pilot / "geometry.npz") as data:
        np.testing.assert_allclose(anchor.theta, data["anchor"], rtol=0, atol=1e-10)
        H = data["hessian"]
    setting = json.loads(Path(config["source_selection"]).read_text())["selection"]["cv1"]
    h = setting["step_size"] * spec["step_scale"]
    previous = json.loads((Path(config["source_chain_validation"]) / "cv1.json").read_text())
    if (previous["seed"] != spec["chain_seed"] or previous["step_size"] != h or
            previous["batch_size"] != setting["batch_size"] or
            previous["warmup"] != spec["warmup"] or
            previous["draws_per_chain"] != spec["retained_prefixes"][0]):
        raise ValueError("saved 60k-chain diagnostic does not match the extension")
    ref = json.loads((Path(config["source_reference"]) /
                      "reference_diagnostics.json").read_text())
    if not ref["reliable"]:
        raise ValueError("saved HMC reference failed its reliability check")
    whitening = np.linalg.cholesky(H)
    initial = anchor.theta + generator(spec["chain_seed"],0,83).normal(
        size=(spec["chains"],model.d)) @ np.linalg.cholesky(np.linalg.inv(H)).T
    estimator = prepare_estimator(model, "cv1", anchor)
    with threadpool_limits(limits=1):
        samples, chain_run = run_chains(estimator, initial=initial,
            step_size=h, batch_size=setting["batch_size"],
            warmup=spec["warmup"], draws=spec["retained_prefixes"][-1],
            chains=spec["chains"], seed=spec["chain_seed"])
    if chain_run.failures or not np.isfinite(samples).all():
        raise RuntimeError("CV1 chain extension failed")
    results = {}
    for draws in spec["retained_prefixes"]:
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
    first = results[str(spec["retained_prefixes"][0])]["diagnostics"]
    old = previous["diagnostics"]
    for key in ("max_rhat","min_bulk_ess","min_tail_ess"):
        if not np.isclose(first[key], old[key], rtol=0, atol=1e-8):
            raise ValueError(f"CV1 60k prefix did not reproduce saved {key}")
    output.mkdir(parents=True)
    save_json(output / "summary.json", {"method":"cv1", "step_size":h,
        "batch_size":setting["batch_size"], "chain_seed":spec["chain_seed"],
        "warmup":spec["warmup"], "chains":spec["chains"],
        "retained_prefixes":results, "full_cost":chain_run.costs[-1],
        "interpretation":"Same-seed chain continuation diagnostic; no K_needed inference from R-hat alone."})
    print("CV1 chain extension complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,
        default=Path("configs/logistic_postheldout_diagnostics.json"))
    parser.add_argument("--output",type=Path,
        default=Path("results/logistic_cv1_chain_extension"))
    args=parser.parse_args()
    run(args.config,args.output)


if __name__ == "__main__":
    main()
