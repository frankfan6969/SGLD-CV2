"""Independent multichain diagnostics for the frozen German confirmation arms."""
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from calibrate_german_geometry_pairs import full_metric
from sgld_cv.anchors import find_anchor
from sgld_cv.data import german_credit
from sgld_cv.diagnostics import chain_diagnostics
from sgld_cv.estimators import prepare_estimator
from sgld_cv.randomness import generator
from sgld_cv.samplers import run_chains
from sgld_cv.transport import load_accuracy_metric, metric_fingerprint
from sgld_cv.workflow import save_json


FOLDER = Path("results/german_frozen_geometry_confirmation")


def main():
    protocol = json.loads((FOLDER / "protocol.json").read_text())
    config = protocol["config"]
    pilot = Path(config["problem_source"])
    problem = json.loads((pilot / "manifest.json").read_text())["config"]["problem"]
    model, _ = german_credit(**{k: v for k, v in problem.items() if k != "kind"})
    anchor = find_anchor(model)
    with np.load(pilot / "geometry.npz") as data:
        np.testing.assert_allclose(anchor.theta, data["anchor"], rtol=0, atol=1e-9)
        hessian = np.asarray(data["hessian"])
    inverse = np.linalg.solve(hessian, np.eye(model.d))
    initial_cholesky = np.linalg.cholesky((inverse + inverse.T) / 2)
    metric = load_accuracy_metric(config["scoring_calibration"], d=model.d)
    if metric_fingerprint(metric) != protocol["scoring_calibration_sha256"]:
        raise ValueError("changed German scoring calibration")
    ref = json.loads((Path(config["reference_source"]) /
                      "reference_diagnostics.json").read_text())
    if not ref["reliable"]:
        raise ValueError("unreliable German reference")
    mean_ref, cov_ref = np.asarray(ref["reference_mean"]), np.asarray(ref["reference_covariance"])
    spec = config["chain_diagnostics"]
    for seed in spec["seeds"]:
        for arm, setting in config["arms"].items():
            path = FOLDER / f"chain_{arm}_seed{seed}.json"
            raw = FOLDER / f"chain_{arm}_seed{seed}.npz"
            if path.exists() and raw.exists():
                print(arm, seed, "already complete", flush=True)
                continue
            if path.exists() or raw.exists():
                raise ValueError(f"partial German chain diagnostic: {arm}, {seed}")
            if setting["geometry"] == "identity":
                length = spec["identity"]
            elif arm == "full_cv1_h01":
                length = spec["full_cv1_h01"]
            else:
                length = spec["full_h03"]
            initial = anchor.theta + 2 * generator(seed, 0, 83).normal(
                size=(spec["chains"], model.d)) @ initial_cholesky.T
            with threadpool_limits(limits=1):
                estimator = prepare_estimator(model, setting["method"], anchor)
                options = {}
                if setting["geometry"] == "full_hessian":
                    matrix, metric_cost = full_metric(model, estimator, hessian)
                    options = {"preconditioner": "frozen_full",
                               "frozen_matrix": matrix, "metric_cost": metric_cost}
                samples, run = run_chains(estimator, initial=initial,
                    step_size=setting["step_size"], batch_size=config["batch_size"],
                    warmup=length["warmup"], draws=length["draws"],
                    chains=spec["chains"], seed=seed, **options)
            if run.failures or not np.isfinite(samples).all():
                raise RuntimeError(f"German chain diagnostic failed: {arm}, {seed}")
            diag = chain_diagnostics(samples, cost=run.costs[-1])
            flat = samples.reshape(-1, model.d)
            mean_error = np.linalg.norm((flat.mean(axis=0) - mean_ref) @
                                        metric.transform) / np.sqrt(model.d)
            cov_error = np.linalg.norm(metric.transform.T @
                (np.cov(flat, rowvar=False, ddof=0) - cov_ref) @
                metric.transform) / model.d
            row = {"arm": arm, "seed": seed, "warmup": length["warmup"],
                   "draws_per_chain": length["draws"], "chains": spec["chains"],
                   "diagnostics": diag, "standardized_mean_error": float(mean_error),
                   "standardized_covariance_error": float(cov_error),
                   "cost": run.costs[-1],
                   "interpretation": "separate multichain validation; not replica K"}
            with raw.open("wb") as stream:
                np.savez_compressed(stream, samples=samples, initial=initial)
            save_json(path, row)
            print(arm, seed, "Rhat", round(diag["max_rhat"], 4),
                  "bulk ESS", round(diag["min_bulk_ess"]), flush=True)


if __name__ == "__main__":
    main()
