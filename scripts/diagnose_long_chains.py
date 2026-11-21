"""Run longer, independently initialized chains at a saved pilot's frozen settings."""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.anchors import supplied_anchor
from sgld_cv.data import german_credit, year_model, synthetic_logistic
from sgld_cv.diagnostics import chain_diagnostics
from sgld_cv.estimators import prepare_estimator
from sgld_cv.randomness import generator
from sgld_cv.samplers import run_chains
from sgld_cv.workflow import METHODS, save_json


def diagnose(pilot, output, *, warmup, draws, seed, methods=METHODS, step_scale=1.):
    pilot, output = Path(pilot), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((pilot / "manifest.json").read_text())["config"]
    problem = config["problem"]
    if problem["kind"] == "german_credit":
        model, _ = german_credit(**{k: v for k, v in problem.items() if k != "kind"})
    elif problem["kind"] == "synthetic_logistic":
        model, _ = synthetic_logistic(**{k: v for k, v in problem.items() if k != "kind"})
    elif problem["kind"] == "year_ridge":
        model, _ = year_model(**{k: v for k, v in problem.items() if k != "kind"})
    else:
        raise ValueError("long-chain diagnostic supports logistic, German and Year pilots")
    with np.load(pilot / "geometry.npz") as geometry:
        theta, H = geometry["anchor"], geometry["hessian"]
    anchor = supplied_anchor(theta, provenance="saved pilot anchor for diagnostic")
    settings = json.loads((pilot / "frozen_selection.json").read_text())["selection"]
    pilots = json.loads((pilot / "metric_pilots.json").read_text())
    initial = theta + generator(seed, 0, 83).normal(size=(4, model.d)) @ np.linalg.cholesky(np.linalg.inv(H)).T
    results = {}
    if not np.isfinite(step_scale) or step_scale <= 0:
        raise ValueError("step_scale must be finite and positive")
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"unknown method {method}")
        setting = settings[method]
        base = method[1:] if method.startswith("p") else method
        options = {}
        if method.startswith("p"):
            options = {"preconditioner": "rmsprop",
                "initial_v": np.array(pilots[str(setting["batch_size"])]["v"]),
                "alpha": config["metric"].get("alpha", .99),
                "epsilon": config["metric"].get("epsilon", 1e-5)}
        with threadpool_limits(limits=1):
            samples, run = run_chains(prepare_estimator(model, base, anchor),
                initial=initial, step_size=step_scale * setting["step_size"],
                batch_size=setting["batch_size"], warmup=warmup, draws=draws,
                chains=4, seed=seed, **options)
            diag = chain_diagnostics(samples, adaptive=method.startswith("p"))
            eig, vectors = np.linalg.eigh(H)
            weak = chain_diagnostics((samples @ vectors[:, 0])[:, :, None],
                                     adaptive=method.startswith("p"))
        results[method] = {"coordinate_diagnostics": diag,
            "weak_direction_diagnostics": weak,
            "mean": samples.mean(axis=(0, 1)),
            "covariance": np.cov(samples.reshape(-1, model.d), rowvar=False),
            "failures": run.failures,
            "sampler_total_seconds_excluding_saved_pilot_setup": run.costs[-1]["total_seconds"]}
        save_json(output / "summary.json", {"source_pilot": str(pilot),
            "warmup": warmup, "draws_per_chain": draws, "seed": seed,
            "step_scale": step_scale,
            "methods": results,
            "interpretation": "Exploratory long-chain mixing check at previously selected steps/batches. Adaptive RMSprop ESS remains exploratory; do not rank methods by this diagnostic run's timing."})
        print(method, "Rhat", diag["max_rhat"], "bulk ESS", diag["min_bulk_ess"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--seed", type=int, default=410300)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--step-scale", type=float, default=1.)
    args = parser.parse_args()
    diagnose(args.pilot, args.output, warmup=args.warmup,
             draws=args.draws, seed=args.seed, methods=args.methods,
             step_scale=args.step_scale)


if __name__ == "__main__":
    main()
