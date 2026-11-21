"""Exploratory full-Hessian preconditioned CV2 control for German and Year.

Uses a fixed inverse mode Hessian in both drift and noise. This incurs full
data geometry/setup costs and is not one of the six primary methods.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.anchors import supplied_anchor
from sgld_cv.data import german_credit, year_model
from sgld_cv.diagnostics import chain_diagnostics
from sgld_cv.estimators import prepare_estimator
from sgld_cv.randomness import generator, BATCH, NOISE
from sgld_cv.transport import load_accuracy_metric
from sgld_cv.workflow import save_json


def run(pilot, output, *, calibration, warmup=1000, draws=4000, batch_size=32,
        steps=(.1, .3), seed=7300):
    pilot, output = Path(pilot), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((pilot / "manifest.json").read_text())["config"]
    problem = config["problem"]
    options = {k: v for k, v in problem.items() if k != "kind"}
    if problem["kind"] == "german_credit":
        model, _ = german_credit(**options)
    elif problem["kind"] == "year_ridge":
        model, _ = year_model(**options)
    else:
        raise ValueError("full-metric control supports German and Year pilots")
    with np.load(pilot / "geometry.npz") as data:
        theta, H = data["anchor"], data["hessian"]
    start = perf_counter()
    covariance = np.linalg.solve(H, np.eye(model.d))
    L = np.linalg.cholesky(covariance)
    geometry_seconds = perf_counter() - start
    anchor = supplied_anchor(theta, provenance="saved mode and full-data Hessian")
    estimator = prepare_estimator(model, "cv2", anchor)
    if problem["kind"] == "german_credit":
        with np.load(pilot / "reference_chains.npz") as data:
            reference = data["longer"].reshape(-1, model.d)
        reference_kind = "approximate independent HMC run"
    else:
        reference = theta + generator(seed + 1, 0, 84).normal(size=(4096, model.d)) @ L.T
        reference_kind = "independent exact Gaussian posterior draws"
    metric = load_accuracy_metric(calibration, d=model.d)
    reference_mean = reference.mean(axis=0)
    reference_covariance = np.cov(reference, rowvar=False)
    initial = theta + generator(seed, 0, 83).normal(size=(4, model.d)) @ L.T
    results = {}
    with threadpool_limits(limits=1):
        for h in steps:
            samples = np.empty((4, draws, model.d))
            start = perf_counter()
            for chain in range(4):
                state = initial[chain].copy()
                batch_rng = generator(seed, chain, BATCH)
                noise_rng = generator(seed, chain, NOISE)
                for iteration in range(warmup + draws):
                    index = batch_rng.choice(model.N, size=batch_size, replace=False)
                    gradient = estimator.gradient(state, index)
                    state = state - (h / 2) * (covariance @ gradient) + np.sqrt(h) * (L @ noise_rng.normal(size=model.d))
                    if not np.isfinite(state).all():
                        raise FloatingPointError(f"nonfinite full-metric control at h={h}")
                    if iteration >= warmup:
                        samples[chain, iteration - warmup] = state
            elapsed = perf_counter() - start
            diag = chain_diagnostics(samples)
            eig, vectors = np.linalg.eigh(H)
            weak = chain_diagnostics((samples @ vectors[:, 0])[:, :, None])
            flat = samples.reshape(-1, model.d)
            # Keep scoring at a fixed, moderately large sample size; HMC is approximate.
            score = metric.distance(flat[:512], reference[:512])
            results[str(h)] = {"coordinate_diagnostics": diag,
                "weak_direction_diagnostics": weak,
                "standardized_mean_error": float(np.linalg.norm((flat.mean(axis=0) - reference_mean) @ metric.transform) / np.sqrt(model.d)),
                "standardized_covariance_error": float(np.linalg.norm(metric.transform.T @
                    (np.cov(flat, rowvar=False) - reference_covariance) @ metric.transform) / model.d),
                "standardized_sliced_w2_512_each": score,
                "transition_seconds_excluding_full_geometry_setup": elapsed}
            save_json(output / "summary.json", {"source_pilot": str(pilot),
                "method": "exploratory full-Hessian-preconditioned CV2 control",
                "reference_kind": reference_kind,
                "warmup": warmup, "draws_per_chain": draws, "chains": 4,
                "batch_size": batch_size, "seed": seed,
                "full_geometry_inversion_seconds_excluding_original_full_data_Hessian_construction": geometry_seconds,
                "results": results,
                "interpretation": "A geometry/bias diagnostic only. Full-data Hessian and inverse are extra setup costs; this control is not eligible for six-method throughput rankings. Compare step sizes for bias as well as mixing."})
            print("h", h, "Rhat", diag["max_rhat"], "bulk ESS", diag["min_bulk_ess"],
                  "mean error", results[str(h)]["standardized_mean_error"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path,
                        default=Path("results/stage456_validation/german_credit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path,
                        default=Path("results/pilot_gap_diagnostics/accuracy_calibration.npz"))
    parser.add_argument("--steps", type=float, nargs="+", default=[.1, .3])
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run(args.pilot, args.output, calibration=args.calibration, steps=args.steps,
        warmup=args.warmup, draws=args.draws, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
