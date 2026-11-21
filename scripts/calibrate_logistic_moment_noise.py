"""Estimate finite-replica standardized-moment noise from independent HMC runs."""
import argparse
import json
from pathlib import Path

import numpy as np

from sgld_cv.workflow import save_json


def run(pilot, reference, output, *, sizes=(96, 192, 384, 768, 1536, 3072), repeats=200):
    pilot, reference, output = map(Path, (pilot, reference, output))
    with np.load(pilot / "geometry.npz") as data:
        whiten = np.linalg.cholesky(data["hessian"])
    with np.load(reference) as data:
        first = data["first"].reshape(-1, data["first"].shape[-1])
        longer = data["longer"].reshape(-1, data["longer"].shape[-1])
    if max(sizes) > len(longer):
        raise ValueError("requested size exceeds HMC draw pool")
    target_mean, target_cov = first.mean(axis=0), np.cov(first, rowvar=False)
    rng = np.random.default_rng(7501)
    results = {}
    for size in sizes:
        mean_errors, covariance_errors = [], []
        for _ in range(repeats):
            x = longer[rng.choice(len(longer), size=size, replace=False)]
            mean_errors.append(float(np.linalg.norm((x.mean(axis=0) - target_mean) @ whiten) /
                                     np.sqrt(x.shape[1])))
            covariance_errors.append(float(np.linalg.norm(whiten.T @
                (np.cov(x, rowvar=False, ddof=0) - target_cov) @ whiten) / x.shape[1]))
        results[str(size)] = {"mean_error_median": float(np.median(mean_errors)),
            "mean_error_q90": float(np.quantile(mean_errors, .9)),
            "covariance_error_median": float(np.median(covariance_errors)),
            "covariance_error_q90": float(np.quantile(covariance_errors, .9))}
    save_json(output, {"source_reference": str(reference),
        "source_geometry": str(pilot / "geometry.npz"),
        "target": "first HMC run", "sample_pool": "independent second HMC run",
        "sample_sizes": sizes, "repeats": repeats,
        "results": results,
        "interpretation": "Finite independent-replica moment-score noise under the HMC reference. Random draws from an MCMC pool are not fully independent, but the pool has high bulk ESS. These are diagnostic quantiles, not sampler error bounds."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path,
        default=Path("results/stage456_validation/synthetic_logistic"))
    parser.add_argument("--reference", type=Path,
        default=Path("results/logistic_reference_refined/reference_chains.npz"))
    parser.add_argument("--output", type=Path,
        default=Path("results/logistic_reference_refined/moment_noise_baseline.json"))
    parser.add_argument("--sizes", type=int, nargs="+",
        default=[96, 192, 384, 768, 1536, 3072])
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    run(args.pilot, args.reference, args.output,
        sizes=tuple(args.sizes), repeats=args.repeats)


if __name__ == "__main__":
    main()
