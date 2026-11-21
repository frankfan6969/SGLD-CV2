"""Auxiliary fixed-calibration sliced-W2 check of retuned logistic horizons."""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.randomness import generator
from sgld_cv.transport import make_accuracy_metric, metric_fingerprint
from sgld_cv.workflow import METHODS, save_csv, save_json


def run(horizon, reference, output, *, sizes=(1536, 3072, 6144),
        replicas=384, projections=256, bootstraps=32):
    horizon, reference, output = map(Path, (horizon, reference, output))
    output.mkdir(parents=True, exist_ok=False)
    progress = json.loads((horizon / "progress.json").read_text())
    seeds = progress["seeds"]
    with np.load(reference) as data:
        first = data["first"].reshape(-1, data["first"].shape[-1])
        longer = data["longer"].reshape(-1, data["longer"].shape[-1])
    metric = make_accuracy_metric(first.shape[1], kind="standardized_sliced_w2",
        center=first.mean(axis=0), covariance=np.cov(first, rowvar=False),
        projections=projections, seed=926)
    np.savez_compressed(output / "calibration.npz", center=metric.center,
        transform=metric.transform, projections=metric.projections)
    rows, baseline, changes = [], [], {}
    rng = np.random.default_rng(2718)
    with threadpool_limits(limits=1):
        for seed in seeds:
            first_order = generator(seed, 1, 85).permutation(len(first))
            second_order = generator(seed, 0, 84).permutation(len(longer))
            for size in sizes:
                ref = longer[second_order[:size]]
                baseline.append({"seed": seed, "replicas": replicas,
                    "reference_size": size,
                    "reference_reference_sw2": metric.distance(first[first_order[:replicas]], ref)})
                for method in METHODS:
                    with np.load(horizon / f"states_{method}_seed{seed}.npz") as data:
                        states = dict(zip(data["checkpoints"], data["states"]))
                    for k, x in states.items():
                        if k:
                            rows.append({"method": method, "seed": seed,
                                "iteration": int(k), "replicas": replicas,
                                "reference_size": size,
                                "accuracy_sw2": metric.distance(x[:replicas], ref)})
            save_csv(output / "scores.csv", rows)
            save_csv(output / "reference_baseline.csv", baseline)
        size = sizes[-2] if len(sizes) > 1 else sizes[0]
        for method in METHODS:
            by_seed = {}
            for seed in seeds:
                with np.load(horizon / f"states_{method}_seed{seed}.npz") as data:
                    by_seed[seed] = dict(zip(data["checkpoints"], data["states"]))
            end = max(by_seed[seeds[0]])
            start = end // 2
            refs = {seed: longer[generator(seed, 0, 84).permutation(len(longer))[:size]]
                    for seed in seeds}
            observed = [metric.distance(by_seed[seed][end][:replicas], refs[seed]) -
                        metric.distance(by_seed[seed][start][:replicas], refs[seed]) for seed in seeds]
            boot = []
            for _ in range(bootstraps):
                paired = []
                for seed in seeds:
                    index = rng.integers(replicas, size=replicas)
                    paired.append(metric.distance(by_seed[seed][end][index], refs[seed]) -
                                  metric.distance(by_seed[seed][start][index], refs[seed]))
                boot.append(float(np.mean(paired)))
            changes[method] = {"start": start, "end": end,
                "reference_size": size,
                "seed_changes": observed, "mean_change": float(np.mean(observed)),
                "conditional_bootstrap_q05": float(np.quantile(boot, .05)),
                "conditional_bootstrap_q95": float(np.quantile(boot, .95))}
            save_json(output / "changes.json", changes)
    save_json(output / "summary.json", {"source_horizon": str(horizon),
        "source_reference": str(reference),
        "metric": "standardized sliced W2, auxiliary only",
        "calibration": "independent first HMC run",
        "calibration_sha256": metric_fingerprint(metric),
        "projection_seed": 926, "projections": projections,
        "reference_sizes": sizes, "replicas": replicas, "seeds": seeds,
        "bootstraps": bootstraps,
        "interpretation": "Auxiliary metric with fixed whitening/projections. Do not substitute for the primary exact-W2 endpoint or infer stationary convergence from a conditional bootstrap."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=Path,
        default=Path("results/logistic_retuned_horizon"))
    parser.add_argument("--reference", type=Path,
        default=Path("results/logistic_reference_refined/reference_chains.npz"))
    parser.add_argument("--output", type=Path,
        default=Path("results/logistic_retuned_horizon_sliced_aux"))
    args = parser.parse_args()
    run(args.horizon, args.reference, args.output)


if __name__ == "__main__":
    main()
