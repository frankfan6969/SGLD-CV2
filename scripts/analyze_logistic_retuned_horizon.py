"""Rescore retuned 2D logistic horizons with a larger HMC reference pool."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.randomness import generator
from sgld_cv.transport import make_accuracy_metric
from sgld_cv.workflow import METHODS, save_csv, save_json


def run(folder, reference, output, *, reference_size=3072, bootstraps=48):
    folder, reference, output = map(Path, (folder, reference, output))
    output.mkdir(parents=True, exist_ok=False)
    with (folder / "progress.json").open() as file:
        progress = json.load(file)
    seeds = progress["seeds"]
    with np.load(reference) as data:
        first = data["first"].reshape(-1, data["first"].shape[-1])
        longer = data["longer"].reshape(-1, data["longer"].shape[-1])
    if reference_size > min(len(first), len(longer)):
        raise ValueError("reference size exceeds HMC pool")
    if bootstraps < 0:
        raise ValueError("bootstraps must be nonnegative")
    metric = make_accuracy_metric(longer.shape[1], kind="w2")
    references = {s: longer[generator(s, 0, 84).permutation(len(longer))[:reference_size]]
                  for s in seeds}
    rows, baselines, changes = [], [], {}
    rng = np.random.default_rng(2718)
    with threadpool_limits(limits=1):
        for seed in seeds:
            check = first[generator(seed, 1, 85).permutation(len(first))[:progress["replicas"]]]
            baselines.append({"seed": seed, "replicas": len(check),
                "reference_size": reference_size,
                "reference_reference_w2": metric.distance(check, references[seed])})
        for method in METHODS:
            by_seed = {}
            for seed in seeds:
                path = folder / f"states_{method}_seed{seed}.npz"
                with np.load(path) as data:
                    by_seed[seed] = {int(k): x for k, x in zip(data["checkpoints"], data["states"])}
                for k, x in by_seed[seed].items():
                    if k == 0:
                        continue
                    for r in (96, 192, progress["replicas"]):
                        rows.append({"method": method, "seed": seed, "iteration": k,
                            "replicas": r, "reference_size": reference_size,
                            "accuracy_w2": metric.distance(x[:r], references[seed])})
            end = max(by_seed[seeds[0]])
            start = end // 2
            observed = [metric.distance(by_seed[s][end], references[s]) -
                        metric.distance(by_seed[s][start], references[s]) for s in seeds]
            boot = []
            for _ in range(bootstraps):
                paired = []
                for seed in seeds:
                    states = by_seed[seed]
                    index = rng.integers(len(states[end]), size=len(states[end]))
                    paired.append(metric.distance(states[end][index], references[seed]) -
                                  metric.distance(states[start][index], references[seed]))
                boot.append(float(np.mean(paired)))
            changes[method] = {"start": start, "end": end,
                "seed_changes": observed, "mean_change": float(np.mean(observed)),
                "conditional_bootstrap_q05": float(np.quantile(boot, .05)) if boot else None,
                "conditional_bootstrap_q95": float(np.quantile(boot, .95)) if boot else None}
            save_csv(output / "scores.csv", rows)
            save_json(output / "changes.json", changes)
    save_csv(output / "reference_baseline.csv", baselines)
    save_json(output / "summary.json", {"source_horizon": str(folder),
        "source_reference": str(reference), "reference_size": reference_size,
        "replicas": progress["replicas"], "seeds": seeds,
        "bootstraps": bootstraps,
        "interpretation": "Final-half paired W2 changes condition on two seeds and fixed HMC reference samples. Bootstrap intervals do not include between-seed or reference-pool uncertainty, so do not alone certify final K."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path,
        default=Path("results/logistic_retuned_horizon"))
    parser.add_argument("--reference", type=Path,
        default=Path("results/logistic_reference_refined/reference_chains.npz"))
    parser.add_argument("--output", type=Path,
        default=Path("results/logistic_retuned_horizon_rescored"))
    parser.add_argument("--reference-size", type=int, default=3072)
    parser.add_argument("--bootstraps", type=int, default=48)
    args = parser.parse_args()
    run(args.folder, args.reference, args.output,
        reference_size=args.reference_size, bootstraps=args.bootstraps)


if __name__ == "__main__":
    main()
