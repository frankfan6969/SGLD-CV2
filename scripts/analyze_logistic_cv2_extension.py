"""Paired exact-W2 intervals for the targeted retuned CV2 R=768 run."""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.randomness import generator
from sgld_cv.transport import make_accuracy_metric
from sgld_cv.workflow import save_json


def run(folder, output, *, bootstraps=48):
    folder, output = Path(folder), Path(output)
    progress = json.loads((folder / "progress.json").read_text())
    seeds = progress["seeds"]
    if len(progress["completed_seeds"]) != len(seeds):
        raise ValueError("CV2 extension is incomplete")
    with np.load(progress["source_reference"]) as data:
        longer = data["longer"].reshape(-1, data["longer"].shape[-1])
    references = {seed: longer[generator(seed, 0, 84).permutation(len(longer))[
        :progress["reference_size"]]] for seed in seeds}
    states = {}
    for seed in seeds:
        with np.load(folder / f"states_cv2_seed{seed}.npz") as data:
            states[seed] = dict(zip(data["checkpoints"], data["states"]))
    metric = make_accuracy_metric(longer.shape[1], kind="w2")
    rng = np.random.default_rng(2718)
    results = {}
    with threadpool_limits(limits=1):
        for r in (384, progress["replicas"]):
            results[str(r)] = {}
            for start, end in ((2048, 4096), (4096, 8192), (8192, 16384)):
                observed = [metric.distance(states[s][end][:r], references[s]) -
                            metric.distance(states[s][start][:r], references[s]) for s in seeds]
                boot = []
                for _ in range(bootstraps):
                    paired = []
                    for seed in seeds:
                        index = rng.integers(r, size=r)
                        paired.append(metric.distance(states[seed][end][index], references[seed]) -
                                      metric.distance(states[seed][start][index], references[seed]))
                    boot.append(float(np.mean(paired)))
                results[str(r)][f"{start}_to_{end}"] = {
                    "seed_changes": observed, "mean_change": float(np.mean(observed)),
                    "conditional_bootstrap_q05": float(np.quantile(boot, .05)),
                    "conditional_bootstrap_q95": float(np.quantile(boot, .95))}
            save_json(output, {"source_extension": str(folder),
                "replicas": (384, progress["replicas"]),
                "reference_size": progress["reference_size"],
                "seeds": seeds, "bootstraps": bootstraps,
                "results": results,
                "interpretation": "Conditional paired-replica intervals; two seeds and fixed HMC samples do not establish a final K."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path,
        default=Path("results/logistic_retuned_cv2_r768"))
    parser.add_argument("--output", type=Path,
        default=Path("results/logistic_retuned_cv2_r768/changes.json"))
    args = parser.parse_args()
    run(args.folder, args.output)


if __name__ == "__main__":
    main()
