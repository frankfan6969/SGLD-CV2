"""Compare CV2 full/half steps at approximately matched cumulative step size."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.randomness import generator
from sgld_cv.transport import make_accuracy_metric
from sgld_cv.workflow import save_json


def _rows(path):
    with Path(path).open() as file:
        return {(int(x["seed"]), int(x["iteration"]), int(x["replicas"])): x
                for x in csv.DictReader(file) if x["method"] == "cv2"}


def run(full_folder, full_scores, half_folder, reference, output, *, seeds=(124, 125),
        replicas=384, reference_size=3072, bootstraps=48):
    full_folder, full_scores, half_folder, reference, output = map(
        Path, (full_folder, full_scores, half_folder, reference, output))
    half_progress = json.loads((half_folder / "progress.json").read_text())
    if sorted(half_progress["completed_seeds"]) != sorted(seeds):
        raise ValueError("half-step replica run is incomplete")
    if half_progress["step_scale"] != .5:
        raise ValueError("expected half-step CV2 run")
    full = _rows(full_scores)
    full_moments = _rows(full_folder / "scores.csv")
    half = _rows(half_folder / "scores.csv")
    with np.load(reference) as data:
        longer = data["longer"].reshape(-1, data["longer"].shape[-1])
    metric = make_accuracy_metric(longer.shape[1], kind="w2")
    rng = np.random.default_rng(2718)
    observed = []
    states = {}
    refs = {}
    with threadpool_limits(limits=1):
        for seed in seeds:
            full_row = full[(seed, 8192, replicas)]
            full_moment_row = full_moments[(seed, 8192, replicas)]
            half_row = half[(seed, 16384, replicas)]
            observed.append({"seed": seed,
                "full_w2": float(full_row["accuracy_w2"]),
                "half_w2": float(half_row["accuracy_w2"]),
                "half_minus_full_w2": float(half_row["accuracy_w2"]) -
                    float(full_row["accuracy_w2"]),
                "full_mean_error_std": float(full_moment_row["mean_error_std"]),
                "half_mean_error_std": float(half_row["mean_error_std"]),
                "full_covariance_error_std": float(full_moment_row["covariance_error_std"]),
                "half_covariance_error_std": float(half_row["covariance_error_std"])})
            with np.load(full_folder / f"states_cv2_seed{seed}.npz") as data:
                full_states = dict(zip(data["checkpoints"], data["states"]))
            with np.load(half_folder / f"states_cv2_seed{seed}.npz") as data:
                half_states = dict(zip(data["checkpoints"], data["states"]))
            states[seed] = (full_states[8192][:replicas], half_states[16384][:replicas])
            refs[seed] = longer[generator(seed, 0, 84).permutation(len(longer))[:reference_size]]
        boot = []
        for _ in range(bootstraps):
            paired = []
            for seed in seeds:
                index = rng.integers(replicas, size=replicas)
                a, b = states[seed]
                paired.append(metric.distance(b[index], refs[seed]) -
                              metric.distance(a[index], refs[seed]))
            boot.append(float(np.mean(paired)))
    save_json(output, {"source_full_step": str(full_folder),
        "source_full_scores": str(full_scores),
        "source_half_step": str(half_folder),
        "reference_size": reference_size, "replicas": replicas,
        "full_step_iteration": 8192, "half_step_iteration": 16384,
        "seeds": seeds, "seed_comparisons": observed,
        "mean_half_minus_full_w2": float(np.mean(
            [x["half_minus_full_w2"] for x in observed])),
        "conditional_bootstrap_q05": float(np.quantile(boot, .05)),
        "conditional_bootstrap_q95": float(np.quantile(boot, .95)),
        "interpretation": "Approximately equal cumulative step size, but half-step uses twice as many transitions. Two seeds and finite-replica W2 noise limit method choice; negative difference favors half-step."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path,
        default=Path("results/logistic_retuned_horizon"))
    parser.add_argument("--full-scores", type=Path,
        default=Path("results/logistic_retuned_horizon_rescored/scores.csv"))
    parser.add_argument("--half", type=Path,
        default=Path("results/logistic_retuned_cv2_half_r384"))
    parser.add_argument("--reference", type=Path,
        default=Path("results/logistic_reference_refined/reference_chains.npz"))
    parser.add_argument("--output", type=Path,
        default=Path("results/logistic_retuned_cv2_half_r384/comparison.json"))
    args = parser.parse_args()
    run(args.full, args.full_scores, args.half, args.reference, args.output)


if __name__ == "__main__":
    main()
