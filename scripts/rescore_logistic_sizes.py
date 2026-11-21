"""Refresh saved size-calibration moments against a longer HMC reference."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sgld_cv.workflow import save_csv, save_json


def run(folder, pilot, reference):
    folder, pilot, reference = map(Path, (folder, pilot, reference))
    diagnostics = json.loads(reference.read_text())
    if not diagnostics["reliable"]:
        raise ValueError("refined reference must pass diagnostics")
    with np.load(pilot / "geometry.npz") as data:
        whiten = np.linalg.cholesky(data["hessian"])
    mean = np.asarray(diagnostics["reference_mean"])
    covariance = np.asarray(diagnostics["reference_covariance"])
    with (folder / "size_scores.csv").open() as file:
        rows = list(csv.DictReader(file))
    errors = {}
    for row in rows:
        method, seed, k, r = row["method"], int(row["seed"]), int(row["iteration"]), int(row["replicas"])
        key = (method, seed, k, r)
        if key not in errors:
            with np.load(folder / f"states_{method}_seed{seed}.npz") as data:
                states = dict(zip(data["checkpoints"], data["states"]))
                x = states[k][:r]
            errors[key] = (
                float(np.linalg.norm((x.mean(axis=0) - mean) @ whiten) / np.sqrt(len(mean))),
                float(np.linalg.norm(whiten.T @
                    (np.cov(x, rowvar=False, ddof=0) - covariance) @ whiten) / len(mean)))
        row["mean_error_std"], row["covariance_error_std"] = errors[key]
    save_csv(folder / "size_scores_refined.csv", rows)
    save_json(folder / "refined_moment_scoring.json", {
        "reference_diagnostics": str(reference),
        "reference_reliable": True,
        "accuracy_scores_unchanged": True,
        "rows": len(rows),
        "interpretation": "Only moment targets changed. Each empirical W2 score is copied from the original saved size experiment."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path("results/logistic_size_calibration"))
    parser.add_argument("--pilot", type=Path,
        default=Path("results/stage456_validation/synthetic_logistic"))
    parser.add_argument("--reference", type=Path,
        default=Path("results/logistic_reference_refined/reference_diagnostics.json"))
    args = parser.parse_args()
    run(args.folder, args.pilot, args.reference)


if __name__ == "__main__":
    main()
