"""Collect six-method retuned logistic R=768, S=3072 diagnostic endpoints."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sgld_cv.workflow import METHODS, save_json


def _read(folder):
    folder = Path(folder)
    progress = json.loads((folder / "progress.json").read_text())
    if len(progress.get("completed_runs", [])):
        expected = len(progress["methods"]) * len(progress["seeds"])
        if len(progress["completed_runs"]) != expected:
            raise ValueError(f"incomplete six-method run: {folder}")
    elif sorted(progress.get("completed_seeds", [])) != sorted(progress["seeds"]):
        raise ValueError(f"incomplete CV2 run: {folder}")
    with (folder / "scores.csv").open() as file:
        rows = list(csv.DictReader(file))
    return progress, rows


def run(other_folder, cv2_folder, reference, pilot, output):
    other_folder, cv2_folder, reference, pilot, output = map(
        Path, (other_folder, cv2_folder, reference, pilot, output))
    other_progress, other = _read(other_folder)
    cv2_progress, cv2 = _read(cv2_folder)
    if other_progress["replicas"] != cv2_progress["replicas"] or (
            other_progress["reference_size"] != cv2_progress["reference_size"]):
        raise ValueError("replica/reference sizes do not match")
    ref = json.loads(reference.read_text())
    with np.load(pilot / "geometry.npz") as data:
        whiten = np.linalg.cholesky(data["hessian"])
    ref_mean = np.asarray(ref["reference_mean"])
    ref_cov = np.asarray(ref["reference_covariance"])
    rows = other + cv2
    results = {}
    for method in METHODS:
        subset = [x for x in rows if x["method"] == method and int(x["replicas"]) == 768]
        seeds = sorted({int(x["seed"]) for x in subset})
        if seeds != [124, 125]:
            raise ValueError(f"unexpected seeds for {method}: {seeds}")
        by_seed = {(int(x["seed"]), int(x["iteration"])): x for x in subset}
        end = max(int(x["iteration"]) for x in subset)
        start = end // 2
        differences = [float(by_seed[(seed,end)]["accuracy_w2"]) -
                       float(by_seed[(seed,start)]["accuracy_w2"]) for seed in seeds]
        guarded_cov = [float(by_seed[(seed,end)]["covariance_error_std"]) +
                       ref["standardized_covariance_difference"] for seed in seeds]
        guarded_mean = [float(by_seed[(seed,end)]["mean_error_std"]) +
                        ref["standardized_mean_difference"] for seed in seeds]
        source = cv2_folder if method == "cv2" else other_folder
        pooled = {}
        for k in (start,end):
            states = []
            for seed in seeds:
                with np.load(source / f"states_{method}_seed{seed}.npz") as data:
                    checkpoints = dict(zip(data["checkpoints"], data["states"]))
                    states.append(checkpoints[k][:768])
            x = np.concatenate(states, axis=0)
            pooled[str(k)] = {"guarded_mean_error": float(np.linalg.norm(
                (x.mean(axis=0)-ref_mean) @ whiten) / np.sqrt(x.shape[1]) +
                ref["standardized_mean_difference"]),
                "guarded_covariance_error": float(np.linalg.norm(whiten.T @
                (np.cov(x,rowvar=False,ddof=0)-ref_cov) @ whiten) / x.shape[1] +
                ref["standardized_covariance_difference"])}
        results[method] = {"start": start, "end": end,
            "seed_changes_w2": differences, "mean_change_w2": float(np.mean(differences)),
            "guarded_mean_errors_at_end": guarded_mean,
            "guarded_covariance_errors_at_end": guarded_cov,
            "pooled_1536_replica_moment_errors": pooled,
            "end_moment_threshold_pass_by_seed": [m <= .1 and c <= .1
                for m,c in zip(guarded_mean, guarded_cov)]}
    save_json(output, {"source_other_methods": str(other_folder),
        "source_cv2": str(cv2_folder), "reference_diagnostics": str(reference),
        "replicas": 768, "reference_size": 3072,
        "seeds": [124,125], "results": results,
        "interpretation": "Two-seed endpoint diagnostic; finite-replica moment errors should be compared with independent-HMC noise. This is not a final K or method ranking."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--other", type=Path,
        default=Path("results/logistic_retuned_horizon_r768_others"))
    parser.add_argument("--cv2", type=Path,
        default=Path("results/logistic_retuned_cv2_r768"))
    parser.add_argument("--reference", type=Path,
        default=Path("results/logistic_reference_refined/reference_diagnostics.json"))
    parser.add_argument("--pilot", type=Path,
        default=Path("results/stage456_validation/synthetic_logistic"))
    parser.add_argument("--output", type=Path,
        default=Path("results/logistic_retuned_horizon_r768_others/two_seed_summary.json"))
    args = parser.parse_args()
    run(args.other, args.cv2, args.reference, args.pilot, args.output)


if __name__ == "__main__":
    main()
