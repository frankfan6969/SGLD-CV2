"""Summarize retuned CV2 replica horizons over four independently seeded runs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sgld_cv.workflow import save_json


def run(folders, reference_diagnostics, output, *, bootstraps=10000):
    folders = [Path(x) for x in folders]
    reference_diagnostics, output = Path(reference_diagnostics), Path(output)
    ref = json.loads(reference_diagnostics.read_text())
    records = {}
    for folder in folders:
        progress = json.loads((folder / "progress.json").read_text())
        if sorted(progress["completed_seeds"]) != sorted(progress["seeds"]):
            raise ValueError(f"incomplete run: {folder}")
        with (folder / "scores.csv").open() as file:
            for row in csv.DictReader(file):
                key = (int(row["seed"]), int(row["replicas"]), int(row["iteration"]))
                if key in records:
                    raise ValueError(f"duplicate score: {key}")
                records[key] = {k: float(row[k]) for k in
                    ("accuracy_w2", "mean_error_std", "covariance_error_std")}
    seeds = sorted({key[0] for key in records})
    if len(seeds) < 4:
        raise ValueError("at least four independent seeds are required")
    rng = np.random.default_rng(7503)
    results = {}
    for r in (384, 768):
        by_k = {}
        for k in (2048, 4096, 8192, 16384):
            current = [records[(seed, r, k)] for seed in seeds]
            by_k[str(k)] = {"mean_w2": float(np.mean([x["accuracy_w2"] for x in current])),
                "seed_w2": [x["accuracy_w2"] for x in current],
                "guarded_mean_errors": [x["mean_error_std"] +
                    ref["standardized_mean_difference"] for x in current],
                "guarded_covariance_errors": [x["covariance_error_std"] +
                    ref["standardized_covariance_difference"] for x in current]}
        changes = {}
        for start, end in ((2048, 4096), (4096, 8192), (8192, 16384)):
            values = np.array([records[(seed, r, end)]["accuracy_w2"] -
                records[(seed, r, start)]["accuracy_w2"] for seed in seeds])
            samples = rng.choice(values, size=(bootstraps, len(values)), replace=True).mean(axis=1)
            changes[f"{start}_to_{end}"] = {"seed_changes": values,
                "mean_change": float(values.mean()),
                "seed_bootstrap_q05": float(np.quantile(samples, .05)),
                "seed_bootstrap_q95": float(np.quantile(samples, .95))}
        results[str(r)] = {"checkpoints": by_k, "changes": changes}
    save_json(output, {"source_folders": [str(x) for x in folders],
        "source_reference": str(reference_diagnostics),
        "seeds": seeds, "reference_size": 3072,
        "seed_bootstraps": bootstraps, "results": results,
        "interpretation": "Bootstrap resamples only four independent run seeds, not replicas or HMC references. This exploratory interval better reflects between-run variation than a replica-only interval but remains imprecise at n=4."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folders", nargs="+", type=Path,
        default=[Path("results/logistic_retuned_cv2_r768"),
                 Path("results/logistic_retuned_cv2_r768_newseeds")])
    parser.add_argument("--reference", type=Path,
        default=Path("results/logistic_reference_refined/reference_diagnostics.json"))
    parser.add_argument("--output", type=Path,
        default=Path("results/logistic_retuned_cv2_r768/four_seed_summary.json"))
    args = parser.parse_args()
    run(args.folders, args.reference, args.output)


if __name__ == "__main__":
    main()
