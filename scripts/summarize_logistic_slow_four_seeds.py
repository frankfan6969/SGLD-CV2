"""Four-seed R=384 check of slow logistic methods' horizon fluctuations."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sgld_cv.workflow import save_json


def run(first_folder, later_folder, reference, output, *, bootstraps=10000):
    first_folder, later_folder, reference, output = map(
        Path, (first_folder, later_folder, reference, output))
    progress = json.loads((later_folder / "progress.json").read_text())
    if len(progress["completed_runs"]) != 4:
        raise ValueError("new-seed slow-method run is incomplete")
    ref = json.loads(reference.read_text())
    records = {}
    for folder in (first_folder, later_folder):
        with (folder / "scores.csv").open() as file:
            for row in csv.DictReader(file):
                if row["method"] not in ("sgld", "psgld") or int(row["replicas"]) != 384:
                    continue
                key = (row["method"], int(row["seed"]), int(row["iteration"]))
                if key in records:
                    raise ValueError(f"duplicate state score: {key}")
                records[key] = row
    rng = np.random.default_rng(7504)
    results = {}
    for method in ("sgld", "psgld"):
        seeds = sorted({seed for m,seed,k in records if m == method})
        if seeds != [124,125,126,127]:
            raise ValueError(f"expected four seeds for {method}, found {seeds}")
        checkpoints = {}
        for k in (2048,4096,8192,16384):
            rows = [records[(method,seed,k)] for seed in seeds]
            checkpoints[str(k)] = {"seed_w2": [float(x["accuracy_w2"]) for x in rows],
                "seed_guarded_covariance_error": [float(x["covariance_error_std"]) +
                    ref["standardized_covariance_difference"] for x in rows]}
        changes = {}
        for start,end in ((2048,4096),(4096,8192),(8192,16384)):
            values = np.array([float(records[(method,s,end)]["accuracy_w2"]) -
                float(records[(method,s,start)]["accuracy_w2"]) for s in seeds])
            samples = rng.choice(values, size=(bootstraps,len(values)), replace=True).mean(axis=1)
            changes[f"{start}_to_{end}"] = {"seed_changes": values,
                "mean_change": float(values.mean()),
                "seed_bootstrap_q05": float(np.quantile(samples,.05)),
                "seed_bootstrap_q95": float(np.quantile(samples,.95))}
        results[method] = {"checkpoints": checkpoints, "changes": changes}
    save_json(output, {"source_first_seeds": str(first_folder),
        "source_new_seeds": str(later_folder),
        "source_reference": str(reference),
        "seeds": [124,125,126,127], "replicas": 384,
        "reference_size": 3072, "seed_bootstraps": bootstraps,
        "results": results,
        "interpretation": "Four-seed R=384 check of SGLD/pSGLD; shared random streams make methods paired, not independent replications. Seed bootstrap at n=4 remains exploratory."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path,
        default=Path("results/logistic_retuned_horizon_r768_others"))
    parser.add_argument("--later", type=Path,
        default=Path("results/logistic_retuned_slow_newseeds_r384"))
    parser.add_argument("--reference", type=Path,
        default=Path("results/logistic_reference_refined/reference_diagnostics.json"))
    parser.add_argument("--output", type=Path,
        default=Path("results/logistic_retuned_slow_newseeds_r384/four_seed_summary.json"))
    args = parser.parse_args()
    run(args.first, args.later, args.reference, args.output)


if __name__ == "__main__":
    main()
