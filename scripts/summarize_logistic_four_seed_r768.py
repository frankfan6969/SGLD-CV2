"""Combine retuned 2D logistic R=768 horizons from four independent seeds."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sgld_cv.workflow import METHODS, save_json


def run(folders, pilot, reference, output, *, bootstraps=10000):
    folders = [Path(x) for x in folders]
    pilot, reference, output = map(Path, (pilot, reference, output))
    ref = json.loads(reference.read_text())
    with np.load(pilot / "geometry.npz") as data:
        whitening = np.linalg.cholesky(data["hessian"])
    target_mean = np.asarray(ref["reference_mean"])
    target_cov = np.asarray(ref["reference_covariance"])
    rows, state_paths = {}, {}
    for folder in folders:
        with (folder / "scores.csv").open() as file:
            for row in csv.DictReader(file):
                if int(row["replicas"]) != 768:
                    continue
                key = row["method"], int(row["seed"]), int(row["iteration"])
                if key in rows:
                    raise ValueError(f"duplicate score: {key}")
                rows[key] = row
                state_paths[(key[0], key[1])] = folder / f"states_{key[0]}_seed{key[1]}.npz"
    rng = np.random.default_rng(7505)
    results = {}
    for method in METHODS:
        seeds = sorted({s for m,s,k in rows if m == method})
        if seeds != [124,125,126,127]:
            raise ValueError(f"expected four complete seeds for {method}: {seeds}")
        checkpoints = sorted({k for m,s,k in rows if m == method})
        if any((method,s,k) not in rows for s in seeds for k in checkpoints):
            raise ValueError(f"missing checkpoint for {method}")
        by_k, changes = {}, {}
        saved_states = {}
        for seed in seeds:
            with np.load(state_paths[(method,seed)]) as data:
                saved_states[seed] = {int(k):x[:768] for k,x in
                    zip(data["checkpoints"],data["states"])}
        for k in checkpoints:
            records = [rows[(method,s,k)] for s in seeds]
            x = np.concatenate([saved_states[s][k] for s in seeds],axis=0)
            pooled_mean = float(np.linalg.norm((x.mean(axis=0)-target_mean) @ whitening)
                / np.sqrt(x.shape[1]) + ref["standardized_mean_difference"])
            pooled_cov = float(np.linalg.norm(whitening.T @
                (np.cov(x,rowvar=False,ddof=0)-target_cov) @ whitening)
                / x.shape[1] + ref["standardized_covariance_difference"])
            by_k[str(k)] = {"seed_w2": [float(row["accuracy_w2"]) for row in records],
                "mean_w2": float(np.mean([float(row["accuracy_w2"]) for row in records])),
                "seed_guarded_mean_errors": [float(row["mean_error_std"]) +
                    ref["standardized_mean_difference"] for row in records],
                "seed_guarded_covariance_errors": [float(row["covariance_error_std"]) +
                    ref["standardized_covariance_difference"] for row in records],
                "pooled_3072_guarded_mean_error": pooled_mean,
                "pooled_3072_guarded_covariance_error": pooled_cov}
        for start,end in zip(checkpoints[:-1],checkpoints[1:]):
            values = np.array([float(rows[(method,s,end)]["accuracy_w2"]) -
                float(rows[(method,s,start)]["accuracy_w2"]) for s in seeds])
            samples = rng.choice(values,size=(bootstraps,len(values)),replace=True).mean(axis=1)
            changes[f"{start}_to_{end}"] = {"seed_changes": values,
                "mean_change": float(values.mean()),
                "seed_bootstrap_q05": float(np.quantile(samples,.05)),
                "seed_bootstrap_q95": float(np.quantile(samples,.95))}
        results[method] = {"checkpoints": by_k, "changes": changes}
    save_json(output, {"source_folders": [str(x) for x in folders],
        "source_reference": str(reference), "source_geometry": str(pilot / "geometry.npz"),
        "seeds": [124,125,126,127], "replicas_per_seed": 768,
        "reference_size": 3072, "seed_bootstraps": bootstraps,
        "results": results,
        "interpretation": "Four-seed six-method diagnostic; bootstrap resamples independent run seeds, not reference draws. Shared random streams pair methods. Use reference-to-reference noise and long-chain stationary-moment checks before declaring K."})
    selection = json.loads(Path("results/logistic_step_validation/diagnostic_selection.json").read_text())["selection"]
    with output.with_suffix(".csv").open("w", newline="") as file:
        columns = ["method", "step_size", "seeds", "replicas_per_seed", "reference_size",
            "w2_k2048", "w2_k4096", "w2_k8192", "w2_k16384", "endpoint_k",
            "endpoint_pooled_guarded_mean_error", "endpoint_pooled_guarded_covariance_error",
            "last_interval", "last_w2_change", "last_w2_change_bootstrap_q05",
            "last_w2_change_bootstrap_q95"]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for method in METHODS:
            summary = results[method]
            checkpoints = summary["checkpoints"]
            endpoint = max(map(int, checkpoints))
            previous = sorted(map(int, checkpoints))[-2]
            change = summary["changes"][f"{previous}_to_{endpoint}"]
            final = checkpoints[str(endpoint)]
            writer.writerow({
                "method": method, "step_size": selection[method]["step_size"],
                "seeds": "124,125,126,127", "replicas_per_seed": 768,
                "reference_size": 3072,
                **{f"w2_k{k}": checkpoints.get(str(k), {}).get("mean_w2", "")
                   for k in (2048, 4096, 8192, 16384)},
                "endpoint_k": endpoint,
                "endpoint_pooled_guarded_mean_error": final["pooled_3072_guarded_mean_error"],
                "endpoint_pooled_guarded_covariance_error": final["pooled_3072_guarded_covariance_error"],
                "last_interval": f"{previous}->{endpoint}",
                "last_w2_change": change["mean_change"],
                "last_w2_change_bootstrap_q05": change["seed_bootstrap_q05"],
                "last_w2_change_bootstrap_q95": change["seed_bootstrap_q95"],
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folders",nargs="+",type=Path,
        default=[Path("results/logistic_retuned_horizon_r768_others"),
                 Path("results/logistic_retuned_cv2_r768"),
                 Path("results/logistic_retuned_cv2_r768_newseeds"),
                 Path("results/logistic_retuned_slow_newseeds_r768"),
                 Path("results/logistic_retuned_fast_newseeds_r768")])
    parser.add_argument("--pilot",type=Path,
        default=Path("results/stage456_validation/synthetic_logistic"))
    parser.add_argument("--reference",type=Path,
        default=Path("results/logistic_reference_refined/reference_diagnostics.json"))
    parser.add_argument("--output",type=Path,
        default=Path("results/logistic_four_seed_r768_summary.json"))
    args = parser.parse_args()
    run(args.folders,args.pilot,args.reference,args.output)


if __name__ == "__main__":
    main()
