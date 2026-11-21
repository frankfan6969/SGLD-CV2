"""Summarize the prespecified, held-out 2D logistic fixed-horizon run."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sgld_cv.workflow import save_json


def run(folder, *, bootstraps=20000):
    folder = Path(folder)
    protocol = json.loads((folder / "protocol.json").read_text())
    config = protocol["config"]
    methods, seeds = config["methods"], config["method_seeds"]
    reference = json.loads((Path(config["reference_source"]) /
                            "reference_diagnostics.json").read_text())
    if not reference["reliable"]:
        raise ValueError("reference is not reliable")
    with np.load(Path(config["problem_source"]) / "geometry.npz") as data:
        whitening = np.linalg.cholesky(data["hessian"])
    ref_mean = np.asarray(reference["reference_mean"])
    ref_cov = np.asarray(reference["reference_covariance"])
    records, rows, states = {}, [], {}
    for method in methods:
        for seed in seeds:
            path = folder / f"result_{method}_seed{seed}.json"
            state_path = folder / f"states_{method}_seed{seed}.npz"
            if not path.exists() or not state_path.exists():
                raise ValueError(f"incomplete held-out run: {method}, {seed}")
            record = json.loads(path.read_text())
            if (record["method"] != method or record["seed"] != seed or
                    [x["iteration"] for x in record["checkpoints"]] != config["checkpoints"]):
                raise ValueError(f"mismatched run record: {method}, {seed}")
            with np.load(state_path) as data:
                if not np.array_equal(data["checkpoints"], config["checkpoints"]):
                    raise ValueError(f"mismatched saved checkpoints: {method}, {seed}")
                states[(method, seed)] = data["states"][-1]
            records[(method, seed)] = record
            for item in record["checkpoints"]:
                rows.append({"method": method, "seed": seed,
                    "iteration": item["iteration"], "batch_size": record["batch_size"],
                    "step_size": record["step_size"], "replicas": record["replicas"],
                    "reference_size": record["reference_size"],
                    "w2": item["w2"],
                    "mean_error_guarded": item["mean_error_guarded"],
                    "covariance_error_guarded": item["covariance_error_guarded"],
                    "total_seconds": item["cost"]["total_seconds"],
                    "observation_gradients": item["cost"]["operations"].get(
                        "observation_gradient_evaluations", 0)})
    with (folder / "scores.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    baselines = [json.loads((folder / f"baseline_seed{s}.json").read_text())[
        "reference_reference_w2"] for s in seeds]
    summary = {}
    for method in methods:
        endpoint = [records[(method, s)]["checkpoints"][-1] for s in seeds]
        w2 = np.array([x["w2"] for x in endpoint])
        cost = np.array([x["cost"]["total_seconds"] for x in endpoint])
        gradients = np.array([x["cost"]["operations"].get(
            "observation_gradient_evaluations", 0) for x in endpoint])
        x = np.concatenate([states[(method,s)] for s in seeds], axis=0)
        mean_error = float(np.linalg.norm((x.mean(axis=0)-ref_mean) @ whitening) /
                           np.sqrt(x.shape[1]) + reference["standardized_mean_difference"])
        covariance_error = float(np.linalg.norm(whitening.T @
            (np.cov(x, rowvar=False, ddof=0)-ref_cov) @ whitening) /
            x.shape[1] + reference["standardized_covariance_difference"])
        summary[method] = {"seed_w2": w2, "mean_w2": float(w2.mean()),
            "sd_w2": float(w2.std(ddof=1)),
            "seed_guarded_mean_errors": [x["mean_error_guarded"] for x in endpoint],
            "seed_guarded_covariance_errors": [x["covariance_error_guarded"] for x in endpoint],
            "pooled_guarded_mean_error": mean_error,
            "pooled_guarded_covariance_error": covariance_error,
            "pooled_moment_threshold_pass": bool(max(mean_error,covariance_error) <=
                                                   config["guarded_moment_threshold_each"]),
            "mean_total_seconds": float(cost.mean()),
            "median_total_seconds": float(np.median(cost)),
            "seed_total_seconds": cost,
            "seed_observation_gradients": gradients,
            "mean_observation_gradients": float(gradients.mean())}
    rng = np.random.default_rng(5109)
    contrasts = {}
    for group, pairs in (("primary", config["primary_contrasts"]),
                         ("secondary", config["secondary_contrasts"])):
        for better, comparator in pairs:
            values = np.asarray(summary[better]["seed_w2"]) - np.asarray(
                summary[comparator]["seed_w2"])
            draws = rng.choice(values, (bootstraps, len(values)), replace=True).mean(axis=1)
            contrasts[f"{better}_minus_{comparator}"] = {
                "role": group, "seed_differences": values,
                "mean_w2_difference": float(values.mean()),
                "bootstrap_q025": float(np.quantile(draws, .025)),
                "bootstrap_q975": float(np.quantile(draws, .975)),
                "mean_cost_ratio": summary[better]["mean_total_seconds"] /
                                   summary[comparator]["mean_total_seconds"]}
    result = {"protocol": protocol, "seeds": seeds,
        "reference_baseline_w2": baselines,
        "reference_baseline_mean": float(np.mean(baselines)),
        "reference_guard_mean": reference["standardized_mean_difference"],
        "reference_guard_covariance": reference["standardized_covariance_difference"],
        "methods": summary, "contrasts": contrasts,
        "bootstrap_repetitions": bootstraps,
        "interpretation": "Held-out comparison at fixed K=8192. Bootstrap resamples eight sampler seeds and conditions on the fixed independent HMC pool. It does not certify K_needed or equal-cost superiority."}
    save_json(folder / "summary.json", result)
    reference_word = ("the previously validated" if config["purpose"] ==
                      "heldout_postdiagnostic_confirmation" else "a new validated")
    lines = ["# Held-out 2D logistic comparison at K=8,192", "",
        f"Eight new sampler seeds, 768 independent replicas per seed, and {reference_word} full-data HMC reference with 3,072 scored draws per seed. The endpoint is fixed by protocol and is not a claimed K_needed.", "",
        "| Method | Mean W2 | Pooled guarded mean/cov error | Mean setup-inclusive seconds | Mean observation gradients |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for method in methods:
        x = summary[method]
        multiplier = config.get("step_size_multipliers", {}).get(method, 1.0)
        label = f"{method} (h×{multiplier:g})" if multiplier != 1 else method
        lines.append(f"| {label} | {x['mean_w2']:.4f} | {x['pooled_guarded_mean_error']:.4f} / {x['pooled_guarded_covariance_error']:.4f} | {x['mean_total_seconds']:.1f} | {x['mean_observation_gradients']:.0f} |")
    lines.extend(["", f"Independent HMC-to-HMC empirical W2 baseline: mean {np.mean(baselines):.4f} across the eight evaluation seeds.", "",
        "Paired W2 differences at K=8,192 (negative favors the first method):", "",
        "| Contrast | Role | Mean difference | 95% seed-bootstrap interval | Mean cost ratio |",
        "| --- | --- | ---: | ---: | ---: |"])
    for name, x in contrasts.items():
        lines.append(f"| {name.replace('_minus_', ' − ')} | {x['role']} | {x['mean_w2_difference']:+.4f} | [{x['bootstrap_q025']:+.4f}, {x['bootstrap_q975']:+.4f}] | {x['mean_cost_ratio']:.2f} |")
    lines.extend(["", "The bootstrap is conditional on the fixed HMC pool. Equal K does not imply equal computational cost. Adaptive RMSprop-chain ESS, if reported separately, remains exploratory. Failed or unresolved comparisons must remain in the report.", ""])
    (folder / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path,
                        nargs="?", default=Path("results/logistic_fixed8192_heldout"))
    args = parser.parse_args()
    run(args.folder)


if __name__ == "__main__":
    main()
