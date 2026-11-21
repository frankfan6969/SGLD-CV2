"""Summarize frozen fresh-seed comparison plus prespecified longer CV2 check."""
import csv
import json
from pathlib import Path

import numpy as np

from sgld_cv.workflow import save_json


FOLDER = Path("results/logistic_postdiagnostic_confirmation")


def main():
    formal = json.loads((FOLDER / "summary.json").read_text())
    config = formal["protocol"]["config"]
    seeds = config["method_seeds"]
    ref = json.loads((Path(config["reference_source"]) /
                      "reference_diagnostics.json").read_text())
    with np.load(Path(config["problem_source"]) / "geometry.npz") as data:
        whitening = np.linalg.cholesky(data["hessian"])
    x_all, records = [], []
    for seed in seeds:
        record = json.loads((FOLDER / f"result_cv2_long_seed{seed}.json").read_text())
        if [x["iteration"] for x in record["checkpoints"]] != [0, 2048, 4096, 8192, 16384]:
            raise ValueError(f"incorrect CV2 extension checkpoints, seed {seed}")
        with np.load(FOLDER / f"states_cv2_long_seed{seed}.npz") as data:
            if list(data["checkpoints"]) != [0, 2048, 4096, 8192, 16384]:
                raise ValueError(f"incorrect CV2 extension states, seed {seed}")
            x_all.append(data["states"][-1])
        records.append(record["checkpoints"][-1])
    x = np.concatenate(x_all)
    d = x.shape[1]
    target_mean = np.asarray(ref["reference_mean"])
    target_cov = np.asarray(ref["reference_covariance"])
    guarded_mean = float(np.linalg.norm((x.mean(axis=0) - target_mean) @ whitening) /
                         np.sqrt(d) + ref["standardized_mean_difference"])
    guarded_cov = float(np.linalg.norm(whitening.T @
        (np.cov(x, rowvar=False, ddof=0) - target_cov) @ whitening) /
        d + ref["standardized_covariance_difference"])
    w2 = np.array([r["w2"] for r in records])
    seconds = np.array([r["cost"]["total_seconds"] for r in records])
    grads = [r["cost"]["operations"]["observation_gradient_evaluations"] for r in records]
    if len(set(grads)) != 1:
        raise ValueError("CV2 long gradient counts differ across seeds")
    rng = np.random.default_rng(81509)
    delta = w2 - np.asarray(formal["methods"]["cv1"]["seed_w2"])
    draws = rng.choice(delta, size=(20000, len(delta)), replace=True).mean(axis=1)
    long = {"mean_w2": float(w2.mean()), "seed_w2": w2,
            "pooled_guarded_mean_error": guarded_mean,
            "pooled_guarded_covariance_error": guarded_cov,
            "mean_total_seconds": float(seconds.mean()),
            "observation_gradients": grads[0],
            "paired_w2_minus_cv1_k8192": float(delta.mean()),
            "paired_w2_ci95": [float(np.quantile(draws, .025)),
                               float(np.quantile(draws, .975))]}
    save_json(FOLDER / "confirmation_summary.json", {
        "protocol": formal["protocol"], "fixed_k": formal,
        "cv2_half_k16384": long,
        "interpretation": "New sampler seeds confirm only frozen-K comparisons; longer CV2 is a separately budgeted check, not a same-cost ranking or K_needed."})
    with (FOLDER / "confirmation_cost_table.csv").open("w", newline="") as stream:
        fields = ["method", "K", "mean_w2", "pooled_guarded_mean_error",
                  "pooled_guarded_covariance_error", "mean_total_seconds",
                  "mean_observation_gradients"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method in config["methods"]:
            row = formal["methods"][method]
            writer.writerow(dict(method=method, K=8192, mean_w2=row["mean_w2"],
                pooled_guarded_mean_error=row["pooled_guarded_mean_error"],
                pooled_guarded_covariance_error=row["pooled_guarded_covariance_error"],
                mean_total_seconds=row["mean_total_seconds"],
                mean_observation_gradients=row["mean_observation_gradients"]))
        writer.writerow(dict(method="cv2_half_long", K=16384,
            mean_w2=long["mean_w2"], pooled_guarded_mean_error=guarded_mean,
            pooled_guarded_covariance_error=guarded_cov,
            mean_total_seconds=long["mean_total_seconds"],
            mean_observation_gradients=long["observation_gradients"]))
    lines = ["# Fresh-seed 2D logistic confirmation", "",
        "Frozen six-method comparison on eight new sampler seeds. CV2 alone uses the post-diagnostic half step; the other five settings are unchanged. The validated fixed HMC reference and pilot geometry are reused, and all W2 uncertainty is conditional on that reference.", "",
        "| Setting | Mean W2 | Pooled guarded mean / covariance | Mean setup-inclusive s | Observation gradients |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for method in config["methods"]:
        r = formal["methods"][method]
        label = "cv2 (h/2)" if method == "cv2" else method
        lines.append(f"| {label}, K=8,192 | {r['mean_w2']:.4f} | {r['pooled_guarded_mean_error']:.4f} / {r['pooled_guarded_covariance_error']:.4f} | {r['mean_total_seconds']:.1f} | {r['mean_observation_gradients']:,.0f} |")
    lines.append(f"| CV2 h/2, K=16,384 (extra-cost) | {long['mean_w2']:.4f} | {guarded_mean:.4f} / {guarded_cov:.4f} | {long['mean_total_seconds']:.1f} | {long['observation_gradients']:,} |")
    lines += ["", "Primary fixed-K paired contrasts (negative favors first method):", "",
              "| Contrast | Mean W2 difference | 95% paired seed-bootstrap interval | Mean cost ratio |",
              "| --- | ---: | ---: | ---: |"]
    for name in ("cv2_minus_cv1", "pcv2_minus_pcv1"):
        contrast = formal["contrasts"][name]
        lines.append(f"| {name} | {contrast['mean_w2_difference']:+.4f} | [{contrast['bootstrap_q025']:+.4f}, {contrast['bootstrap_q975']:+.4f}] | {contrast['mean_cost_ratio']:.2f} |")
    lines += ["", f"Extra-cost CV2 h/2 K=16,384 minus CV1 K=8,192: W2 {long['paired_w2_minus_cv1_k8192']:+.4f}, 95% interval [{long['paired_w2_ci95'][0]:+.4f}, {long['paired_w2_ci95'][1]:+.4f}]. This is not an equal-budget contrast.", "",
              f"Reference HMC–HMC empirical W2 mean: {formal['reference_baseline_mean']:.4f}, a finite-sample comparison rather than a lower bound. The guarded moment gate is 0.1 per component. No K_needed is asserted, and R-hat/ESS evidence comes from separate exploratory multichain runs.", ""]
    (FOLDER / "CONFIRMATION.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
