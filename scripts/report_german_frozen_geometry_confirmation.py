"""Report fresh-seed German geometry/CV-order accuracy, mixing, and total cost."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sgld_cv.transport import load_accuracy_metric, metric_fingerprint
from sgld_cv.workflow import save_json


FOLDER = Path("results/german_frozen_geometry_confirmation")


def main():
    protocol = json.loads((FOLDER / "protocol.json").read_text())
    config = protocol["config"]
    arms, seeds = config["arms"], config["method_seeds"]
    metric = load_accuracy_metric(config["scoring_calibration"], d=61)
    if metric_fingerprint(metric) != protocol["scoring_calibration_sha256"]:
        raise ValueError("German frozen scoring calibration changed")
    refmeta = json.loads((Path(config["reference_source"]) /
                          "reference_diagnostics.json").read_text())
    if not refmeta["reliable"]:
        raise ValueError("German confirmation reference is unreliable")
    target_mean = np.asarray(refmeta["reference_mean"])
    target_cov = np.asarray(refmeta["reference_covariance"])
    score_rows = []
    records, states = {}, {}
    for arm in arms:
        for seed in seeds:
            record = json.loads((FOLDER / f"result_{arm}_seed{seed}.json").read_text())
            if (record["arm"] != arm or record["seed"] != seed or
                    [x["iteration"] for x in record["checkpoints"]] != config["checkpoints"] or
                    record["scoring_calibration_sha256"] != protocol["scoring_calibration_sha256"]):
                raise ValueError(f"invalid German result: {arm}, {seed}")
            with np.load(FOLDER / f"states_{arm}_seed{seed}.npz") as data:
                if list(data["checkpoints"]) != config["checkpoints"]:
                    raise ValueError(f"invalid saved German states: {arm}, {seed}")
                states[(arm, seed)] = data["states"]
            records[(arm, seed)] = record
            for point in record["checkpoints"]:
                score_rows.append({"arm": arm, "seed": seed,
                    "iteration": point["iteration"], "sw2": point["sw2"],
                    "standardized_mean_error": point["standardized_mean_error"],
                    "standardized_covariance_error": point["standardized_covariance_error"],
                    "total_seconds": point["cost"]["total_seconds"],
                    "observation_gradients": point["cost"]["operations"].get(
                        "observation_gradient_evaluations", 0),
                    "observation_hessians": point["cost"]["operations"].get(
                        "observation_hessian_evaluations", 0)})
    with (FOLDER / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)
    summary = {}
    curves = {}
    for arm in arms:
        points = []
        for index, k in enumerate(config["checkpoints"]):
            items = [records[(arm, seed)]["checkpoints"][index] for seed in seeds]
            cloud = np.concatenate([states[(arm, seed)][index] for seed in seeds])
            d = cloud.shape[1]
            mean_error = np.linalg.norm((cloud.mean(axis=0) - target_mean) @
                                        metric.transform) / np.sqrt(d)
            cov_error = np.linalg.norm(metric.transform.T @
                (np.cov(cloud, rowvar=False, ddof=0) - target_cov) @
                metric.transform) / d
            sw2 = np.asarray([x["sw2"] for x in items])
            cost = np.asarray([x["cost"]["total_seconds"] for x in items])
            point = {"iteration": k, "mean_sw2": float(sw2.mean()),
                "sd_sw2": float(sw2.std(ddof=1)), "seed_sw2": sw2,
                "pooled_standardized_mean_error": float(mean_error),
                "pooled_standardized_covariance_error": float(cov_error),
                "mean_total_seconds": float(cost.mean()),
                "mean_observation_gradients": float(np.mean([
                    x["cost"]["operations"].get("observation_gradient_evaluations", 0)
                    for x in items])),
                "mean_observation_hessians": float(np.mean([
                    x["cost"]["operations"].get("observation_hessian_evaluations", 0)
                    for x in items]))}
            points.append(point)
        curves[arm] = points
        final = points[-1]
        summary[arm] = {**final, "step_size": arms[arm]["step_size"],
            "geometry": arms[arm]["geometry"], "method": arms[arm]["method"],
            "pooled_moment_gate_pass": bool(max(final["pooled_standardized_mean_error"],
                                                 final["pooled_standardized_covariance_error"]) <
                                            config["moment_threshold_each"])}
    rng = np.random.default_rng(60309)
    contrasts = {}
    for role, pairs in (("primary", config["primary_contrasts"]),
                        ("secondary", config["secondary_contrasts"])):
        for first, second in pairs:
            difference = (np.asarray(summary[first]["seed_sw2"]) -
                          np.asarray(summary[second]["seed_sw2"]))
            bootstrap = rng.choice(difference, size=(20000, len(difference)),
                                   replace=True).mean(axis=1)
            contrasts[f"{first}_minus_{second}"] = {"role": role,
                "mean_sw2_difference": float(difference.mean()),
                "seed_differences": difference,
                "bootstrap_q025": float(np.quantile(bootstrap, .025)),
                "bootstrap_q975": float(np.quantile(bootstrap, .975)),
                "mean_cost_ratio": summary[first]["mean_total_seconds"] /
                                   summary[second]["mean_total_seconds"]}
    chains = {}
    gate = config["chain_diagnostics"]["gate"]
    for arm in arms:
        chains[arm] = []
        for seed in config["chain_diagnostics"]["seeds"]:
            row = json.loads((FOLDER / f"chain_{arm}_seed{seed}.json").read_text())
            diag = row["diagnostics"]
            row["prespecified_gate_pass"] = bool(
                diag["max_rhat"] < gate["max_rhat_exclusive"] and
                diag["min_bulk_ess"] >= gate["min_bulk_ess"] and
                diag["min_tail_ess"] >= gate["min_tail_ess"])
            chains[arm].append(row)
    baselines = [json.loads((FOLDER / f"baseline_seed{seed}.json").read_text())[
        "reference_reference_sw2"] for seed in seeds]
    result = {"protocol": protocol, "reference_baseline_sw2": baselines,
        "reference_baseline_mean": float(np.mean(baselines)),
        "methods": summary, "curves": curves, "contrasts": contrasts,
        "chain_diagnostics": chains, "bootstrap_repetitions": 20000,
        "interpretation": "Fresh-seed fixed-K and setup-inclusive-cost comparisons. Chain diagnostics are separate runs. Paired bootstrap conditions on the fixed HMC pool. No K_needed claim."}
    save_json(FOLDER / "summary.json", result)

    phase_rows = []
    keys = set().union(*(record["checkpoints"][-1]["cost"]["seconds"]
                         for record in records.values()))
    for arm in arms:
        costs = [records[(arm, seed)]["checkpoints"][-1]["cost"] for seed in seeds]
        row = {"arm": arm}
        row.update({f"seconds_{key}": float(np.mean([
            cost["seconds"].get(key, 0) for cost in costs])) for key in sorted(keys)})
        row["total_seconds"] = summary[arm]["mean_total_seconds"]
        phase_rows.append(row)
    with (FOLDER / "cost_breakdown.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(phase_rows[0]))
        writer.writeheader()
        writer.writerows(phase_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    colors = {"identity_cv1": "#1f77b4", "identity_cv2": "#ff7f0e",
              "full_cv1_h03": "#2ca02c", "full_cv2_h03": "#d62728",
              "full_cv1_h01": "#9467bd"}
    labels = {"identity_cv1": "CV1 · identity",
              "identity_cv2": "CV2 · identity",
              "full_cv1_h03": "CV1 · full H, h=0.3",
              "full_cv2_h03": "CV2 · full H, h=0.3",
              "full_cv1_h01": "CV1 · full H, h=0.1"}
    for arm, points in curves.items():
        values = points[1:]
        x = [p["mean_total_seconds"] for p in values]
        axes[0].errorbar(x, [p["mean_sw2"] for p in values],
            yerr=[p["sd_sw2"] / np.sqrt(len(seeds)) for p in values],
            marker="o", capsize=2, color=colors[arm], label=labels[arm])
        axes[1].plot(x, [p["pooled_standardized_covariance_error"]
                         for p in values], marker="o", color=colors[arm])
    axes[0].axhline(np.mean(baselines), linestyle=":", color="0.4",
                    label="HMC–HMC finite-sample comparison")
    axes[1].axhline(config["moment_threshold_each"], linestyle=":",
                    color="0.4", label="provisional 0.1 moment gate")
    axes[0].set_ylabel("Standardized sliced W2 (mean ± seed SE)")
    axes[1].set_ylabel("Pooled standardized covariance error")
    for ax in axes:
        ax.set_xlabel("Setup-inclusive ensemble seconds (mean)")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=8)
    fig.savefig(FOLDER / "accuracy_vs_total_cost.png", dpi=180)
    plt.close(fig)

    lines = ["# German Credit: frozen geometry/CV-order confirmation", "",
        "Eight fresh paired sampler seeds, 512 independent replicas per seed, K=4,096, batch=32, 1,024 scored draws from a new independently validated full-data HMC reference. All methods use the exact pilot-frozen whitening and 512 sliced projections. This fixed-K endpoint does not assert K_needed.", "",
        "| Arm | Mean SW2 | Pooled standardized mean / covariance | Mean setup-inclusive s | Mean observation gradients | Hessian evaluations |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm, row in summary.items():
        lines.append(f"| {labels[arm]} | {row['mean_sw2']:.4f} | {row['pooled_standardized_mean_error']:.4f} / {row['pooled_standardized_covariance_error']:.4f} | {row['mean_total_seconds']:.1f} | {row['mean_observation_gradients']:,.0f} | {row['mean_observation_hessians']:,.0f} |")
    lines += ["", f"Independent HMC–HMC SW2 mean: {np.mean(baselines):.4f}; this is a resolution comparison, not a lower bound.", "",
        "Paired fixed-K SW2 differences (negative favors the first arm):", "",
        "| Contrast | Role | Mean difference | 95% paired seed-bootstrap interval | Mean time ratio |",
        "| --- | --- | ---: | ---: | ---: |"]
    for name, value in contrasts.items():
        lines.append(f"| {name} | {value['role']} | {value['mean_sw2_difference']:+.4f} | [{value['bootstrap_q025']:+.4f}, {value['bootstrap_q975']:+.4f}] | {value['mean_cost_ratio']:.2f} |")
    lines += ["", "Separate overdispersed four-chain diagnostics:", "",
        "| Arm | Chain seed | Warmup + retained / chain | Max R-hat | Min bulk / tail ESS | Chain seconds | Min bulk ESS/s | Pass frozen gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for arm, items in chains.items():
        for row in items:
            d = row["diagnostics"]
            seconds = row["cost"]["total_seconds"]
            lines.append(f"| {labels[arm]} | {row['seed']} | {row['warmup']:,} + {row['draws_per_chain']:,} | {d['max_rhat']:.4f} | {d['min_bulk_ess']:.0f} / {d['min_tail_ess']:.0f} | {seconds:.2f} | {d['min_bulk_ess'] / seconds:.0f} | {row['prespecified_gate_pass']} |")
    lines += ["", "![German accuracy versus total cost](accuracy_vs_total_cost.png)", "",
        "The h=0.3 full-Hessian pair holds batch, geometry, h and paired random streams fixed, isolating CV order. CV2 h=0.3 versus CV1 h=0.1 compares method-specific usable steps and cost. Identity methods have a different h scale and are interpreted as a calibrated geometry-regime comparison. Full-Hessian setup includes Hessian construction, inverse and factorization; phase details are in cost_breakdown.csv. The anchor is physically computed once by the runner, but its measured optimizer cost is charged to each arm/seed as the standalone setup cost; sharing it across many collections would amortize that component. HMC generation and scoring are evaluation costs outside the samplers. The bootstrap conditions on the fixed HMC pool. Failure to mix within the frozen chain budget remains a reported outcome, not a reason to retune or extend retrospectively.", ""]
    (FOLDER / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
