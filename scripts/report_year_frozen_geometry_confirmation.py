"""Summarize the frozen Year ridge study without retuning its protocol."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sgld_cv.workflow import save_json


FOLDER = Path("results/year_frozen_geometry_confirmation")


def interval(differences):
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(260929)
    samples = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    return [float(values.mean()), *map(float, np.quantile(samples, [.025, .975]))]


def main():
    protocol = json.loads((FOLDER / "protocol.json").read_text())
    config = protocol["config"]
    seeds = config["method_seeds"]
    rows = {arm: [json.loads((FOLDER / f"result_{arm}_seed{seed}.json").read_text())
                  for seed in seeds] for arm in config["arms"]}
    baselines = [json.loads((FOLDER / f"direct_gaussian_16000_seed{seed}.json").read_text())
                 for seed in seeds]
    names = ["identity_cv1", "identity_cv2", "full_cv1_h01", "full_cv2_h03", "full_cv2_h06"]
    stats = {}
    for arm in names:
        observations = rows[arm]
        stats[arm] = {
            "median_sw2": float(np.median([r["standardized_sw2_512_vs_1024"] for r in observations])),
            "median_mean_error": float(np.median([r["standardized_mean_error"] for r in observations])),
            "median_covariance_error": float(np.median([r["standardized_covariance_error"] for r in observations])),
            "median_total_seconds": float(np.median([r["cost"]["total_seconds"] for r in observations])),
            "max_rhat_across_seeds": float(max(r["diagnostics"]["max_rhat"] for r in observations)),
            "min_bulk_ess_across_seeds": float(min(r["diagnostics"]["min_bulk_ess"] for r in observations)),
            "passes_rhat_1_01_and_ess_400": int(sum(r["diagnostics"]["max_rhat"] < 1.01 and
                r["diagnostics"]["min_bulk_ess"] >= 400 for r in observations)),
            "observation_gradient_evaluations_per_seed": observations[0]["cost"]["operations"].get(
                "observation_gradient_evaluations", 0),
            "data_row_visits_per_seed": observations[0]["cost"]["operations"].get("data_row_visits", 0),
            "observation_hessian_evaluations_per_seed": observations[0]["cost"]["operations"].get(
                "observation_hessian_evaluations", 0)}
    paired = {}
    for left, right in [("full_cv2_h03", "full_cv1_h01"),
                        ("full_cv2_h06", "full_cv2_h03"),
                        ("identity_cv2", "identity_cv1")]:
        label = f"{left} minus {right}"
        paired[label] = {}
        for key in ("standardized_sw2_512_vs_1024", "standardized_mean_error",
                    "standardized_covariance_error"):
            paired[label][key] = interval([a[key] - b[key] for a, b in zip(rows[left], rows[right])])
        paired[label]["total_seconds"] = interval([a["cost"]["total_seconds"] -
                                                     b["cost"]["total_seconds"]
                                                     for a, b in zip(rows[left], rows[right])])
    exact = {"median_sw2": float(np.median([r["exact_vs_exact_sw2"] for r in baselines])),
             "median_mean_error": float(np.median([r["standardized_mean_error"] for r in baselines])),
             "median_covariance_error": float(np.median([r["standardized_covariance_error"] for r in baselines])),
             "median_total_seconds": float(np.median([r["cost"]["total_seconds"] for r in baselines])),
             "data_row_visits_per_seed": baselines[0]["cost"]["operations"]["data_row_visits"]}
    summary = {"seeds": seeds, "metric": "frozen standardized sliced W2 (512 scored / 1024 exact reference)",
               "arms": stats, "direct_gaussian": exact, "paired_mean_and_bootstrap_95_percent_ci": paired,
               "interpretation_limit": "Identity arms are budget-censored when Rhat/ESS fail. Direct exact Gaussian draws are a special analytic-target benchmark. SW2 has a nonzero finite-sample baseline."}
    save_json(FOLDER / "summary.json", summary)
    invariants = json.loads((Path(config["calibration_source"]) / "invariants.json").read_text())
    lines = ["# YearMSD frozen geometry confirmation", "",
        f"Eight fresh paired seeds ({seeds[0]}–{seeds[-1]}); exact Gaussian posterior reference; fixed whitening and 512 projections. Each arm uses four overdispersed chains. SW2 scores 128 evenly spaced retained draws per chain against 1,024 independent exact draws.", "",
        f"Quadratic CV2 gradient agrees with the exact full gradient to relative error at most {invariants['max_relative_cv2_gradient_error']:.2e} in calibration checks. The posterior precision condition number is {invariants['precision_condition_number']:.2e}; at the frozen identity step size, its slowest mean direction needs about {invariants['identity_cv2_slowest_efold_steps']:.2e} transitions per e-fold. These are analytic/model checks, not evidence that a finite identity chain mixed.", "",
        "| Arm | SW2 median | Mean error | Covariance error | Total sec | Worst Rhat | Min bulk ESS | Passing seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in names:
        s = stats[arm]
        lines.append(f"| {arm} | {s['median_sw2']:.4f} | {s['median_mean_error']:.4f} | {s['median_covariance_error']:.4f} | {s['median_total_seconds']:.2f} | {s['max_rhat_across_seeds']:.4f} | {s['min_bulk_ess_across_seeds']:.0f} | {s['passes_rhat_1_01_and_ess_400']}/8 |")
    lines += [f"| Direct exact Gaussian draws | {exact['median_sw2']:.4f} | {exact['median_mean_error']:.4f} | {exact['median_covariance_error']:.4f} | {exact['median_total_seconds']:.2f} | — | independent | — |", "",
        "Costs include the full-data sufficient-statistic anchor, CV cache construction, full inverse and factorization when used, and all four chains. Model loading and scoring are excluded equally. Direct exact draws include sufficient statistics and 16,000 independent Gaussian samples (512 scored). Full-H CV2 also retains 16,000 samples; full-H CV1 retains 80,000 because of its longer mixing run. Operation counts are in `summary.json`.", "",
        "Paired differences are first arm minus second; intervals are seed-paired bootstrap 95% intervals, with only eight seeds:", ""]
    for pair, measures in paired.items():
        sw = measures["standardized_sw2_512_vs_1024"]
        cov = measures["standardized_covariance_error"]
        sec = measures["total_seconds"]
        lines.append(f"- {pair}: SW2 {sw[0]:+.4f} [{sw[1]:+.4f}, {sw[2]:+.4f}]; covariance error {cov[0]:+.4f} [{cov[1]:+.4f}, {cov[2]:+.4f}]; total seconds {sec[0]:+.2f} [{sec[1]:+.2f}, {sec[2]:+.2f}].")
    lines += ["", "Identity arms should be read as budget-censored controls if their multichain diagnostics fail. The exact Gaussian baseline shows the sampling noise floor and what direct posterior sampling costs when this model admits it. Matching or approaching its SW2 is not proof of equal distributions; inspect moments and mixing together.", ""]
    lines += [f"For full-metric CV2, exact Gaussian ULA theory predicts stationary variance inflation of {100*invariants['full_cv2_stationary_variance_inflation']['0.3']:.1f}% at h=0.3 and {100*invariants['full_cv2_stationary_variance_inflation']['0.6']:.1f}% at h=0.6. Faster mixing at h=0.6 therefore does not ensure higher posterior accuracy. CV1 has a slightly smaller mean error than CV2 at h=0.3 after its much longer run, so the observed advantage is specific to SW2, covariance error, and cost.", "",
        f"Per seed, full-H CV1 h=0.1 uses {stats['full_cv1_h01']['observation_gradient_evaluations_per_seed']:,} observation gradients; full-H CV2 h=0.3 uses {stats['full_cv2_h03']['observation_gradient_evaluations_per_seed']:,} gradients plus {stats['full_cv2_h03']['observation_hessian_evaluations_per_seed']:,} observation Hessian contributions. Direct Gaussian sampling visits {exact['data_row_visits_per_seed']:,} rows once for sufficient statistics. Gradient counts, Hessian work, and row visits are separate units, not converted into an artificial common FLOP estimate.", ""]
    (FOLDER / "RESULTS.md").write_text("\n".join(lines))
    fig, ax = plt.subplots(figsize=(7.4, 5.1), constrained_layout=True)
    colors = {"full_cv1_h01": "#be5a3c", "full_cv2_h03": "#207aa8",
              "full_cv2_h06": "#6c55a5"}
    for arm in ("full_cv1_h01", "full_cv2_h03", "full_cv2_h06"):
        xs = [r["cost"]["total_seconds"] for r in rows[arm]]
        ys = [r["standardized_sw2_512_vs_1024"] for r in rows[arm]]
        ax.scatter(xs, ys, label=arm, color=colors[arm], alpha=.8, s=34)
        ax.scatter(np.median(xs), np.median(ys), color=colors[arm], s=130, marker="x", linewidths=2.5)
    ax.scatter([r["cost"]["total_seconds"] for r in baselines],
               [r["exact_vs_exact_sw2"] for r in baselines],
               label="direct exact Gaussian", color="#333333", marker="D", s=29, alpha=.65)
    ax.set(xlabel="Setup-inclusive seconds per run", ylabel="Standardized sliced W2",
           title="YearMSD: accuracy versus measured total cost")
    ax.grid(alpha=.22)
    ax.legend(frameon=False)
    fig.savefig(FOLDER / "accuracy_vs_total_cost.png", dpi=180)
    plt.close(fig)
    print("wrote", FOLDER / "RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
