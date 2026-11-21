"""Report the frozen Gaussian 2x2 without treating analytic CV2 as CV1 truth."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sgld_cv.workflow import save_json


FOLDER = Path("results/gaussian_compact_geometry_confirmation")


def cluster_interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(260930)
    draws = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "q025": float(np.quantile(draws, .025)),
            "q975": float(np.quantile(draws, .975)), "design_seed_values": values.tolist()}


def main():
    protocol = json.loads((FOLDER / "protocol.json").read_text())
    c = protocol["config"]
    cells, analytic = {}, {}
    for d in c["dimensions"]:
        for kappa in c["target_condition_numbers"]:
            cell = f"d{d}_k{kappa}"
            rows, baselines, targets = [], [], []
            for design_seed in c["design_seeds"]:
                target_id = f"{cell}_design{design_seed}"
                targets.append(json.loads((FOLDER / f"target_{target_id}.json").read_text()))
                for sampler_seed in c["sampler_seeds"]:
                    baselines.append(json.loads((FOLDER /
                        f"baseline_{target_id}_seed{sampler_seed}.json").read_text()))
                    for method in c["methods"]:
                        rows.append(json.loads((FOLDER /
                            f"result_{target_id}_{method}_seed{sampler_seed}.json").read_text()))
            analytic[cell] = {
                "identity_cv2_w2_over_sqrt_d_at_k4096": targets[0][
                    "identity_cv2_exact_w2_over_sqrt_d_from_mode"][-1],
                "full_cv2_w2_over_sqrt_d_at_k4096": targets[0][
                    "full_cv2_exact_w2_over_sqrt_d_from_mode"][-1],
                "achieved_condition_numbers": [t["target"]["actual_kappa"] for t in targets]}
            methods = {}
            for method in c["methods"]:
                subset = [r for r in rows if r["method"] == method]
                methods[method] = {
                    "median_sw2": float(np.median([r["standardized_sw2_512_vs_1024"] for r in subset])),
                    "median_mean_error": float(np.median([r["standardized_mean_error"] for r in subset])),
                    "median_covariance_error": float(np.median([r["standardized_covariance_error"] for r in subset])),
                    "median_total_seconds": float(np.median([r["cost"]["total_seconds"] for r in subset])),
                    "max_rhat": float(max(r["max_rhat"] for r in subset)),
                    "min_bulk_ess": float(min(r["min_bulk_ess"] for r in subset)),
                    "mixing_passes": int(sum(r["passes_mixing_gate"] for r in subset)),
                    "runs": len(subset),
                    "observation_gradients_per_run": subset[0]["cost"]["operations"].get(
                        "observation_gradient_evaluations", 0),
                    "observation_hessians_per_run": subset[0]["cost"]["operations"].get(
                        "observation_hessian_evaluations", 0)}
            by_key = {(r["design_seed"], r["sampler_seed"], r["method"]): r for r in rows}
            paired = {}
            for measure in ("standardized_sw2_512_vs_1024", "standardized_mean_error",
                            "standardized_covariance_error"):
                design_diffs = []
                for ds in c["design_seeds"]:
                    design_diffs.append(np.mean([
                        by_key[(ds, ss, "cv2")][measure] - by_key[(ds, ss, "cv1")][measure]
                        for ss in c["sampler_seeds"]]))
                paired[measure] = cluster_interval(design_diffs)
            cells[cell] = {"d": d, "kappa": kappa, "methods": methods,
                           "exact_vs_exact_sw2_median": float(np.median([
                               b["exact_vs_exact_sw2"] for b in baselines])),
                           "cv2_minus_cv1_by_design_seed": paired}
    dimension = {}
    for kappa in c["target_condition_numbers"]:
        low = cells[f"d{c['dimensions'][0]}_k{kappa}"]["cv2_minus_cv1_by_design_seed"]
        high = cells[f"d{c['dimensions'][1]}_k{kappa}"]["cv2_minus_cv1_by_design_seed"]
        dimension[str(kappa)] = {}
        for measure in ("standardized_sw2_512_vs_1024", "standardized_covariance_error"):
            dimension[str(kappa)][measure] = cluster_interval(np.asarray(high[measure]["design_seed_values"]) -
                                                              np.asarray(low[measure]["design_seed_values"]))
    summary = {"cells": cells, "analytic_controls": analytic,
               "dimension_change_in_cv2_minus_cv1": dimension,
               "bootstrap_unit": "five independent finite-sum design seeds; four paired sampler seeds averaged within each design",
               "limits": "Gaussian CV2 is exact-gradient; analytic finite-k W2 does not apply to stochastic CV1. Identity analytic path starts at the mode; empirical full-metric chains start overdispersed. SW2 includes a finite-sample exact-vs-exact baseline."}
    save_json(FOLDER / "summary.json", summary)
    lines = ["# Compact Gaussian 2×2 geometry confirmation", "",
        "Four quadratic finite-sum cells, $N=1000$, batch 32, full inverse-precision geometry, matched $h=0.3$, five held-out design seeds and four paired sampler seeds per design. Each run has four chains, 1,000 warmup and 4,000 retained draws per chain. The posterior is exactly Gaussian; SW2 uses exact-covariance whitening and 512 frozen projections, with 512 spaced sampler draws versus 1,024 independent exact draws. All costs include fresh CV and metric setup plus four-chain sampling.", "",
        "| $d$ | $\\kappa$ | Method | SW2 median | Mean error | Cov error | Total s | Worst Rhat | Min bulk ESS | Gate |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d in c["dimensions"]:
        for kappa in c["target_condition_numbers"]:
            cell = cells[f"d{d}_k{kappa}"]
            for method in c["methods"]:
                s = cell["methods"][method]
                lines.append(f"| {d} | {kappa:,} | {method.upper()} | {s['median_sw2']:.4f} | {s['median_mean_error']:.4f} | {s['median_covariance_error']:.4f} | {s['median_total_seconds']:.2f} | {s['max_rhat']:.4f} | {s['min_bulk_ess']:.0f} | {s['mixing_passes']}/{s['runs']} |")
    lines += ["", "Paired CV2−CV1 differences below average the four sampler seeds within each design, then bootstrap the five independent design seeds. Negative favors CV2. These narrow conditional intervals do not establish a universal dimension law outside this Gaussian construction.", "",
        "| $d$ | $\\kappa$ | $\\Delta$ SW2 [95%] | $\\Delta$ covariance error [95%] | Exact-vs-exact SW2 | Analytic identity/full CV2 $W_2/\\sqrt d$ at $K=4096$ |",
        "|---:|---:|---:|---:|---:|---:|"]
    for d in c["dimensions"]:
        for kappa in c["target_condition_numbers"]:
            cell = f"d{d}_k{kappa}"
            x = cells[cell]
            sw = x["cv2_minus_cv1_by_design_seed"]["standardized_sw2_512_vs_1024"]
            cov = x["cv2_minus_cv1_by_design_seed"]["standardized_covariance_error"]
            a = analytic[cell]
            lines.append(f"| {d} | {kappa:,} | {sw['mean']:+.4f} [{sw['q025']:+.4f}, {sw['q975']:+.4f}] | {cov['mean']:+.4f} [{cov['q025']:+.4f}, {cov['q975']:+.4f}] | {x['exact_vs_exact_sw2_median']:.4f} | {a['identity_cv2_w2_over_sqrt_d_at_k4096']:.4f} / {a['full_cv2_w2_over_sqrt_d_at_k4096']:.4f} |")
    lines += ["", "The analytic identity/full CV2 entries are exact Gaussian $W_2$ from the mode under different metrics; they are not empirical sliced SW2 and must not be subtracted from the SW2 columns. The identity step is set to $1/\\lambda_{\\max}(Q)$; the full-metric step is 0.3. Because the full correction whitens this constructed target, its exact-gradient behavior is nearly independent of $\\kappa$ by design. CV1's stochastic gradient does not inherit the analytic Gaussian transition law.", "",
        "![Controlled Gaussian geometry and CV-order mechanism](mechanism.png)", ""]
    (FOLDER / "RESULTS.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), constrained_layout=True)
    labels = [f"d={d}, κ={kappa:g}" for d in c["dimensions"] for kappa in c["target_condition_numbers"]]
    keys = [f"d{d}_k{kappa}" for d in c["dimensions"] for kappa in c["target_condition_numbers"]]
    x = np.arange(len(keys))
    axes[0, 0].bar(x - .17, [analytic[k]["identity_cv2_w2_over_sqrt_d_at_k4096"] for k in keys],
                   width=.34, label="identity CV2", color="#aa6248")
    axes[0, 0].bar(x + .17, [analytic[k]["full_cv2_w2_over_sqrt_d_at_k4096"] for k in keys],
                   width=.34, label="full-metric CV2", color="#307a9a")
    axes[0, 0].set(ylabel="Exact Gaussian W2 / √d", title="Geometry: analytic CV2 at K=4096")
    axes[0, 0].legend(frameon=False)
    for method, offset, color in [("cv1", -.11, "#b9603e"), ("cv2", .11, "#287f9e")]:
        axes[0, 1].scatter(x + offset, [cells[k]["methods"][method]["median_sw2"] for k in keys],
                           label=method.upper(), color=color, s=62)
    axes[0, 1].scatter(x, [cells[k]["exact_vs_exact_sw2_median"] for k in keys],
                       color="#555555", marker="_", s=270, label="exact-vs-exact")
    axes[0, 1].set(ylabel="Standardized sliced W2", title="Full geometry: posterior accuracy")
    axes[0, 1].legend(frameon=False)
    for j, measure in enumerate(("standardized_sw2_512_vs_1024", "standardized_covariance_error")):
        ax = axes[1, j]
        for i, key in enumerate(keys):
            values = cells[key]["cv2_minus_cv1_by_design_seed"][measure]["design_seed_values"]
            ax.scatter(np.full(len(values), i), values, alpha=.5, color="#6355a6", s=25)
            ax.scatter(i, np.mean(values), marker="x", color="#231b67", s=105, linewidths=2.4)
        ax.axhline(0, color="#777777", linewidth=1)
        ax.set(ylabel="CV2 − CV1", title="Paired design-level " + ("SW2 difference" if j == 0 else "covariance-error difference"))
    for ax in axes.flat:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=.18)
    fig.savefig(FOLDER / "mechanism.png", dpi=180)
    plt.close(fig)
    print("wrote", FOLDER / "RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
