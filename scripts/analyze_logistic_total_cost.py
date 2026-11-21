"""Cost/accuracy accounting from saved 2D logistic runs; never reruns samplers."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sgld_cv.workflow import save_json


FORMAL = Path("results/logistic_fixed8192_heldout")
FOLLOWUP = Path("results/logistic_postheldout_replica_sensitivity")
OUTPUT = Path("results/logistic_total_cost_analysis")
METHODS = ("sgld", "cv1", "cv2", "psgld", "pcv1", "pcv2")
SETUP_PHASES = ("anchor_optimization", "cv_anchor_gradient",
                "cv_anchor_hessian", "preconditioner_setup")


def load_case(folder, label, seeds, target_mean, target_cov, whitening, guard):
    records = [json.loads((folder / f"result_{label}_seed{seed}.json").read_text())
               for seed in seeds]
    checkpoints = [int(x["iteration"]) for x in records[0]["checkpoints"]]
    if any([int(x["iteration"]) for x in r["checkpoints"]] != checkpoints for r in records):
        raise ValueError(f"unaligned checkpoints for {label}")
    states = []
    for seed in seeds:
        with np.load(folder / f"states_{label}_seed{seed}.npz") as data:
            if list(data["checkpoints"]) != checkpoints:
                raise ValueError(f"state/checkpoint mismatch for {label}, seed {seed}")
            states.append(data["states"])
    out = []
    for index, k in enumerate(checkpoints):
        items = [r["checkpoints"][index] for r in records]
        costs = [item["cost"] for item in items]
        cloud = np.concatenate([x[index] for x in states])
        d = cloud.shape[1]
        mean = np.linalg.norm((cloud.mean(axis=0) - target_mean) @ whitening) / np.sqrt(d)
        cov = np.linalg.norm(whitening.T @
            (np.cov(cloud, rowvar=False, ddof=0) - target_cov) @ whitening) / d
        w2 = np.asarray([item["w2"] for item in items])
        seconds = np.asarray([c["total_seconds"] for c in costs])
        setup = np.asarray([sum(c["seconds"].get(p, 0) for p in SETUP_PHASES)
                            for c in costs])
        grads = np.asarray([c["operations"].get("observation_gradient_evaluations", 0)
                            for c in costs])
        hess = np.asarray([c["operations"].get("observation_hessian_evaluations", 0)
                           for c in costs])
        row = {"case": label, "iteration": k,
               "mean_w2": float(w2.mean()), "sd_w2": float(w2.std(ddof=1)),
               "mean_total_seconds": float(seconds.mean()),
               "mean_setup_seconds": float(setup.mean()),
               "mean_sampling_seconds": float((seconds - setup).mean()),
               "observation_gradients": int(grads[0]),
               "observation_hessians": int(hess[0]),
               "pooled_guarded_mean_error": float(mean + guard[0]),
               "pooled_guarded_covariance_error": float(cov + guard[1])}
        if not np.all(grads == grads[0]) or not np.all(hess == hess[0]):
            raise ValueError(f"operation counts vary across seeds for {label}, K={k}")
        if any(abs(c["total_seconds"] - sum(c["seconds"].values())) > 1e-6
               for c in costs):
            raise ValueError(f"cost total mismatch for {label}, K={k}")
        out.append(row)
    return out, records


def main():
    formal = json.loads((FORMAL / "summary.json").read_text())
    followup = json.loads((FOLLOWUP / "summary.json").read_text())
    seeds = formal["seeds"]
    if seeds != followup["protocol"]["seeds"]:
        raise ValueError("formal and follow-up seeds differ")
    ref = json.loads((Path(formal["protocol"]["config"]["reference_source"]) /
                      "reference_diagnostics.json").read_text())
    with np.load(Path(formal["protocol"]["config"]["problem_source"]) /
                 "geometry.npz") as data:
        whitening = np.linalg.cholesky(data["hessian"])
    target_mean = np.asarray(ref["reference_mean"])
    target_cov = np.asarray(ref["reference_covariance"])
    guard = (ref["standardized_mean_difference"],
             ref["standardized_covariance_difference"])
    rows = []
    records_by_case = {}
    for folder, labels in ((FORMAL, METHODS),
                           (FOLLOWUP, ("cv1_extended", "cv2_half"))):
        for label in labels:
            points, records = load_case(folder, label, seeds, target_mean,
                                        target_cov, whitening, guard)
            rows.extend(points)
            records_by_case[label] = records
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "accuracy_cost_checkpoints.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    phases = set().union(*(r["checkpoints"][-1]["cost"]["seconds"]
                           for records in records_by_case.values() for r in records))
    setup_rows = []
    for label, records in records_by_case.items():
        endpoints = [r["checkpoints"][-1]["cost"] for r in records]
        row = {"case": label,
               "iteration": int(records[0]["checkpoints"][-1]["iteration"])}
        row.update({f"seconds_{p}": float(np.mean([c["seconds"].get(p, 0)
                                                   for c in endpoints])) for p in sorted(phases)})
        row["total_seconds"] = float(np.mean([c["total_seconds"] for c in endpoints]))
        setup_rows.append(row)
    with (OUTPUT / "phase_breakdown.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(setup_rows[0]))
        writer.writeheader()
        writer.writerows(setup_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), constrained_layout=True)
    colors = dict(zip(METHODS, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for label in (*METHODS, "cv1_extended", "cv2_half"):
        points = [r for r in rows if r["case"] == label and r["iteration"] > 0]
        if label == "cv1_extended":
            points = [r for r in points if r["iteration"] >= 8192]
        color = colors.get(label.replace("_extended", "").replace("_half", ""))
        style = "--" if label.endswith(("_extended", "_half")) else "-"
        marker = "s" if style == "--" else "o"
        name = {"cv1_extended": "CV1 longer K (exploratory)",
                "cv2_half": "CV2 h/2 (exploratory)"}.get(label, label.upper())
        xs = [r["mean_total_seconds"] for r in points]
        axes[0].errorbar(xs, [r["mean_w2"] for r in points],
                         yerr=[r["sd_w2"] / np.sqrt(len(seeds)) for r in points],
                         color=color, linestyle=style, marker=marker, capsize=2,
                         linewidth=1.3, markersize=4, label=name)
        axes[1].plot(xs, [r["pooled_guarded_covariance_error"] for r in points],
                     color=color, linestyle=style, marker=marker,
                     linewidth=1.3, markersize=4)
    axes[0].axhline(formal["reference_baseline_mean"], color="0.4", linestyle=":",
                    label="HMC–HMC finite-sample comparison")
    axes[1].axhline(.1, color="0.4", linestyle=":", label="provisional 0.1 gate")
    axes[0].set_ylabel("Exact empirical W2 (mean ± seed SE)")
    axes[1].set_ylabel("Pooled guarded covariance error")
    for ax in axes:
        ax.set_xlabel("Setup-inclusive ensemble seconds (mean)")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7, ncol=2, loc="best")
    axes[1].legend(fontsize=8)
    fig.savefig(OUTPUT / "accuracy_vs_total_cost.png", dpi=180)
    plt.close(fig)

    endpoint = {r["case"]: r for r in rows if r["iteration"] == 8192}
    # The formal points are authoritative for duplicate CV1 at 8,192.
    endpoint["cv1"] = next(r for r in rows if r["case"] == "cv1" and r["iteration"] == 8192)
    lines = ["# 2D logistic: accuracy versus measured ensemble cost", "",
             "All points come from saved runs on the same eight inspected seeds (500–507); dashed follow-up curves are exploratory. No sampler was rerun for this analysis.", "",
             "| Setting at K=8,192 | Mean W2 | Pooled guarded mean / covariance | Total s | Setup s | Observation gradients | Observation Hessians |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label in (*METHODS, "cv2_half"):
        r = endpoint[label]
        lines.append(f"| {label} | {r['mean_w2']:.4f} | {r['pooled_guarded_mean_error']:.4f} / {r['pooled_guarded_covariance_error']:.4f} | {r['mean_total_seconds']:.1f} | {r['mean_setup_seconds']:.4f} | {r['observation_gradients']:,} | {r['observation_hessians']:,} |")
    lines += ["", "Additional exploratory horizons:", "",
              "| Setting | Mean W2 | Pooled guarded mean / covariance | Total s | Gradients | Hessians |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, k in (("cv1_extended", 16384), ("cv1_extended", 32768),
                     ("cv2_half", 16384)):
        r = next(x for x in rows if x["case"] == label and x["iteration"] == k)
        lines.append(f"| {label}, K={k:,} | {r['mean_w2']:.4f} | {r['pooled_guarded_mean_error']:.4f} / {r['pooled_guarded_covariance_error']:.4f} | {r['mean_total_seconds']:.1f} | {r['observation_gradients']:,} | {r['observation_hessians']:,} |")
    lines += ["", "![Accuracy versus total cost](accuracy_vs_total_cost.png)", "",
              "Cost scope: sequential 768-replica ensemble, one BLAS thread; the ledger charges anchor optimization once per ensemble, CV anchor full gradient, CV2 Hessian construction, and p-method RMSprop pilot once. Sampling includes metric updates and Hessian-vector work. Reference HMC generation, scoring, data generation, and prior exploratory tuning are outside these timers. The setup rows in phase_breakdown.csv expose the measured split.", "",
              "For this N=500, d=2 example, setup seconds are tiny relative to millions of transitions; this is not a general claim about high-dimensional setup cost. Logical observation-gradient counts omit Hessian operations and dense linear algebra; compare them alongside elapsed time and the separate Hessian column. Reusing an anchor or metric across many ensembles would amortize setup, but the table deliberately charges a fresh setup to each ensemble.", "",
              f"The independent HMC–HMC empirical W2 comparison averages {formal['reference_baseline_mean']:.4f}; it indicates finite-sample score resolution, not a lower bound or a threshold. W2 bootstrap uncertainty conditions on the fixed HMC reference. The provisional guarded covariance gate is 0.1. Replica K is not interchangeable with retained draws per diagnostic chain, so these curves do not certify K_needed or the cost of passing R-hat/ESS.", ""]
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines))
    save_json(OUTPUT / "summary.json", {"seeds": seeds, "rows": rows,
        "reference_baseline_mean": formal["reference_baseline_mean"],
        "scope": "retrospective cost analysis; new CV2 h/2 is exploratory"})


if __name__ == "__main__":
    main()
