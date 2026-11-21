"""Measure German Credit scoring noise from two validated HMC reference pools."""
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.transport import load_accuracy_metric, metric_fingerprint
from sgld_cv.workflow import save_json


PILOT = Path("results/german_sliced_pilot")
REFERENCE = Path("results/german_sliced_heldout_long_reference")
OUTPUT = Path("results/german_targeted_calibration")
SIZES = (128, 256, 512, 1024)
REPEATS = 12
SEED = 97531


def main():
    diagnostics = json.loads((REFERENCE / "reference_diagnostics.json").read_text())
    if not diagnostics["reliable"]:
        raise ValueError("German HMC reference is not validated")
    with np.load(REFERENCE / "reference_chains.npz") as data:
        first = data["first"].reshape(-1, data["first"].shape[-1])
        longer = data["longer"].reshape(-1, data["longer"].shape[-1])
    metric = load_accuracy_metric(PILOT / "accuracy_calibration.npz", d=first.shape[1])
    frozen = json.loads((PILOT / "frozen_selection.json").read_text())
    fingerprint = metric_fingerprint(metric)
    if fingerprint != frozen["accuracy_metric"]["calibration_sha256"]:
        raise ValueError("frozen German scoring calibration changed")
    rng = np.random.default_rng(SEED)
    rows = []
    d = first.shape[1]
    with threadpool_limits(limits=1):
        for n in SIZES:
            for repeat in range(REPEATS):
                a = first[rng.choice(len(first), n, replace=False)]
                b = longer[rng.choice(len(longer), n, replace=False)]
                mean = np.linalg.norm((a.mean(axis=0) - b.mean(axis=0)) @ metric.transform) / np.sqrt(d)
                cov = np.linalg.norm(metric.transform.T @
                    (np.cov(a, rowvar=False, ddof=0) - np.cov(b, rowvar=False, ddof=0)) @
                    metric.transform) / d
                rows.append({"sample_size_each": n, "repeat": repeat,
                             "reference_reference_sw2": metric.distance(a, b),
                             "standardized_mean_difference": float(mean),
                             "standardized_covariance_difference": float(cov)})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "reference_resolution.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for n in SIZES:
        subset = [r for r in rows if r["sample_size_each"] == n]
        summary[str(n)] = {}
        for key in ("reference_reference_sw2", "standardized_mean_difference",
                    "standardized_covariance_difference"):
            values = np.array([r[key] for r in subset])
            summary[str(n)][key] = {"median": float(np.median(values)),
                                    "q25": float(np.quantile(values, .25)),
                                    "q75": float(np.quantile(values, .75)),
                                    "range": [float(values.min()), float(values.max())]}
    save_json(OUTPUT / "reference_resolution.json", {
        "source_reference": str(REFERENCE), "source_calibration": str(PILOT),
        "calibration_sha256": fingerprint, "reference_pool_sizes": [len(first), len(longer)],
        "reference_reliable": True, "seed": SEED, "repeats": REPEATS,
        "summary": summary,
        "interpretation": "Descriptive HMC-to-HMC sample resolution. HMC draws are correlated; these repeated subsamples are not independent experiments or a SW2 lower bound. Sampler replica resolution must be checked separately."})
    lines = ["# German Credit score-resolution check", "",
        "The pilot's frozen 512-projection whitening and the independently validated longer held-out HMC pools were reused. The two HMC pools are approximate and internally correlated; numbers describe score sensitivity rather than a precision guarantee.", "",
        "| Draws per HMC side | SW2 median [IQR] | Standardized mean difference | Standardized covariance difference |",
        "| ---: | ---: | ---: | ---: |"]
    for n in SIZES:
        row = summary[str(n)]
        def fmt(key):
            value = row[key]
            return f"{value['median']:.4f} [{value['q25']:.4f}, {value['q75']:.4f}]"
        lines.append(f"| {n:,} | {fmt('reference_reference_sw2')} | {fmt('standardized_mean_difference')} | {fmt('standardized_covariance_difference')} |")
    lines += ["", "The HMC-to-HMC SW2 value is not a lower bound on sampler-to-reference SW2. This check varies both sample counts together; a separate sampler-replica study is still needed before choosing final R and reference S.", ""]
    (OUTPUT / "REFERENCE_RESOLUTION.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
