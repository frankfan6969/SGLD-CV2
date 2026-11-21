"""Reassess saved reference resolution and slow-direction pilot geometry.

These are exploratory diagnostics. They do not tune or certify a sampler.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.transport import make_accuracy_metric, load_accuracy_metric, metric_fingerprint
from sgld_cv.workflow import METHODS, save_csv, save_json


def reference_resolution(folder, output, *, seed=913, repeats=8,
                         sizes=(24, 48, 96, 192, 384)):
    folder, output = Path(folder), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(folder / "reference_chains.npz") as data:
        first = data["first"].reshape(-1, data["first"].shape[-1])
        longer = data["longer"].reshape(-1, data["longer"].shape[-1])
    d = first.shape[1]
    saved = folder / "accuracy_calibration.npz"
    if saved.exists():
        metric = load_accuracy_metric(saved, d=d)
        source = "saved pilot calibration"
    else:
        metric = make_accuracy_metric(d, kind="standardized_sliced_w2",
            center=first.mean(axis=0), covariance=np.cov(first, rowvar=False),
            projections=512, seed=926)
        source = "reconstructed from saved first reference chains; exploratory"
    np.savez_compressed(output / "accuracy_calibration.npz", center=metric.center,
                        transform=metric.transform, projections=metric.projections)
    rng = np.random.default_rng(seed)
    anchor = np.load(folder / "geometry.npz")["anchor"]
    manifest = json.loads((folder / "manifest.json").read_text())
    evaluation_seed = manifest["config"]["evaluation"]["seeds"][0]
    endpoints = {}
    for method in METHODS:
        with np.load(folder / f"replicas_{method}_seed{evaluation_seed}.npz") as data:
            endpoints[method] = data["states"][-1]
    rows = []
    with threadpool_limits(limits=1):
        for n in sizes:
            if n > min(len(first), len(longer)):
                raise ValueError("requested samples exceed independent reference pools")
            for repeat in range(repeats):
                a = first[rng.choice(len(first), n, replace=False)]
                b = longer[rng.choice(len(longer), n, replace=False)]
                rows.append({"sample_size": n, "repeat": repeat, "comparison": "reference vs reference",
                             "accuracy": metric.distance(a, b)})
                rows.append({"sample_size": n, "repeat": repeat, "comparison": "point mass vs reference",
                             "accuracy": metric.distance(np.broadcast_to(anchor, (n, d)), b)})
                for method, endpoint in endpoints.items():
                    rows.append({"sample_size": n, "repeat": repeat,
                                 "comparison": f"{method} endpoint ({len(endpoint)} replicas) vs reference",
                                 "accuracy": metric.distance(endpoint, b)})
    save_csv(output / "resolution.csv", rows)
    summary = {}
    for n in sizes:
        summary[str(n)] = {}
        for comparison in sorted({r["comparison"] for r in rows}):
            values = np.array([r["accuracy"] for r in rows if r["sample_size"] == n
                               and r["comparison"] == comparison])
            summary[str(n)][comparison] = {"median": float(np.median(values)),
                "q25": float(np.quantile(values, .25)), "q75": float(np.quantile(values, .75))}
    save_json(output / "resolution_summary.json", {"metric": metric.kind,
        "calibration_source": source, "calibration_sha256": metric_fingerprint(metric),
        "reference_pool_sizes": [len(first), len(longer)],
        "repeats": repeats, "seed": seed, "evaluation_seed": evaluation_seed,
        "endpoint_replicas": len(next(iter(endpoints.values()))), "summary": summary,
        "interpretation": "Descriptive finite-sample sensitivity on one target. Reference pools are HMC draws, not IID samples. Increasing the reference side alone cannot determine the required sampler replica count."})


def slow_direction_geometry(folders, output):
    rows = []
    for folder in map(Path, folders):
        with np.load(folder / "geometry.npz") as data:
            H = data["hessian"]
        frozen = json.loads((folder / "frozen_selection.json").read_text())
        pilots = json.loads((folder / "metric_pilots.json").read_text())
        for method, setting in frozen["selection"].items():
            if method.startswith("p"):
                diagonal = np.asarray(pilots[str(setting["batch_size"])]["diagonal"])
            else:
                diagonal = np.ones(len(H))
            eig = np.linalg.eigvalsh(np.sqrt(diagonal)[:, None] * H * np.sqrt(diagonal)[None, :])
            h = setting["step_size"]
            rows.append({"dataset": folder.name, "method": method,
                "batch_size": setting["batch_size"], "step_size": h,
                "effective_min_curvature": float(eig[0]),
                "effective_max_curvature": float(eig[-1]),
                "slow_direction_drift_per_step": float(h * eig[0] / 2),
                "linearized_efold_steps": float(-1 / np.log1p(-h * eig[0] / 2))
                    if 0 < h * eig[0] < 2 else None,
                "note": "Local frozen-metric linear approximation; adaptive RMSprop and stochastic gradients can differ."})
    save_csv(output, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--german", type=Path,
                        default=Path("results/stage456_validation/german_credit"))
    parser.add_argument("--gaussian", type=Path,
                        default=Path("results/stage456_validation/gaussian"))
    parser.add_argument("--logistic", type=Path,
                        default=Path("results/stage456_validation/synthetic_logistic"))
    parser.add_argument("--year", type=Path,
                        default=Path("results/stage456_validation/year_ridge"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference_resolution(args.german, args.output)
    slow_direction_geometry([args.gaussian, args.logistic, args.german, args.year],
                            args.output / "slow_direction_geometry.csv")


if __name__ == "__main__":
    main()
