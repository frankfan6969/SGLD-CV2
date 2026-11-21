"""Measure scoring sample-size sensitivity against an exact ridge posterior."""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.transport import make_accuracy_metric, metric_fingerprint
from sgld_cv.workflow import METHODS, save_csv, save_json


def diagnose(pilot, output, *, sizes=(16, 64, 256, 1024), repeats=8, seed=913):
    pilot, output = Path(pilot), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(pilot / "geometry.npz") as data:
        mean, precision = data["anchor"], data["hessian"]
    d = len(mean)
    covariance = np.linalg.solve(precision, np.eye(d))
    metric = make_accuracy_metric(d, kind="standardized_sliced_w2", center=mean,
                                  covariance=covariance, projections=512, seed=926)
    np.savez_compressed(output / "accuracy_calibration.npz", center=metric.center,
                        transform=metric.transform, projections=metric.projections)
    chol = np.linalg.cholesky(covariance)
    rng = np.random.default_rng(seed)
    endpoints = {}
    for method in METHODS:
        with np.load(pilot / f"replicas_{method}_seed300.npz") as data:
            endpoints[method] = data["states"][-1]
    rows = []
    with threadpool_limits(limits=1):
        for n in sizes:
            for repeat in range(repeats):
                a = mean + rng.normal(size=(n, d)) @ chol.T
                b = mean + rng.normal(size=(n, d)) @ chol.T
                clouds = {"exact vs exact": a, "point mass vs exact": np.broadcast_to(mean, (n, d))}
                clouds.update({f"{m} endpoint (16 replicas) vs exact": endpoint
                               for m, endpoint in endpoints.items()})
                for label, x in clouds.items():
                    rows.append({"reference_size": n, "repeat": repeat,
                                 "comparison": label, "accuracy": metric.distance(x, b)})
    save_csv(output / "resolution.csv", rows)
    summary = {}
    for n in sizes:
        summary[str(n)] = {label: {"median": float(np.median(values)),
            "q25": float(np.quantile(values, .25)), "q75": float(np.quantile(values, .75))}
            for label in sorted({r["comparison"] for r in rows})
            for values in [[r["accuracy"] for r in rows if r["reference_size"] == n
                            and r["comparison"] == label]]}
    save_json(output / "resolution_summary.json", {"metric": metric.kind,
        "target": "analytic Gaussian ridge posterior", "dimension": d,
        "calibration_sha256": metric_fingerprint(metric), "seed": seed,
        "repeats": repeats, "summary": summary,
        "interpretation": "Exact reference-size sensitivity only. Stored sampler endpoints retain 16 replicas, so their finite-sample error does not vanish as the reference size grows."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("results/stage456_validation/year_ridge"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.pilot, args.output)


if __name__ == "__main__":
    main()
