"""Reconcile historical counts and measure matched CV implementation overhead.

Read-only with respect to all historical results. No sampler tuning or frontier
selection. Benchmarks are local implementation checks, not algorithm rankings.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.anchors import find_anchor
from sgld_cv.costs import CostLedger
from sgld_cv.data import german_credit
from sgld_cv.estimators import prepare_estimator
from sgld_cv.protocol import sha256
from sgld_cv.workflow import save_json, save_csv

ROOT = Path(__file__).resolve().parents[1]


def reconcile(folder, dataset):
    rows = []
    for path in sorted(folder.glob("result_*.json")):
        record = json.loads(path.read_text())
        first, last = record["checkpoints"][0], record["checkpoints"][-1]
        initial = first["cost"]["operations"]
        final = last["cost"]["operations"]
        expected = initial.get("observation_gradient_evaluations", 0) + record["replicas"] * last["iteration"] * record["batch_size"]
        if expected != final["observation_gradient_evaluations"]:
            raise AssertionError(f"historical gradient count mismatch: {path}")
        rows.append({"dataset": dataset, "file": str(path.relative_to(ROOT)), "sha256": sha256(path),
            "method": record["method"], "arm": record.get("arm", record["method"]), "seed": record["seed"],
            "batch_size": record["batch_size"], "replicas": record["replicas"], "K": last["iteration"],
            "setup_gradients": initial.get("observation_gradient_evaluations", 0),
            "sampling_gradients": record["replicas"] * last["iteration"] * record["batch_size"],
            "observed_total_gradients": final["observation_gradient_evaluations"],
            "setup_hessians": initial.get("observation_hessian_evaluations", 0),
            "sampling_hessians": final.get("observation_hessian_evaluations", 0)-initial.get("observation_hessian_evaluations", 0),
            "gradient_count_reconciled": True,
            "initialization": "frozen mode; independent replica streams (verified runner source)",
            "evaluation_object": "finite-K replica endpoints"})
    return rows


def benchmark(model, anchor):
    rng = np.random.default_rng(141901)
    batch_size, loops, repeats = 32, 512, 5
    batches = [rng.choice(model.N, batch_size, replace=False) for _ in range(loops)]
    perturbations = rng.normal(size=(loops, model.d)) @ np.linalg.cholesky(np.linalg.inv(model.full_hessian(anchor.theta))).T
    states = anchor.theta + perturbations
    estimators = {m: prepare_estimator(model, m, anchor) for m in ("cv1", "cv2")}
    rows = []
    for recorded in (False, True):
        timings = {m: [] for m in estimators}
        for repeat in range(repeats):
            for method in rng.permutation(list(estimators)):
                est = estimators[str(method)]
                ledger = CostLedger() if recorded else None
                start = perf_counter()
                for theta, idx in zip(states, batches):
                    est.gradient(theta, idx, ledger)
                timings[str(method)].append(perf_counter()-start)
        for method, seconds in timings.items():
            rows.append({"method": method, "instrumented": recorded, "N": model.N, "d": model.d,
                "batch_size": batch_size, "calls_per_repeat": loops, "repeats": repeats,
                "median_seconds": float(np.median(seconds)), "min_seconds": float(min(seconds)),
                "max_seconds": float(max(seconds)), "scope": "gradient kernel only; matched states/batches; excludes setup and sampler"})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"results/phase1_history_audit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = reconcile(ROOT/"results/logistic_postdiagnostic_confirmation", "2D logistic")
    rows += reconcile(ROOT/"results/german_frozen_geometry_confirmation", "German")
    if len(rows) != 96:
        raise AssertionError(f"expected 56 logistic and 40 German records, got {len(rows)}")
    with threadpool_limits(limits=1):
        model, _ = german_credit(ROOT/"data/raw/german.zip")
        anchor = find_anchor(model)
        optimizer_hessians = anchor.cost.operations["observation_hessian_evaluations"]
        german_rows = [r for r in rows if r["dataset"] == "German"]
        for row in german_rows:
            additional = model.N if row["method"] == "cv2" or row["arm"].startswith("full_") else 0
            if row["setup_hessians"] != optimizer_hessians + additional or row["sampling_hessians"] != 0:
                raise AssertionError("German Hessian setup not explained by optimizer plus cache/metric")
        timings = benchmark(model, anchor)
    save_csv(args.output/"historical_cost_reconciliation.csv", rows)
    save_csv(args.output/"matched_gradient_kernel_benchmark.csv", timings)
    save_json(args.output/"audit.json", {
        "status": "complete", "historical_records_reconciled": len(rows),
        "two_dimensional_batch_sizes": {m: sorted({r["batch_size"] for r in rows if r["dataset"] == "2D logistic" and r["method"] == m})
                                       for m in ("sgld", "cv1", "cv2", "psgld", "pcv1", "pcv2")},
        "German_optimizer_hessians": optimizer_hessians,
        "German_hessian_rule": "optimizer Hessians plus N for CV2 cache OR full-H metric; reuse prevents double charge; no fresh Hessians per step",
        "two_dimensional_evaluation": "runner uses run_replicas(initial=frozen mode); endpoints, not pooled chain draws",
        "timing_scope": "one-thread float64 CPU; gradient-kernel audit only; no efficiency superiority inference",
        "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in [
            ROOT/"scripts/run_logistic_fixed8192_heldout.py", ROOT/"scripts/calibrate_german_geometry_pairs.py",
            ROOT/"scripts/audit_phase1_history.py"]},
        "historical_results_modified": False})
    print(f"Reconciled {len(rows)} historical records; optimizer Hessians={optimizer_hessians}.")


if __name__ == "__main__":
    main()
