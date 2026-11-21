"""Rebuild the phase-one report and functional-family tables from saved results."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sgld_cv.protocol import verify_lock, sha256
from sgld_cv.workflow import save_csv, save_json

ROOT = Path(__file__).resolve().parents[1]


def family(name):
    if name.startswith("coordinate_mean"):
        return "coordinate_means"
    if name.startswith("coordinate_variance"):
        return "coordinate_variances"
    if name.startswith("prediction"):
        return "prediction_probabilities"
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=ROOT/"results/phase1_validation_final")
    parser.add_argument("--audit", type=Path, default=ROOT/"results/phase1_history_audit")
    args = parser.parse_args()
    manifest = json.loads((args.validation/"manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("cannot report incomplete validation as completed")
    lock = json.loads((args.validation/"protocol.lock.json").read_text())
    verify_lock(lock, ROOT)
    audit = json.loads((args.audit/"audit.json").read_text())
    family_rows = []
    path_errors = []
    for target in manifest["targets"]:
        folder = args.validation/target
        checks = json.loads((folder/"checks.json").read_text())
        path_errors.extend(c["cv2_ula_max_path_error"] for c in checks if "cv2_ula_max_path_error" in c)
        groups = defaultdict(list)
        with (folder/"functional_rmse.csv").open() as stream:
            for row in csv.DictReader(stream):
                groups[(row["arm"], row["seed"], row["iteration"], family(row["functional"]))].append(row)
        for (arm, seed, iteration, label), rows in groups.items():
            valid = all(r["status"] == "ok" for r in rows)
            values = [float(r["normalized_rmse"]) for r in rows] if valid else []
            family_rows.append({"target": target, "arm": arm, "seed": seed, "iteration": iteration,
                "family": label, "functional_count": len(rows), "status": "ok" if valid else "failed_ensemble",
                "mean_normalized_rmse": float(np.mean(values)) if valid else None,
                "max_normalized_rmse": float(max(values)) if valid else None,
                "total_ensemble_seconds": rows[0]["total_ensemble_seconds"],
                "cost_scope": "sequential ensemble, setup once; not per-chain cost"})
    save_csv(args.validation/"functional_family_rmse.csv", family_rows)
    derivative_error = manifest["formula_checks"]["derivative_max_error"]
    enum_error = max(v["exact_enumeration_max_error"] for v in manifest["formula_checks"].values() if isinstance(v, dict))
    table = "\n".join(f"| {name} | {r['N']} | {r['d']} | {r['arms']} | {r['seeds']} | {r['status']} |"
                       for name, r in manifest["targets"].items())
    report = f"""# Phase 1 validation and audit report

Status: completed for implementation audit and small validation only.
Date: 2026-10-14. CPU float64; one BLAS thread. No Phase 2 search executed.

## Correctness evidence

| Target | N | d | Arms | Validation seeds | Status |
|---|---:|---:|---:|---:|---|
{table}

- {sum(r['arms']*r['seeds'] for r in manifest['targets'].values())} target/arm/seed runs completed with independent mode-started replicas.
- Exact finite-batch enumeration max gradient bias: {enum_error:.3g}.
- Logistic finite-difference gradient max discrepancy: {derivative_error:.3g}.
- Quadratic paired CV2/ULA max path discrepancy: {max(path_errors):.3g}.
- All final operation counters reconciled exactly with setup + sampling phases.
- {manifest['historical_seed_inventory_count']} declared legacy seeds inspected; new seed roles are disjoint.
- Pilot, selection and heldout seeds remain unconsumed.
- Protocol, numerical sources and target/reference assets pass SHA-256 lock verification.

Tests additionally verify that online functional averages match full saved
trajectories without changing RNG, posterior variances use the specified
moment estimator, RMSE includes both bias and replica variance, and failed
replicas remain invalid rather than being silently dropped.

## Historical cost explanations

All {audit['historical_records_reconciled']} saved logistic/German records reconcile.
2D CV1/CV2 used batch 8, while pCV1/pCV2 used batch 32. The fourfold difference
in sampling gradient counts comes from batch size, not fourfold duplicate
gradient evaluation. Their original endpoint evaluation already used
independent replicas from the frozen mode; it was underspecified in the summary.

German mode optimization accounts for {audit['German_optimizer_hessians']} per-observation
Hessian contributions. CV2 cache or a full-H metric contributes a further 1000;
CV2 reuses the cache for its metric. No fresh Hessians are evaluated per step.

See `../phase1_history_audit/historical_cost_reconciliation.csv` and
`../phase1_history_audit/matched_gradient_kernel_benchmark.csv`. The latter
compares matched states/batches with and without instrumentation, excluding
setup and sampler work. It is an implementation audit, not an efficiency ranking.

## Saved outputs

Each target folder contains frozen data/geometry/reference assets, functional
targets/scales and reference uncertainty, gradient MSE, endpoint SW2 (plus
low-dimensional POT W2), per-functional RMSE, per-phase operation counts,
replica endpoints/online averages, full timing snapshots and validation checks.
`functional_family_rmse.csv` separately summarizes means and worst functional
within each family. Functional evaluation is excluded from sampler timing.

## Interpretation and remaining stages

The validation uses 32 replicas and 64 transitions, not publication budgets.
Nonzero accuracy errors and slow adaptive exploration at conservative smoke
steps are expected; no convergence or method superiority is inferred.

HMC references remain approximate/correlated. Their diagnostics, independent
agreement, functional MCSE and disagreement are recorded. Reference-reference
distances are neither lower bounds nor bias-correction terms.

Cost rows cover complete sequential ensembles with setup charged once. They
cannot be divided by replica count to claim single-chain first-use cost. ESS
denominators and failure rules are tested; full overdispersed diagnostic
campaigns belong to the next-stage locked target configurations.

Next: German frontier pilot, then a large non-quadratic target and controlled
synthetic regime study. Resolve/freeze production adaptive step grids, accuracy
thresholds and chain budgets before selection seeds. No N-scaling experiment,
full theory proof revision, GPU backend or manuscript rewrite is claimed here.
"""
    (args.validation/"REPORT.md").write_text(report)
    save_json(args.validation/"report_provenance.json", {
        "reporter_sha256": sha256(Path(__file__)),
        "manifest_sha256": sha256(args.validation/"manifest.json"),
        "lock_sha256": sha256(args.validation/"protocol.lock.json"),
        "historical_audit_sha256": sha256(args.audit/"audit.json"),
        "family_rows": len(family_rows)})
    print(f"Report rebuilt; frozen assets verified: {args.validation/'REPORT.md'}")


if __name__ == "__main__":
    main()
