"""Small, matched-pair German Credit step/geometry feasibility calibration."""
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.anchors import find_anchor
from sgld_cv.costs import CostLedger
from sgld_cv.data import german_credit
from sgld_cv.diagnostics import chain_diagnostics
from sgld_cv.estimators import prepare_estimator
from sgld_cv.randomness import generator
from sgld_cv.samplers import run_chains
from sgld_cv.transport import load_accuracy_metric, metric_fingerprint
from sgld_cv.workflow import save_json


PILOT = Path("results/german_sliced_pilot")
REFERENCE = Path("results/german_sliced_heldout_long_reference")
OUTPUT = Path("results/german_targeted_calibration")
SEEDS = (110, 111)
BATCH_SIZE = 32
WARMUP = 1000
DRAWS = 4000
CHAINS = 4
REFERENCE_SIZE = 512
IDENTITY_STEPS = (0.0005288347054977375, 0.001057669410995475)
FULL_STEPS = (0.1, 0.3)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def full_metric(model, estimator, saved_hessian):
    cost = CostLedger()
    if estimator.method == "cv2":
        hessian = estimator.hessian_hat + model.prior_precision * np.eye(model.d)
        np.testing.assert_allclose(hessian, saved_hessian, rtol=0, atol=1e-9)
    else:
        with cost.measure("full_metric_hessian"):
            hessian = model.full_hessian(estimator.theta_hat)
            cost.count("observation_hessian_evaluations", model.N)
            cost.count("dense_hessian_assemblies")
            cost.count("data_row_visits", model.N)
        np.testing.assert_allclose(hessian, saved_hessian, rtol=0, atol=1e-9)
    with cost.measure("full_metric_inverse"):
        inverse = np.linalg.solve(hessian, np.eye(model.d))
        inverse = (inverse + inverse.T) / 2
    return inverse, cost


def main():
    frozen = json.loads((PILOT / "frozen_selection.json").read_text())
    refmeta = json.loads((REFERENCE / "reference_diagnostics.json").read_text())
    if not refmeta["reliable"]:
        raise ValueError("German reference failed validation")
    metric = load_accuracy_metric(PILOT / "accuracy_calibration.npz", d=61)
    fingerprint = metric_fingerprint(metric)
    if fingerprint != frozen["accuracy_metric"]["calibration_sha256"]:
        raise ValueError("German scoring calibration fingerprint changed")
    problem = json.loads((PILOT / "manifest.json").read_text())["config"]["problem"]
    model, source = german_credit(**{k: v for k, v in problem.items() if k != "kind"})
    if model.d != 61:
        raise ValueError("German design dimension changed")
    anchor = find_anchor(model)
    with np.load(PILOT / "geometry.npz") as data:
        np.testing.assert_allclose(anchor.theta, data["anchor"], rtol=0, atol=1e-9)
        saved_hessian = np.asarray(data["hessian"])
    hessian_inverse = np.linalg.solve(saved_hessian, np.eye(model.d))
    initial_cholesky = np.linalg.cholesky((hessian_inverse + hessian_inverse.T) / 2)
    with np.load(REFERENCE / "reference_chains.npz") as data:
        reference_pool = data["longer"].reshape(-1, model.d)
    target_mean = np.asarray(refmeta["reference_mean"])
    target_cov = np.asarray(refmeta["reference_covariance"])
    spec = {"purpose": "small_geometry_pair_calibration", "pilot": str(PILOT),
            "reference": str(REFERENCE), "pilot_geometry_sha256": digest(PILOT / "geometry.npz"),
            "reference_chains_sha256": digest(REFERENCE / "reference_chains.npz"),
            "accuracy_calibration_sha256": fingerprint, "data_sha256": source["source_sha256"],
            "seeds": SEEDS, "batch_size": BATCH_SIZE, "warmup": WARMUP,
            "draws_per_chain": DRAWS, "chains": CHAINS, "reference_size": REFERENCE_SIZE,
            "identity_steps": IDENTITY_STEPS, "full_steps": FULL_STEPS,
            "initialization": "independent 2x mode-Hessian Gaussian dispersion",
            "scored_draws": REFERENCE_SIZE,
            "interpretation": "exploratory calibration only; shared h and batch within each geometry pair, and same frozen full-Hessian matrix for full-geometry CV1/CV2"}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUTPUT / "geometry_pair_protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != json.loads(json.dumps(spec)):
            raise ValueError("existing German geometry-pair protocol differs")
    else:
        save_json(protocol_path, spec)
    rows = []
    with threadpool_limits(limits=1):
        for geometry, step_sizes in (("identity", IDENTITY_STEPS), ("full_hessian", FULL_STEPS)):
            for h in step_sizes:
                for seed in SEEDS:
                    for method in ("cv1", "cv2"):
                        name = f"{geometry}_{method}_h{h:g}_seed{seed}"
                        row_path = OUTPUT / f"{name}.json"
                        sample_path = OUTPUT / f"{name}.npz"
                        if row_path.exists() and sample_path.exists():
                            rows.append(json.loads(row_path.read_text()))
                            print(name, "already complete", flush=True)
                            continue
                        if row_path.exists() or sample_path.exists():
                            raise ValueError(f"partial calibration output: {name}")
                        estimator = prepare_estimator(model, method, anchor)
                        options = {}
                        if geometry == "full_hessian":
                            matrix, metric_cost = full_metric(model, estimator, saved_hessian)
                            options = {"preconditioner": "frozen_full",
                                       "frozen_matrix": matrix, "metric_cost": metric_cost}
                        initial = anchor.theta + 2 * generator(seed, 0, 83).normal(
                            size=(CHAINS, model.d)) @ initial_cholesky.T
                        samples, run = run_chains(estimator, initial=initial,
                            step_size=h, batch_size=BATCH_SIZE, warmup=WARMUP,
                            draws=DRAWS, chains=CHAINS, seed=seed, **options)
                        if run.failures or not np.isfinite(samples).all():
                            raise RuntimeError(f"failed German calibration: {name}")
                        diag = chain_diagnostics(samples, cost=run.costs[-1])
                        flat = samples.transpose(1, 0, 2).reshape(-1, model.d)
                        score = flat[np.linspace(0, len(flat) - 1, REFERENCE_SIZE).astype(int)]
                        ref = reference_pool[generator(seed, 0, 84).permutation(
                            len(reference_pool))[:REFERENCE_SIZE]]
                        sw2 = metric.distance(score, ref)
                        mean_error = np.linalg.norm((flat.mean(axis=0) - target_mean) @
                                                    metric.transform) / np.sqrt(model.d)
                        cov_error = np.linalg.norm(metric.transform.T @
                            (np.cov(flat, rowvar=False, ddof=0) - target_cov) @
                            metric.transform) / model.d
                        row = {"name": name, "geometry": geometry, "method": method,
                               "step_size": h, "seed": seed, "batch_size": BATCH_SIZE,
                               "warmup": WARMUP, "draws_per_chain": DRAWS,
                               "chains": CHAINS, "reference_size": REFERENCE_SIZE,
                               "sw2": sw2, "standardized_mean_error": float(mean_error),
                               "standardized_covariance_error": float(cov_error),
                               "max_rhat": diag["max_rhat"],
                               "min_bulk_ess": diag["min_bulk_ess"],
                               "min_tail_ess": diag["min_tail_ess"],
                               "diagnostics_reliable": diag["reliable"],
                               "total_seconds": run.costs[-1]["total_seconds"],
                               "setup_seconds": sum(value for key, value in
                                   run.costs[-1]["seconds"].items() if key not in
                                   {"burn_in", "sampling"}),
                               "cost": run.costs[-1], "scoring_calibration_sha256": fingerprint,
                               "interpretation": "exploratory, not a publication ranking"}
                        with sample_path.open("wb") as stream:
                            np.savez_compressed(stream, samples=samples, initial=initial,
                                                scored_draws=score)
                        save_json(row_path, row)
                        rows.append(row)
                        print(name, "Rhat", f"{diag['max_rhat']:.3f}",
                              "cov", f"{cov_error:.3f}",
                              "s", f"{row['total_seconds']:.2f}", flush=True)
    with (OUTPUT / "geometry_pair_summary.csv").open("w", newline="") as stream:
        fields = ["geometry", "method", "step_size", "seed", "sw2",
                  "standardized_mean_error", "standardized_covariance_error",
                  "max_rhat", "min_bulk_ess", "min_tail_ess",
                  "diagnostics_reliable", "total_seconds", "setup_seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)
    lines = ["# German Credit matched-geometry feasibility calibration", "",
        "Two calibration seeds, four overdispersed chains per run, 1,000 warmup and 4,000 retained draws per chain. CV1/CV2 share batch size and h within each geometry condition; the full-Hessian pair uses the same frozen inverse mode Hessian in drift and noise. Frozen pilot whitening/projections and the validated held-out HMC pool are reused.", "",
        "| Geometry | Method | h | Seed | Max R-hat | Min bulk ESS | Mean / cov error | SW2 | Setup-inclusive s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['geometry']} | {row['method']} | {row['step_size']:.6g} | {row['seed']} | {row['max_rhat']:.3f} | {row['min_bulk_ess']:.0f} | {row['standardized_mean_error']:.3f} / {row['standardized_covariance_error']:.3f} | {row['sw2']:.3f} | {row['total_seconds']:.2f} |")
    lines += ["", "These short chains determine feasibility and cost, not a final h. A low SW2 or moment error on a non-mixing chain is not posterior validation. Full-Hessian geometry is charged for the extra Hessian (CV1), inverse, and Cholesky; CV2 reuses its already charged estimator Hessian. All runs use one BLAS thread and a fresh setup per collection.", ""]
    (OUTPUT / "GEOMETRY_PAIRS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
