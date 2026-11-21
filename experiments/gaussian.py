"""Run paired Gaussian replicas. Evaluate OT only after sampling timers stop.

Run from repository root: python experiments/gaussian.py --config ... --output ...
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

from sgld_cv import supplied_anchor, prepare_estimator, run_replicas
from sgld_cv.gaussian import make_gaussian_target, analytic_ula_w2
from sgld_cv.randomness import generator, REFERENCE, BASELINE, TIMING_ORDER
from sgld_cv.transport import empirical_w2


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_config(c):
    for key in ["N", "batch_size", "steps", "replicas", "reference_size", "baseline_repeats", "checkpoint_count"]:
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["batch_size"] > c["N"] or c["N"] <= max(c["dimensions"]):
        raise ValueError("require batch_size <= N and N > every d")
    if not 0 < c["step_fraction"] < 4:
        raise ValueError("step_fraction=h*lambda_max(Q) must lie in (0,4)")
    for name in ["dimensions", "kappas", "seeds", "methods"]:
        if not c[name] or len(c[name]) != len(set(c[name])):
            raise ValueError(f"{name} must be nonempty and unique")
    if not set(c["methods"]).issubset({"sgld", "cv1", "cv2", "ula"}):
        raise ValueError("only stage 1-3 methods implemented")
    if c["replicas"] * c["reference_size"] > 4_000_000:
        raise ValueError("configuration exceeds exact-OT memory budget")


def run_experiment(config, output):
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Never silently overwrite evidence.
    checkpoints = np.unique(np.r_[0, np.geomspace(1, config["steps"], config["checkpoint_count"]).astype(int), config["steps"]])
    rows, setups, baselines, controls, failures = [], [], [], [], []
    metadata = {"config": config, "python": sys.version, "platform": platform.platform(),
        "processor": platform.processor(), "versions": {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "matplotlib", "POT", "sgld-cv-research"]},
        "thread_environment": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"]},
        "timing_scope": "entire sequential replica collection; setup once per method/seed; evaluation excluded",
        "initialization": "exact known zero mode; no optimization cost for any method",
        "reference": "independent exact Gaussian draws; finite-reference W2 baseline not subtracted",
        "precision": "float64", "data_generation_seconds": {}, "evaluation_seconds": 0.0,
        "targets": [], "method_orders": [], "status": "running"}
    root = Path(__file__).resolve().parents[1]
    source_paths = [*sorted((root / "src/sgld_cv").glob("*.py")), Path(__file__).resolve()]
    metadata["source_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    write_json(output / "manifest.json", metadata)
    for d in config["dimensions"]:
        for kappa in config["kappas"]:
            key = f"d{d}_k{kappa:g}"
            start = perf_counter()
            target = make_gaussian_target(N=config["N"], d=d, kappa=kappa, seed=config["dataset_seed"])
            metadata["data_generation_seconds"][key] = perf_counter() - start
            model = target.model
            h = config["step_fraction"] / np.linalg.eigvalsh(target.precision)[-1]
            target.metadata["data_sha256"] = hashlib.sha256(model.X.tobytes() + model.y.tobytes()).hexdigest()
            metadata["targets"].append(target.metadata)
            np.savez_compressed(output / f"{key}_target.npz", X=model.X, y=model.y,
                                covariance=target.covariance, precision=target.precision, mean=target.mean)
            anchor = supplied_anchor(np.zeros(d), provenance="exact supplied Gaussian benchmark mode")
            analytic = analytic_ula_w2(target, h, checkpoints)
            for k, w in zip(checkpoints, analytic):
                controls.append({"d": d, "kappa": kappa, "iteration": int(k), "step_size": h,
                                 "analytic_ula_w2": float(w)})
            for seed in config["seeds"]:
                t_eval = perf_counter()
                reference = target.sample(config["reference_size"], generator(seed, 0, REFERENCE))
                baseline_samples = []
                for repetition in range(config["baseline_repeats"]):
                    x = target.sample(config["replicas"], generator(seed, 2 * repetition, BASELINE))
                    y = target.sample(config["reference_size"], generator(seed, 2 * repetition + 1, BASELINE))
                    baseline_samples.extend([x, y])
                    baselines.append({"d": d, "kappa": kappa, "seed": seed, "repeat": repetition,
                                      "w2": empirical_w2(x, y)})
                metadata["evaluation_seconds"] += perf_counter() - t_eval
                np.savez_compressed(output / f"{key}_seed{seed}_reference.npz", reference=reference,
                                    **{f"baseline_{i}": x for i, x in enumerate(baseline_samples)})
                order = generator(seed, 0, TIMING_ORDER).permutation(config["methods"]).tolist()
                metadata["method_orders"].append({"d": d, "kappa": kappa, "seed": seed, "order": order})
                for method in order:
                    print(f"d={d} kappa={kappa:g} seed={seed} method={method}", flush=True)
                    estimator = prepare_estimator(model, method, anchor)
                    setup = {"d": d, "kappa": kappa, "seed": seed, "method": method,
                             "cache_bytes": estimator.cache_bytes, "anchor_provenance": anchor.metadata["source"]}
                    setup.update({f"seconds_{k}": v for k, v in estimator.setup_cost.seconds.items()})
                    setup.update({f"operations_{k}": v for k, v in estimator.setup_cost.operations.items()})
                    setup["setup_seconds"] = sum(estimator.setup_cost.seconds.values())
                    setups.append(setup)
                    run = run_replicas(estimator, initial=anchor.theta, step_size=h,
                        batch_size=config["batch_size"], steps=config["steps"], replicas=config["replicas"],
                        seed=seed, checkpoints=checkpoints)
                    stem = f"{key}_seed{seed}_{method}"
                    np.savez_compressed(output / f"{stem}.npz", checkpoints=run.checkpoints, states=run.states)
                    write_json(output / f"{stem}_costs.json", {"metadata": run.metadata, "costs": run.costs, "failures": run.failures})
                    failures.extend({"d": d, "kappa": kappa, "seed": seed, "method": method, **f} for f in run.failures)
                    t_eval = perf_counter()
                    for k, states, cost in zip(run.checkpoints, run.states, run.costs):
                        # Never drop failed replicas: that would change the law being measured.
                        row = {"d": d, "kappa": kappa, "rho": target.rho, "seed": seed, "method": method,
                               "iteration": int(k), "step_size": h, "batch_size": config["batch_size"],
                               "replicas": config["replicas"], "reference_size": config["reference_size"],
                               "total_seconds": cost["total_seconds"], "status": "ok"}
                        try:
                            row["w2"] = empirical_w2(states, reference)
                            row["w2_over_sqrt_d"] = row["w2"] / np.sqrt(d)
                            row["mean_error"] = float(np.linalg.norm(states.mean(axis=0) - target.mean))
                            row["covariance_frobenius_error"] = float(np.linalg.norm(np.cov(states, rowvar=False, ddof=0) - target.covariance))
                        except (ValueError, FloatingPointError, RuntimeError, Warning) as exc:
                            row.update({"w2": "", "w2_over_sqrt_d": "", "status": f"unavailable: {exc}"})
                        row.update({f"operations_{name}": value for name, value in cost["operations"].items()})
                        rows.append(row)
                    metadata["evaluation_seconds"] += perf_counter() - t_eval
                # Persist progress after each complete paired seed group.
                write_csv(output / "metrics.csv", rows)
                write_csv(output / "setup_costs.csv", setups)
                write_csv(output / "reference_baselines.csv", baselines)
                write_csv(output / "analytic_controls.csv", controls)
                write_json(output / "failures.json", failures)
                write_json(output / "manifest.json", metadata)
    metadata["status"] = "complete"
    write_json(output / "manifest.json", metadata)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/gaussian_smoke.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_experiment(json.loads(args.config.read_text()), args.output)
