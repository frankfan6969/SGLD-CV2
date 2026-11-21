"""Nested replica/reference-size experiment for the saved 2D logistic pilot."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.anchors import supplied_anchor
from sgld_cv.data import synthetic_logistic
from sgld_cv.estimators import prepare_estimator
from sgld_cv.randomness import generator
from sgld_cv.samplers import run_replicas
from sgld_cv.transport import make_accuracy_metric
from sgld_cv.workflow import METHODS, save_csv, save_json


def run(pilot, old_probe, output, *, seeds=(122, 123), replica_sizes=(96, 192, 384),
        reference_sizes=(96, 192, 384), steps=4096, bootstraps=48):
    pilot, old_probe, output = map(Path, (pilot, old_probe, output))
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((pilot / "manifest.json").read_text())["config"]
    model, _ = synthetic_logistic(**{k: v for k, v in config["problem"].items() if k != "kind"})
    with np.load(pilot / "geometry.npz") as geometry:
        theta, H = geometry["anchor"], geometry["hessian"]
    anchor = supplied_anchor(theta, provenance="saved pilot mode")
    settings = json.loads((pilot / "frozen_selection.json").read_text())["selection"]
    pilots = json.loads((pilot / "metric_pilots.json").read_text())
    with np.load(old_probe / "reference_chains.npz") as data:
        first = data["first"].reshape(-1, model.d)
        longer = data["longer"].reshape(-1, model.d)
    ref_mean = longer.mean(axis=0)
    ref_cov = np.cov(longer, rowvar=False)
    moment_transform = np.linalg.cholesky(H)
    metric = make_accuracy_metric(model.d, kind="w2")
    checkpoints = np.array([0, 1024, 2048, 3072, steps])
    rows = []
    saved = {}
    references = {}
    with threadpool_limits(limits=1):
        for seed in seeds:
            # Draw one permutation per seed, so every reference-size comparison
            # is nested and shares the same underlying independent HMC pool.
            index = generator(seed, 0, 84).permutation(len(longer))
            references[seed] = {s: longer[index[:s]] for s in reference_sizes}
        for method in METHODS:
            setting = settings[method]
            batch = setting["batch_size"]
            base = method[1:] if method.startswith("p") else method
            options = {}
            if method.startswith("p"):
                options = {"preconditioner": "rmsprop",
                    "initial_v": np.array(pilots[str(batch)]["v"]),
                    "alpha": config["metric"].get("alpha", .99),
                    "epsilon": config["metric"].get("epsilon", 1e-5)}
            estimator = prepare_estimator(model, base, anchor)
            for seed in seeds:
                trajectory = run_replicas(estimator, initial=theta,
                    step_size=setting["step_size"], batch_size=batch,
                    steps=steps, replicas=max(replica_sizes), seed=seed,
                    checkpoints=checkpoints, **options)
                if trajectory.failures:
                    raise RuntimeError(f"{method} seed {seed} had failed replicas")
                saved[(method, seed)] = dict(zip(trajectory.checkpoints, trajectory.states))
                with np.load(old_probe / f"probe_{method}_seed{seed}.npz") as old:
                    original = dict(zip(old["checkpoints"], old["states"]))
                    for k in checkpoints:
                        np.testing.assert_array_equal(saved[(method, seed)][k][:96], original[k])
                np.savez_compressed(output / f"states_{method}_seed{seed}.npz",
                    states=trajectory.states, checkpoints=trajectory.checkpoints)
                for k in checkpoints[1:]:
                    for r in replica_sizes:
                        x = saved[(method, seed)][k][:r]
                        mean_error = np.linalg.norm((x.mean(axis=0) - ref_mean) @ moment_transform) / np.sqrt(model.d)
                        cov_error = np.linalg.norm(moment_transform.T @
                            (np.cov(x, rowvar=False, ddof=0) - ref_cov) @ moment_transform) / model.d
                        for s in reference_sizes:
                            rows.append({"method": method, "seed": seed, "iteration": int(k),
                                "replicas": r, "reference_size": s,
                                "accuracy": metric.distance(x, references[seed][s]),
                                "mean_error_std": mean_error,
                                "covariance_error_std": cov_error})
                save_csv(output / "size_scores.csv", rows)
        baselines = []
        for seed in seeds:
            ref_index = generator(seed, 1, 85).permutation(len(first))
            for s in reference_sizes:
                baselines.append({"seed": seed, "reference_size": s,
                    "reference_reference_accuracy": metric.distance(first[ref_index[:s]], references[seed][s])})
        save_csv(output / "reference_baselines.csv", baselines)

        rng = np.random.default_rng(2718)
        bootstrap = {}
        max_r, max_s = max(replica_sizes), max(reference_sizes)
        for method in METHODS:
            point = []
            for seed in seeds:
                states = saved[(method, seed)]
                ref = references[seed][max_s]
                point.append(metric.distance(states[4096][:max_r], ref) -
                             metric.distance(states[2048][:max_r], ref))
            resamples = []
            for _ in range(bootstraps):
                changes = []
                for seed in seeds:
                    states = saved[(method, seed)]
                    ref = references[seed][max_s]
                    index = rng.integers(max_r, size=max_r)
                    changes.append(metric.distance(states[4096][index], ref) -
                                   metric.distance(states[2048][index], ref))
                resamples.append(np.mean(changes))
            bootstrap[method] = {"mean_change": float(np.mean(point)),
                "conditional_q05": float(np.quantile(resamples, .05)),
                "conditional_q95": float(np.quantile(resamples, .95))}
    save_json(output / "summary.json", {"target": config["problem"],
        "source_selection": str(pilot / "frozen_selection.json"),
        "source_reference": str(old_probe / "reference_chains.npz"),
        "seeds": seeds, "replica_sizes": replica_sizes,
        "reference_sizes": reference_sizes, "steps": steps,
        "prefix_matches_previous_96_replicas_exactly": True,
        "bootstraps": bootstraps, "paired_change_2048_to_4096_at_max_sizes": bootstrap,
        "interpretation": "Nested size sensitivity on inspected diagnostic seeds. Bootstrap intervals condition on reference pools and seeds, so they do not certify a final horizon."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path,
        default=Path("results/stage456_validation/synthetic_logistic"))
    parser.add_argument("--probe", type=Path,
        default=Path("results/logistic_horizon_tuned_4096_diagnostic"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.pilot, args.probe, args.output)


if __name__ == "__main__":
    main()
