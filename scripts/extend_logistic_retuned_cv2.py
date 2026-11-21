"""Targeted CV2 replica/horizon extension after guarded step-size tuning."""
import argparse
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
from sgld_cv.workflow import save_csv, save_json


def run(scan_file, old_folder, reference_file, output, *, seeds=(124, 125),
        replicas=768, reference_size=3072, require_prefix=True,
        step_scale=None):
    scan_file, old_folder, reference_file, output = map(
        Path, (scan_file, old_folder, reference_file, output))
    output.mkdir(parents=True, exist_ok=False)
    scan = json.loads(scan_file.read_text())
    pilot = Path(scan["source_pilot"])
    config = json.loads((pilot / "manifest.json").read_text())["config"]
    model, _ = synthetic_logistic(**{k: v for k, v in config["problem"].items() if k != "kind"})
    with np.load(pilot / "geometry.npz") as data:
        theta, H = data["anchor"], data["hessian"]
    settings = json.loads((pilot / "frozen_selection.json").read_text())["selection"]
    selected = next(row for row in scan["results"]["cv2"]
        if row["scale"] == scan["provisional_largest_passing_scale"]["cv2"])
    if step_scale is None:
        step_scale = selected["scale"]
    if require_prefix and step_scale != selected["scale"]:
        raise ValueError("changed step size cannot match saved trajectory prefix")
    step_size = settings["cv2"]["step_size"] * step_scale
    estimator = prepare_estimator(model, "cv2",
        supplied_anchor(theta, provenance="saved pilot mode"))
    with np.load(reference_file) as data:
        first = data["first"].reshape(-1, model.d)
        longer = data["longer"].reshape(-1, model.d)
    if reference_size > len(longer) or replicas > len(first):
        raise ValueError("requested sample size exceeds HMC pool")
    refmeta = json.loads(Path(scan["reference_diagnostics"]).read_text())
    mean, covariance = np.asarray(refmeta["reference_mean"]), np.asarray(refmeta["reference_covariance"])
    whiten = np.linalg.cholesky(H)
    metric = make_accuracy_metric(model.d, kind="w2")
    checkpoints = [0, 2048, 4096, 8192, 16384]
    rows, baselines = [], []
    with threadpool_limits(limits=1):
        for seed in seeds:
            ref = longer[generator(seed, 0, 84).permutation(len(longer))[:reference_size]]
            order = generator(seed, 1, 85).permutation(len(first))
            for r in sorted({384, replicas}):
                baselines.append({"seed": seed, "replicas": r,
                    "reference_size": reference_size,
                    "reference_reference_w2": metric.distance(first[order[:r]], ref)})
            result = run_replicas(estimator, initial=theta,
                step_size=step_size,
                batch_size=settings["cv2"]["batch_size"], steps=16384,
                replicas=replicas, seed=seed, checkpoints=checkpoints)
            if result.failures:
                raise RuntimeError(f"CV2 seed {seed} failed")
            state = dict(zip(result.checkpoints, result.states))
            if require_prefix:
                with np.load(old_folder / f"states_cv2_seed{seed}.npz") as data:
                    previous = dict(zip(data["checkpoints"], data["states"]))
                    for k in (0, 2048, 4096, 8192):
                        np.testing.assert_array_equal(state[k][:384], previous[k])
            np.savez_compressed(output / f"states_cv2_seed{seed}.npz",
                checkpoints=result.checkpoints, states=result.states)
            for k in checkpoints[1:]:
                for r in sorted({384, replicas}):
                    x = state[k][:r]
                    rows.append({"method": "cv2", "seed": seed, "iteration": k,
                        "replicas": r, "reference_size": reference_size,
                        "accuracy_w2": metric.distance(x, ref),
                        "mean_error_std": float(np.linalg.norm((x.mean(axis=0)-mean) @ whiten)
                            / np.sqrt(model.d)),
                        "covariance_error_std": float(np.linalg.norm(whiten.T @
                            (np.cov(x, rowvar=False, ddof=0)-covariance) @ whiten) / model.d)})
            save_csv(output / "scores.csv", rows)
            save_csv(output / "reference_baseline.csv", baselines)
            save_json(output / "progress.json", {"source_scan": str(scan_file),
                "source_first_384": str(old_folder),
                "source_reference": str(reference_file),
                "first_384_replicas_match_exactly": True if require_prefix else None,
                "seeds": seeds, "replicas": replicas,
                "step_scale": step_scale, "step_size": step_size,
                "reference_size": reference_size, "checkpoints": checkpoints,
                "completed_seeds": [s for s in seeds if
                    (output / f"states_cv2_seed{s}.npz").exists()],
                "interpretation": "Targeted CV2 new-step extension; no final K declaration."})
            print("cv2", seed, "done", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path,
        default=Path("results/logistic_step_scan_guarded/scan.json"))
    parser.add_argument("--old-folder", type=Path,
        default=Path("results/logistic_retuned_horizon"))
    parser.add_argument("--reference", type=Path,
        default=Path("results/logistic_reference_refined/reference_chains.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[124, 125])
    parser.add_argument("--replicas", type=int, default=768)
    parser.add_argument("--step-scale", type=float)
    parser.add_argument("--no-prefix", action="store_true",
        help="use new seeds without a saved R=384 prefix")
    args = parser.parse_args()
    run(args.scan, args.old_folder, args.reference, args.output,
        seeds=tuple(args.seeds), replicas=args.replicas,
        require_prefix=not args.no_prefix, step_scale=args.step_scale)


if __name__ == "__main__":
    main()
